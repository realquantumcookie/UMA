from typing import Optional, Union, Dict, Any, Tuple, Type, Sequence, List
import copy
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, and_masks, or_masks, BlockMask
from tensordict import TensorDict, TensorClass, NonTensorData
from xbarray.transformations.pointcloud.pytorch import (
    depth_image_to_world,
    random_point_sampling,
)
from xbarray.transformations.rotation_conversions.pytorch import rotation_6d_to_matrix

from unienv_interface.space import BoxSpace
from uma.policy.base import PolicyActor, PolicyFeatureExtractor, PolicyActorConfig
from uma.network.attention import TransformerBlock, FlexAttention
from uma.network.dit import MaskedDiTBlock
from uma.network.dynamic_tanh import DynamicTanh
from uma.network.util import *
from uma.network.positional_embedding import TimestepPosEmb, PosEmb3D, PosEmb2D
from uma.network.pointnet import DP3Encoder, IDP3Encoder, PointNet2Encoder
from uma.network.spatio_temporal_transformer import SpatioTemporalTransformer
from uma.tokenizers.language import (
    EncoderType, get_image_embedding_dim, get_language_embedding_dim, tokenize_languages
)
from uma.tokenizers.image_network import KeypointFeatureExtractor, ImageTokenizer

from .pointcloud_fusion import PointCloudPatchFusion
from .input_output import UMATokenItem, UMAContextItem, UMAImageContext, UMAMaskedTokens, UMAFlowSpecContextItem, UMAFlowSpecInputContext, UMAInputContext, dict_replace, transform_3d, random_se3_augmentation_matrix, action_rotation_to_absolute, action_rotation_to_relative, ACTION_POSE_DIM, safe_rotation_6d_to_matrix, rotation_6d_degenerate
from .specification_encoder import SpecificationEncoder, SPEC_DISPLACEMENT_SCALE, spec_first_valid_anchor

def sample_beta(alpha, beta, bsize : Union[int, Sequence[int]], dtype : torch.dtype = torch.float32, device : Union[torch.device, str] = 'cpu'):
    dist = torch.distributions.Beta(alpha, beta)
    sample_shape = (bsize,) if isinstance(bsize, int) else tuple(bsize)
    return dist.sample(sample_shape).to(dtype=dtype, device=device)

_BETA_DISTRIBUTIONS : Dict[Tuple[float, float, str], torch.distributions.Beta] = {}

def sample_beta_on_device(alpha, beta, bsize : Union[int, Sequence[int]], dtype : torch.dtype = torch.float32, device : Union[torch.device, str] = 'cpu'):
    """:func:`sample_beta` drawn directly on ``device`` (a CPU draw plus copy would sync the host every step)."""
    key = (float(alpha), float(beta), str(device))
    dist = _BETA_DISTRIBUTIONS.get(key)
    if dist is None:
        dist = _BETA_DISTRIBUTIONS[key] = torch.distributions.Beta(
            torch.tensor(float(alpha), device=device), torch.tensor(float(beta), device=device)
        )
    sample_shape = (bsize,) if isinstance(bsize, int) else tuple(bsize)
    return dist.sample(sample_shape).to(dtype=dtype)

def masked_rotation_6d_degenerate_frac(
    d6 : torch.Tensor,
    valid_mask : torch.Tensor,
) -> torch.Tensor:
    """Fraction of the valid 6D rotations that are degenerate (:func:`rotation_6d_degenerate`)."""
    degenerate = rotation_6d_degenerate(d6) & valid_mask
    return degenerate.sum().float() / torch.clamp(valid_mask.sum(), min=1).float()

def masked_rotation_angle_error_deg(
    pred_6d : torch.Tensor,
    target_6d : torch.Tensor,
    valid_mask : torch.Tensor,
) -> torch.Tensor:
    """Mean angle (degrees) between predicted and target 6D rotations, in fp32
    (a bf16 matmul under autocast reads errors of a few degrees as 0)."""
    if any(dim == 0 for dim in pred_6d.shape):
        return pred_6d.sum() * 0.0

    with torch.autocast(device_type=pred_6d.device.type, enabled=False):
        pred_matrix = safe_rotation_6d_to_matrix(pred_6d.float())
        target_matrix = safe_rotation_6d_to_matrix(target_6d.detach().float())
        relative_matrix = torch.matmul(pred_matrix, target_matrix.transpose(-1, -2))
        trace = relative_matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, min=-1.0, max=1.0))
        angle = torch.where(valid_mask, angle, 0.0)
        denominator = torch.clamp(valid_mask.sum(), min=1).float()
        return torch.rad2deg(angle.sum() / denominator)

PLACEHOLDER_TIME_ID = int(-1e3)
PLACEHOLDER_KEYPOINT_ID = int(-1e3)
PRED_T_KNOWN = 1.0

def spec_time_reverse(
    flow_spec_context : UMAFlowSpecInputContext,
) -> UMAFlowSpecInputContext:
    """Same scene, trajectory played backwards — a contrastive hard negative."""
    return dict_replace(
        flow_spec_context,
        flow_spec=torch.flip(flow_spec_context['flow_spec'], dims=[1]),
        valid_mask=torch.flip(flow_spec_context['valid_mask'], dims=[1]),
    )

def spec_freeze(
    flow_spec_context : UMAFlowSpecInputContext,
) -> UMAFlowSpecInputContext:
    """Same scene, no motion at all — a contrastive hard negative."""
    anchor = spec_first_valid_anchor(
        flow_spec_context['flow_spec'],
        flow_spec_context['valid_mask'],
    ) # (B, 1, M_fs, 3)
    return dict_replace(
        flow_spec_context,
        flow_spec=anchor.expand_as(flow_spec_context['flow_spec']).contiguous(),
    )

def spec_mean_displacement(
    flow_spec_context : UMAFlowSpecInputContext,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean (last valid - first valid) displacement over valid keypoints.

    Returns:
        target: (B, 3) mean displacement in the spec's input frame, meters
        target_valid: (B,) whether any keypoint had a valid timestep
    """
    flow_spec = flow_spec_context['flow_spec'] # (B, T_fs, M_fs, 3)
    valid_mask = flow_spec_context['valid_mask'].bool() # (B, T_fs, M_fs)
    T_fs = flow_spec.shape[1]

    first_pos = spec_first_valid_anchor(flow_spec, valid_mask)[:, 0] # (B, M_fs, 3)
    last_valid_idx = (T_fs - 1) - torch.argmax(valid_mask.flip(1).int(), dim=1) # (B, M_fs)
    last_pos = torch.gather(
        flow_spec,
        dim=1,
        index=last_valid_idx[:, None, :, None].expand(-1, 1, -1, flow_spec.shape[-1])
    )[:, 0] # (B, M_fs, 3)

    point_valid = valid_mask.any(dim=1) # (B, M_fs)
    displacement = torch.where(point_valid[..., None], last_pos - first_pos, 0.0)
    target = displacement.sum(dim=1) / torch.clamp(point_valid.sum(dim=1, keepdim=True), min=1) # (B, 3)
    return torch.nan_to_num(target, nan=0.0), point_valid.any(dim=-1)


class NormalizedMeanMotionReadout(nn.Module):
    """Stable low-capacity readout for a specification's net 3D motion.

    Specification tokens are learned aggregation slots rather than individual
    tracked points. The supervision is only the mean first-to-last displacement
    of valid points, so token-wise normalization and mean pooling provide the
    appropriate inductive bias without a high-gain flattened projection.
    """

    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 128,
        max_displacement_m: float = 2.0,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if max_displacement_m <= 0:
            raise ValueError(
                "max_displacement_m must be positive, "
                f"got {max_displacement_m}"
            )

        # Parameter-free LayerNorm: token-wise normalization without learned affine weights.
        self.input_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.head = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.max_scaled_displacement = (
            float(max_displacement_m) * SPEC_DISPLACEMENT_SCALE
        )

        # Begin at a safe zero-motion prediction. This prevents a randomly
        # initialized auxiliary head from perturbing the shared spec encoder.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, spec_tokens: torch.Tensor) -> torch.Tensor:
        if spec_tokens.ndim < 3:
            raise ValueError(
                "spec_tokens must end in (n_spec_tokens, token_dim), got "
                f"shape {tuple(spec_tokens.shape)}"
            )
        normalized = self.input_norm(spec_tokens)
        pooled = normalized.mean(dim=-2)
        raw_prediction = self.head(pooled)
        bound = self.max_scaled_displacement
        # Radially squash the vector so its norm, rather than each coordinate,
        # is bounded. Clamping the norm before both tanh and division preserves
        # an identity derivative at the zero initialization.
        safe_norm = torch.clamp(
            raw_prediction.norm(dim=-1, keepdim=True),
            min=torch.finfo(raw_prediction.dtype).eps,
        )
        scale = bound * torch.tanh(safe_norm / bound) / safe_norm
        return raw_prediction * scale

class UMANormalizedMaskedTokens(UMAMaskedTokens):
    action_mean : torch.Tensor # Shape (B, 9): current position + current rotation anchor
    flow_mean : torch.Tensor # Shape (B, M, action_dim)

    @staticmethod
    def _clip_output(
        masked_tokens : "UMANormalizedMaskedTokens"
    ) -> "UMANormalizedMaskedTokens":
        return dict_replace(
            masked_tokens,
            action=dict_replace(
                masked_tokens['action'],
                token_data=UMATokenItem._clip_normalize(
                    masked_tokens['action']['token_data']
                )
            ),
            flow=dict_replace(
                masked_tokens['flow'],
                token_data=UMATokenItem._clip_normalize(
                    masked_tokens['flow']['token_data']
                )
            )
        )

class UMAContextEncoder(nn.Module):
    @staticmethod
    def stack_tokens(
        token_feature : torch.Tensor,
        token_keypoint_id : torch.Tensor,
        token_valid_mask : torch.Tensor,
        existing_stacked : Optional[Tuple[
            torch.Tensor, # data
            torch.Tensor, # keypoint id
            torch.Tensor, # valid mask
        ]] = None
    ) -> Tuple[ # stacked result
        torch.Tensor, # data
        torch.Tensor, # keypoint id
        torch.Tensor, # valid mask
    ]:
        length = np.prod(token_feature.shape[1:-1])
        assert np.prod(token_valid_mask.shape[1:]) == length
        if existing_stacked is None:
            stacked = (
                token_feature.flatten(1, -2),
                token_keypoint_id.flatten(1, -1),
                token_valid_mask.flatten(1, -1),
            )
            return stacked
        else:
            stacked = (
                torch.concat([existing_stacked[0], token_feature.flatten(1, -2)], dim=1),
                torch.concat([existing_stacked[1], token_keypoint_id.flatten(1, -1)], dim=1),
                torch.concat([existing_stacked[2], token_valid_mask.flatten(1, -1)], dim=1),
            )
            return stacked 

    @staticmethod
    def stack_spec_tokens(
        spec_token : torch.Tensor, # (B, N_spec, D)
        spec_token_valid : torch.Tensor, # (B, )
        existing_stacked : Optional[Tuple[
            torch.Tensor, # data
            torch.Tensor, # keypoint id
            torch.Tensor, # valid mask
        ]] = None
    ) -> Tuple[ # stacked result
        torch.Tensor, # data
        torch.Tensor, # keypoint id
        torch.Tensor, # valid mask
    ]:
        spec_token_valid = spec_token_valid[:, None].expand(-1, spec_token.shape[1]) # (B, 1) -> (B, N_spec)
        keypoint_id = torch.full_like(spec_token_valid, PLACEHOLDER_KEYPOINT_ID, dtype=torch.int32)
        return UMAContextEncoder.stack_tokens(
            spec_token, keypoint_id, spec_token_valid,
            existing_stacked
        )

    def __init__(
        self,
        token_dim : int,
        n_spec_tokens : int,
        num_layers_stt : int,
        num_layers_agg : int,
        num_layers_language : int,
        flow_spec_time_embedding_dim : int = 64,
        flow_spec_dim : int = 3,
        encoder_type : EncoderType = "siglip2",
    ) -> None:
        super().__init__()
        self.token_dim = token_dim

        self.flow_spec_time_embedding_dim = flow_spec_time_embedding_dim
        self.flow_spec_dim = flow_spec_dim
        self.encoder_type = encoder_type

        # Get embedding dimensions for the chosen encoder
        self.language_embedding_dim = get_language_embedding_dim(encoder_type)
        self.image_embedding_dim = get_image_embedding_dim(encoder_type)

        self.flow_spec_time_embedding_dim = flow_spec_time_embedding_dim

        # ========== Initialize Specification Encoder ==========
        # Shared geometry-only PointNet++ backbone.
        self.pointcloud_encoder = PointNet2Encoder(
            d_input=3,
            sa_intermediate_hidden_dim_start=[128, 256, 512],
            sa_d_outputs=[128, 256, 512],
            sa_group_nums=[512, 256, 128],
            sa_group_max_points=[32, 16, 16],
            sa_group_radius=[0.05, 0.1, 0.2],
            reduce_method='max',
            norm_layer=nn.Identity,
            enable_cls_token_layers=False,
            enable_cls_token=False,
        )
        self.pointcloud_fusion = PointCloudPatchFusion(
            point_feature_dim=self.pointcloud_encoder.d_output,
            visual_feature_dim=self.image_embedding_dim,
            token_dim=token_dim,
        )
        point_feature_dim = self.pointcloud_encoder.d_output
        self.specification_encoder = SpecificationEncoder(
            token_dim,
            n_spec_tokens=n_spec_tokens,
            num_layers_stt=num_layers_stt,
            num_layers_agg=num_layers_agg,
            num_layers_language=num_layers_language,
            encoder_type=encoder_type,
            flow_spec_dim=flow_spec_dim,
            flow_spec_time_embedding_dim=flow_spec_time_embedding_dim,
            point_feature_dim=point_feature_dim,
            visual_feature_dim=self.image_embedding_dim,
        )
        self.gripper_patch_projection = nn.Linear(self.image_embedding_dim + 2, token_dim)

    def tokenize_ctx_pointcloud(
        self,
        context : UMAInputContext,
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # keypoint id
        torch.Tensor # valid mask
    ]:
        pointcloud_features, pointcloud_features_mask = self.pointcloud_fusion(
            self.pointcloud_encoder,
            context['pointcloud'],
            context['pointcloud_patch_features']['token_data'],
            context['pointcloud_image_size'],
            context['pointcloud_intrinsics'],
            image_valid_mask=context['pointcloud_patch_features']['valid_mask'],
        )
        return (
            pointcloud_features, # (B, N_groups, token_dim)
            torch.full(pointcloud_features_mask.shape, PLACEHOLDER_KEYPOINT_ID, dtype=torch.int32, device=pointcloud_features.device), # (B, N_groups)
            pointcloud_features_mask # (B, N_groups)
        )
    
    @torch.compile
    def tokenize_gripper_image(
        self,
        gripper_patch_token : UMAContextItem,
        missing_gripper_embedding : torch.Tensor, # (image_embedding_dim,)
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # keypoint id
        torch.Tensor # valid mask
    ]:
        features = torch.where(
            gripper_patch_token['valid_mask'][:, None, None, None], # (B, 1, 1, 1)
            gripper_patch_token['token_data'], # (B, H', W', image_embedding_dim)
            missing_gripper_embedding[None, None, None, :]
        ) # (B, H', W', image_embedding_dim)
        features = ImageTokenizer.add_2d_positional_encoding(features)
        features = self.gripper_patch_projection(features)
        return (
            features,
            torch.full(features.shape[:-1], PLACEHOLDER_KEYPOINT_ID, dtype=torch.int32, device=features.device),
            # gripper_patch_token['valid_mask'][:, None, None].expand(-1, features.shape[1], features.shape[2]) # (B, H', W')
            torch.ones(features.shape[:-1], dtype=torch.bool, device=features.device) # (B, H', W')
        )

    def forward(
        self, 
        context : UMAInputContext,
        missing_gripper_embedding : torch.Tensor,
        *additional_context_items : Union[UMAFlowSpecInputContext, UMAContextItem]
    ):
        stacked_tokens = None
        stacked_tokens = self.stack_tokens(*self.tokenize_ctx_pointcloud(context), stacked_tokens)
        stacked_tokens = self.stack_tokens(*self.tokenize_gripper_image(context['gripper_features'], missing_gripper_embedding), stacked_tokens)

        all_spec_tokens = []
        all_spec_token_valids = []
        main_spec_token, main_spec_token_valid = self.specification_encoder.forward(
            context['flow_spec_context'],
            context['tokenized_language_instructions'],
            self.pointcloud_encoder,
        ) # (B, N_spec, token_dim), (B, )
        all_spec_tokens.append(main_spec_token)
        all_spec_token_valids.append(main_spec_token_valid)

        for item in additional_context_items:
            if 'token_data' not in item.keys():
                token, token_valid = self.specification_encoder.forward_keypoint(
                    item,
                    self.pointcloud_encoder,
                )
                all_spec_tokens.append(token)
                all_spec_token_valids.append(token_valid)
            else:
                token, token_valid = self.specification_encoder.forward_language(
                    item
                )
                all_spec_tokens.append(token)
                all_spec_token_valids.append(token_valid)

        if len(all_spec_tokens) > 0:
            all_spec_tokens = torch.stack(all_spec_tokens, dim=1) # (B, spec_tokens, N_spec, token_dim)
            all_spec_token_valids = torch.stack(all_spec_token_valids, dim=1) # (B, spec_tokens)
        else:
            all_spec_tokens = None
            all_spec_token_valids = None
        return stacked_tokens, (all_spec_tokens, all_spec_token_valids)

class UMADecoder(nn.Module):
    @staticmethod
    def stack_tokens(
        token_feature : torch.Tensor,
        token_absolute_time : torch.Tensor,
        token_keypoint_id : torch.Tensor,
        token_pred_mask : torch.Tensor,
        token_valid_mask : torch.Tensor,
        existing_stacked : Optional[Tuple[
            torch.Tensor, # data
            torch.Tensor, # absolute time
            torch.Tensor, # keypoint id
            torch.Tensor, # pred mask
            torch.Tensor, # valid mask
        ]] = None
    ) -> Tuple[
        Tuple[ # stacked result
            torch.Tensor, # data
            torch.Tensor, # absolute time
            torch.Tensor, # keypoint id
            torch.Tensor, # pred mask
            torch.Tensor, # valid mask
        ],
        Tuple[ # start idx, length, target shape
            int, int, Tuple[int, ...]
        ]
    ]:
        length = np.prod(token_feature.shape[1:-1])
        assert np.prod(token_absolute_time.shape[1:]) == length
        assert np.prod(token_keypoint_id.shape[1:]) == length
        assert np.prod(token_valid_mask.shape[1:]) == length

        target_shape = token_feature.shape[:-1]
        if existing_stacked is None:
            stacked = (
                token_feature.flatten(1, -2),
                token_absolute_time.flatten(1, -1),
                token_keypoint_id.flatten(1, -1),
                token_pred_mask.flatten(1, -1),
                token_valid_mask.flatten(1, -1),
            )
            return stacked, (0, length, target_shape)
        else:
            stacked = (
                torch.concat([existing_stacked[0], token_feature.flatten(1, -2)], dim=1),
                torch.concat([existing_stacked[1], token_absolute_time.flatten(1, -1)], dim=1),
                torch.concat([existing_stacked[2], token_keypoint_id.flatten(1, -1)], dim=1),
                torch.concat([existing_stacked[3], token_pred_mask.flatten(1, -1)], dim=1),
                torch.concat([existing_stacked[4], token_valid_mask.flatten(1, -1)], dim=1),
            )
            return stacked, (stacked[0].shape[1] - length, length, target_shape)

    @staticmethod
    def unstack_token(
        stacked : Tuple[
            torch.Tensor, # data
            torch.Tensor, # absolute time
            torch.Tensor, # keypoint id
            torch.Tensor, # pred mask
            torch.Tensor, # valid mask
        ],
        reference : Tuple[int, int, Tuple[int, ...]]
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # absolute time
        torch.Tensor, # keypoint id
        torch.Tensor, # pred mask
        torch.Tensor, # valid mask
    ]:
        start_idx, length, target_shape = reference
        return (
            stacked[0][:, start_idx:start_idx + length].reshape(target_shape + (-1,)),
            stacked[1][:, start_idx:start_idx + length].reshape(target_shape),
            stacked[2][:, start_idx:start_idx + length].reshape(target_shape),
            stacked[3][:, start_idx:start_idx + length].reshape(target_shape),
            stacked[4][:, start_idx:start_idx + length].reshape(target_shape),
        )

    def __init__(
        self,
        obs_horizon : int,
        pred_horizon : int,
        far_future_times : List[int],
        token_dim : int,
        num_layers : int,
        absolute_time_encoding_dim : int = 64,
        pred_t_encoding_dim : int = 64,
        action_dim : int = 10,
        flow_dim : int = 4,
        is_flow_matching : bool = False,
        encoder_type : EncoderType = "siglip2",
        isolate_context : bool = False,
        **transformer_kwargs,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.num_layers = num_layers
        # Scene, wrist and specification tokens attend only to each other, not to the noisy action / flow tokens.
        self.isolate_context = bool(isolate_context)

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.far_future_times = far_future_times

        self.action_dim = action_dim
        self.flow_dim = flow_dim
        self.is_flow_matching = is_flow_matching
        self.absolute_time_encoding_dim = absolute_time_encoding_dim
        self.hide_current_pose = False  # set by UMAActor
        self.image_embedding_dim = get_image_embedding_dim(encoder_type)
        self.encoder_type = encoder_type

        # ========== Initialize Time Encoders ==========
        # Prediction time encoding spans pred_t_encoding_dim
        self.pred_t_encoding = TimestepPosEmb(
            pred_t_encoding_dim,
            min_period=0.004,
            max_period=4
        )
        self.pred_t_known_embedding = nn.Parameter(torch.zeros(pred_t_encoding_dim))
        nn.init.trunc_normal_(self.pred_t_known_embedding, std=0.02)

        self.register_buffer(
            "absolute_times_full",
            torch.concat([
                torch.arange(-obs_horizon + 1, pred_horizon + 1),
                torch.tensor(far_future_times)
            ])
        ) # (L, )
        
        self.absolute_times_full_embedding = nn.Parameter(torch.zeros(len(self.absolute_times_full), absolute_time_encoding_dim)) # (L, absolute_time_encoding_dim)
        nn.init.trunc_normal_(self.absolute_times_full_embedding, std=0.02)

        # ========== Tokenizer Layers ==========
        # Current pose anchor: position + 6D orientation.
        self.action_context_dim = ACTION_POSE_DIM
        self.action_encoder = nn.Sequential(
            nn.Linear(pred_t_encoding_dim + self.action_context_dim + action_dim + absolute_time_encoding_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        # Tracked points are anchored at their current 3D position.
        self.flow_encoder = nn.Sequential(
            nn.Linear(pred_t_encoding_dim + 3 + flow_dim + absolute_time_encoding_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, token_dim),
        )

        # ========== Transformer Layers ==========
        # Each discrete time gets a CLS token
        self.cls_token_time = nn.Parameter(torch.zeros(
            1, obs_horizon + pred_horizon + len(far_future_times), token_dim
        ))
        self.cls_token_time = nn.init.trunc_normal_(self.cls_token_time, std=0.02)

        if is_flow_matching:
            self.pred_t_condition_net = nn.Sequential(
                nn.Linear(pred_t_encoding_dim, pred_t_encoding_dim * 2),
                nn.ReLU(),
                nn.Linear(pred_t_encoding_dim * 2, pred_t_encoding_dim)
            )
        else:
            self.pred_t_condition_net = None

        self.layers_keypoint = nn.ModuleList([
            MaskedDiTBlock(
                FlexAttention(
                    token_dim,
                    num_heads=token_dim // 32,
                    compile=True,
                ),
                in_dim_condition=pred_t_encoding_dim,
                has_context=False,
                norm_layer=nn.LayerNorm,
                **transformer_kwargs
            ) if self.is_flow_matching else TransformerBlock(
                FlexAttention(
                    token_dim,
                    num_heads=token_dim // 32,
                    compile=True,
                ),
                has_context=False,
                norm_layer=nn.LayerNorm,
                **transformer_kwargs
            )
            for _ in range(num_layers)
        ])
        self.layers_time = nn.ModuleList([
            MaskedDiTBlock(
                FlexAttention(
                    token_dim,
                    num_heads=token_dim // 32,
                    compile=True,
                ),
                in_dim_condition=pred_t_encoding_dim,
                has_context=False,
                norm_layer=nn.LayerNorm,
                **transformer_kwargs
            ) if self.is_flow_matching else TransformerBlock(
                FlexAttention(
                    token_dim,
                    num_heads=token_dim // 32,
                    compile=True,
                ),
                has_context=False,
                norm_layer=nn.LayerNorm,
                **transformer_kwargs
            )
            for _ in range(num_layers)
        ])

        # ========= Detokenizer Layers ==========
        self.action_decoder = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, action_dim)
        )
        self.flow_decoder = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.ReLU(),
            nn.Linear(token_dim * 2, flow_dim)
        )
     
    @property
    def full_horizon(self):
        return self.obs_horizon + self.pred_horizon + len(self.far_future_times)

    def tokenize_cls(
        self,
        batch_size : int,
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # absolute time
        torch.Tensor, # keypoint id
        torch.Tensor, # pred mask
        torch.Tensor # valid mask
    ]:
        return (
            self.cls_token_time.expand(batch_size, -1, -1),
            self.absolute_times_full[None, :].expand(batch_size, -1),
            torch.full((batch_size, self.full_horizon), PLACEHOLDER_KEYPOINT_ID, dtype=torch.int32, device=self.cls_token_time.device),
            torch.zeros((batch_size, self.full_horizon), dtype=torch.bool, device=self.cls_token_time.device),
            torch.ones((batch_size, self.full_horizon), dtype=torch.bool, device=self.cls_token_time.device)
        )

    @torch.compile
    def tokenize_flow(
        self,
        flow_token : UMATokenItem, # (B, T, M, flow_dim)
        flow_mean : torch.Tensor, # (B, M, flow_dim)
        pred_t : torch.Tensor, # (B,)
        pred_t_embedding : torch.Tensor, # (B, pred_t_encoding_dim)
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # absolute time
        torch.Tensor, # keypoint id
        torch.Tensor, # pred mask
        torch.Tensor, # valid mask
    ]:
        B, T, M, _ = flow_token['token_data'].shape
        absolute_time_embedding = self.absolute_times_full_embedding[None, :, None, :].expand(flow_token['token_data'].shape[0], -1, flow_token['token_data'].shape[2], -1)
        flow_times = self.absolute_times_full[None, :, None].expand(flow_token['token_data'].shape[0], -1, flow_token['token_data'].shape[2])

        # Compute 3D positional embedding for flow tokens
        flow_masked_pred_t_embedding = torch.where(
            flow_token['pred_mask'][..., None],
            pred_t_embedding[:, None, None, :],
            self.pred_t_known_embedding[None, None, None, :],
        )
        flow_features = torch.concat([
            flow_masked_pred_t_embedding,
            flow_mean[:, None, :, :].expand(-1, flow_token['token_data'].shape[1], -1, -1),
            flow_token['token_data'],
            absolute_time_embedding,
        ], dim=-1)

        flow_features = self.flow_encoder(flow_features) # (B, T, M, token_dim)
        return (
            flow_features,
            flow_times,
            torch.arange(flow_features.shape[2], device=flow_features.device)[None, None, :].expand(flow_features.shape[0], flow_features.shape[1], -1),
            flow_token['pred_mask'],
            flow_token['valid_mask']
        )
    
    @torch.compile
    def tokenize_action(
        self,
        action_token : UMATokenItem, # (B, T, action_dim)
        action_mean : torch.Tensor, # (B, 9): current position + current rotation anchor
        pred_t : torch.Tensor, # (B,)
        pred_t_embedding : torch.Tensor, # (B, pred_t_encoding_dim)
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # absolute time
        torch.Tensor, # keypoint id
        torch.Tensor, # pred mask
        torch.Tensor, # valid mask
    ]:
        action_times = self.absolute_times_full[None, :].expand(action_token['token_data'].shape[0], -1)

        action_masked_pred_t_embedding = torch.where(
            action_token['pred_mask'][..., None],
            pred_t_embedding[:, None, :],
            self.pred_t_known_embedding[None, None, :],
        )

        action_features = torch.concat([
            action_masked_pred_t_embedding,
            action_mean[:, None, :].expand(-1, action_token['token_data'].shape[1], -1),
            action_token['token_data'],
            self.absolute_times_full_embedding[None, :, :].expand(action_token['token_data'].shape[0], -1, -1),
        ], dim=-1)
        action_features = self.action_encoder(action_features) # (B, T, token_dim)
        return (
            action_features,
            action_times,
            torch.full(action_features.shape[:-1], PLACEHOLDER_KEYPOINT_ID, dtype=torch.int32, device=action_features.device),
            action_token['pred_mask'],
            action_token['valid_mask']
        )

    def forward_transformer_layers(
        self,
        token_features : torch.Tensor, # (B, N, token_dim)
        token_absolute_time : torch.Tensor, # (B, N)
        token_keypoint_id : torch.Tensor, # (B, N)
        token_pred_mask : torch.Tensor, # (B, N)
        token_valid_mask : torch.Tensor, # (B, N)
        context_features : torch.Tensor, # (B, N_ctx, token_dim)
        context_keypoint_id : torch.Tensor, # (B, N_ctx)
        context_valid_mask : torch.Tensor, # (B, N_ctx)
        pred_t : torch.Tensor, # (B, )
        pred_t_embedding : torch.Tensor, # (B, pred_t_encoding_dim)
    ) -> torch.Tensor:
        aggregated_features = torch.concat([token_features, context_features], dim=1) # (B, N + N_ctx, token_dim)
        aggregated_absolute_time = torch.concat([token_absolute_time, torch.full_like(context_keypoint_id, PLACEHOLDER_TIME_ID)], dim=1) # (B, N + N_ctx)
        aggregated_keypoint_id = torch.concat([token_keypoint_id, context_keypoint_id], dim=1) # (B, N + N_ctx)
        aggregated_pred_mask = torch.concat([token_pred_mask, torch.zeros_like(context_valid_mask)], dim=1) # (B, N + N_ctx)
        aggregated_valid_mask = torch.concat([token_valid_mask, context_valid_mask], dim=1) # (B, N + N_ctx)

        B, N_all, _ = aggregated_features.shape

        # Compute DiT conditionings
        if self.is_flow_matching:
            masked_pred_t_condition = self.pred_t_condition_net(pred_t_embedding) # (B, pred_t_encoding_dim)
        else:
            masked_pred_t_condition = None

        # Utility functions
        def is_same_keypoint(b, q_idx, kv_idx):
            return (aggregated_keypoint_id[b, q_idx] == aggregated_keypoint_id[b, kv_idx]) & (aggregated_keypoint_id[b, q_idx] != PLACEHOLDER_KEYPOINT_ID)

        def is_same_time(b, q_idx, kv_idx):
            return (aggregated_absolute_time[b, q_idx] == aggregated_absolute_time[b, kv_idx]) & (aggregated_absolute_time[b, q_idx] != PLACEHOLDER_TIME_ID)

        # Mask functions
        def mask_same_time_sa(b, h, q_idx, kv_idx):
            return is_same_time(b, q_idx, kv_idx) | (aggregated_absolute_time[b, kv_idx] == PLACEHOLDER_TIME_ID)

        def mask_same_keypoint_sa(b, h, q_idx, kv_idx):
            return is_same_keypoint(b, q_idx, kv_idx) | (aggregated_keypoint_id[b, kv_idx] == PLACEHOLDER_KEYPOINT_ID)

        def mask_valid_sa(b, h, q_idx, kv_idx):
            return aggregated_valid_mask[b, kv_idx]
    
        def mask_same_token(b, h, q_idx, kv_idx):
            # Invalid query rows are not used for losses, but FlexAttention can
            # produce NaNs when a query has no legal keys. Keep valid-query
            # behavior unchanged and give invalid queries a finite self path.
            return (q_idx == kv_idx)

        n_tokens = token_features.shape[1]  # context tokens follow the action / flow / cls tokens

        def mask_context_isolated(b, h, q_idx, kv_idx):
            return (q_idx < n_tokens) | (kv_idx >= n_tokens)

        time_mask = and_masks(mask_valid_sa, mask_same_time_sa)
        keypoint_mask = and_masks(mask_valid_sa, mask_same_keypoint_sa)
        if self.isolate_context:
            time_mask = and_masks(time_mask, mask_context_isolated)
            keypoint_mask = and_masks(keypoint_mask, mask_context_isolated)
        layer_time_mask_sa = create_block_mask(
            or_masks(time_mask, mask_same_token),
            B, self.layers_time[0].num_heads, N_all, N_all, device=aggregated_features.device
        )
        layer_keypoint_mask_sa = create_block_mask(
            or_masks(keypoint_mask, mask_same_token),
            B, self.layers_keypoint[0].num_heads, N_all, N_all, device=aggregated_features.device
        )

        # Forward through transformer layers
        for i in range(self.num_layers):
            layer_timestep : MaskedDiTBlock = self.layers_time[i]
            layer_keypoint : MaskedDiTBlock = self.layers_keypoint[i]

            if self.is_flow_matching:
                aggregated_features = layer_keypoint.forward(
                    aggregated_features,
                    condition=masked_pred_t_condition,
                    condition_mask=aggregated_pred_mask,
                    block_mask=layer_keypoint_mask_sa
                )
            else:
                aggregated_features = layer_keypoint.forward(
                    aggregated_features,
                    block_mask=layer_keypoint_mask_sa
                )
            
            if self.is_flow_matching:
                aggregated_features = layer_timestep.forward(
                    aggregated_features,
                    condition=masked_pred_t_condition,
                    condition_mask=aggregated_pred_mask,
                    block_mask=layer_time_mask_sa
                )
            else:
                aggregated_features = layer_timestep.forward(
                    aggregated_features,
                    block_mask=layer_time_mask_sa
                )
        return aggregated_features[:, :token_features.shape[1], :] # (B, N, token_dim)
    
    @torch.compile
    def untokenize_action(
        self,
        action_features : torch.Tensor, # (B, N, token_dim)
    ) -> torch.Tensor:
        return self.action_decoder(action_features)
    
    @torch.compile
    def untokenize_flow(
        self,
        flow_features : torch.Tensor, # (B, N, token_dim)
    ) -> torch.Tensor:
        return self.flow_decoder(flow_features)

    def forward(
        self,
        context_tokens : Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masked_tokens : UMANormalizedMaskedTokens,
        pred_t : torch.Tensor,
    ) -> UMANormalizedMaskedTokens:
        B = masked_tokens['action']['token_data'].shape[0] if masked_tokens['action'] is not None else masked_tokens['flow']['token_data'].shape[0]
        L = masked_tokens['action']['token_data'].shape[1] if masked_tokens['action'] is not None else masked_tokens['flow']['token_data'].shape[1]
        assert L == self.obs_horizon + self.pred_horizon + len(self.far_future_times), f"Input length {L} does not match expected {self.obs_horizon + 1 + self.pred_horizon + len(self.far_future_times)}"

        # Clamp pred_t for numerical stability
        pred_t = torch.clamp(pred_t, min=1e-5, max=1.0 - 1e-5)
        pred_t_embedding = self.pred_t_encoding.compute_embedding_with_position(pred_t)

        stacked_tokens = None
        stacked_tokens, _ = self.stack_tokens(
            *self.tokenize_cls(
                batch_size=B
            ), stacked_tokens
        )

        if masked_tokens['flow'] is not None:
            stacked_tokens, flow_token_reference = self.stack_tokens(
                *self.tokenize_flow(
                    masked_tokens['flow'],
                    masked_tokens['flow_mean'],
                    pred_t,
                    pred_t_embedding,
                ), stacked_tokens
            )
        else:
            flow_token_reference = None

        if masked_tokens['action'] is not None:
            stacked_tokens, action_token_reference = self.stack_tokens(
                *self.tokenize_action(
                    masked_tokens['action'],
                    (
                        torch.zeros_like(masked_tokens['action_mean'])
                        if self.hide_current_pose
                        else masked_tokens['action_mean']
                    ),
                    pred_t,
                    pred_t_embedding
                ), stacked_tokens
            )
        else:
            action_token_reference = None

        token_features, token_absolute_time, token_keypoint_id, token_pred_mask, token_valid_mask = stacked_tokens
        context_features, context_keypoint_id, context_valid_mask = context_tokens

        token_features = self.forward_transformer_layers(
            token_features,
            token_absolute_time,
            token_keypoint_id,
            token_pred_mask,
            token_valid_mask,
            context_features,
            context_keypoint_id,
            context_valid_mask,
            pred_t,
            pred_t_embedding
        ) # (B, N, token_dim)

        stacked_tokens = (
            token_features,
            token_absolute_time,
            token_keypoint_id,
            token_pred_mask,
            token_valid_mask
        )

        ret = {}
        if flow_token_reference is not None:
            flow_data, _, _, _, flow_valid_mask = self.unstack_token(
                stacked_tokens,
                flow_token_reference
            )
            flow_data = self.untokenize_flow(flow_data)
            ret['flow'] = dict_replace(
                masked_tokens['flow'],
                token_data=flow_data
            )
        else:
            ret['flow'] = None
        if action_token_reference is not None:
            action_data, _, _, _, action_valid_mask = self.unstack_token(
                stacked_tokens,
                action_token_reference
            )
            action_data = self.untokenize_action(action_data)
            ret['action'] = dict_replace(
                masked_tokens['action'],
                token_data=action_data
            )
        else:
            ret['action'] = None
        ret['action_mean'] = masked_tokens['action_mean']
        ret['flow_mean'] = masked_tokens['flow_mean']
        return ret

class UMATransformer(PolicyFeatureExtractor[UMAImageContext, UMANormalizedMaskedTokens]):
    def __init__(
        self,
        obs_horizon : int,
        pred_horizon : int,
        far_future_times : List[int],
        token_dim : int,
        n_spec_tokens : int,
        num_layers : int,
        num_layers_stt : int,
        num_layers_agg : int,
        num_layers_language : int,
        absolute_time_encoding_dim : int = 64,
        pred_t_encoding_dim : int = 64,
        flow_spec_time_embedding_dim : int = 64,
        action_dim : int = 10,
        flow_dim : int = 4,
        flow_spec_dim : int = 3,
        is_flow_matching : bool = False,
        encoder_type : EncoderType = "siglip2",
        isolate_context : bool = False,
        motion_readout_hidden_dim : int = 128,
        motion_readout_max_displacement_m : float = 2.0,
        contrastive_projection_dim : int = 512,
        motion_readout_detach : bool = False,
        **transformer_kwargs,
    ) -> None:
        super().__init__()
        
        # Use TF32 precision for matmul operations on float32 tensors
        torch.set_float32_matmul_precision('high')

        self.encoder_type = encoder_type
        self.image_emedding_dim = get_image_embedding_dim(encoder_type)
        self.missing_gripper_embedding = nn.Parameter(torch.zeros(self.image_emedding_dim))
        nn.init.trunc_normal_(self.missing_gripper_embedding, std=0.02)

        self.n_spec_tokens = n_spec_tokens
        self.task_latent = nn.Parameter(torch.zeros((n_spec_tokens, token_dim)))
        nn.init.trunc_normal_(self.task_latent, std=0.02)

        self.context_encoder = UMAContextEncoder(
            token_dim=token_dim,
            n_spec_tokens=n_spec_tokens,
            num_layers_stt=num_layers_stt,
            num_layers_agg=num_layers_agg,
            num_layers_language=num_layers_language,
            flow_spec_time_embedding_dim=flow_spec_time_embedding_dim,
            flow_spec_dim=flow_spec_dim,
            encoder_type=encoder_type,
        )
        self.decoder = UMADecoder(
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            far_future_times=far_future_times,
            token_dim=token_dim,
            num_layers=num_layers,
            absolute_time_encoding_dim=absolute_time_encoding_dim,
            pred_t_encoding_dim=pred_t_encoding_dim,
            action_dim=action_dim,
            flow_dim=flow_dim,
            is_flow_matching=is_flow_matching,
            encoder_type=encoder_type,
            isolate_context=isolate_context,
            **transformer_kwargs
        )
        # The ``_v2`` suffix keeps these parameter names distinct from an earlier, much
        # larger flattened projection head (6144 -> 12288 -> 6144) whose gradients
        # dominated the DDP all-reduce for a loss weighted 0.001. Checkpoints that
        # contain that head still load (strict=False) and simply skip it.
        self.contrastive_projection_v2 = nn.Sequential(
            nn.Linear(n_spec_tokens * token_dim, contrastive_projection_dim),
            nn.ReLU(),
            nn.Linear(contrastive_projection_dim, contrastive_projection_dim)
        )
        # The ``_v2`` suffix likewise keeps this readout distinct from an earlier,
        # less stable flattened readout head. PolicyActor loads networks with
        # strict=False, so checkpoints that contain that head skip it on load.
        self.motion_readout_v2 = NormalizedMeanMotionReadout(
            token_dim=token_dim,
            hidden_dim=motion_readout_hidden_dim,
            max_displacement_m=motion_readout_max_displacement_m,
        )
        # Detached: the readout is a probe of the spec tokens, its loss trains only
        # its own head (every parameter still takes part in the loss, as DDP needs).
        self.motion_readout_detach = bool(motion_readout_detach)

    @property
    def obs_horizon(self):
        return self.decoder.obs_horizon

    @property
    def pred_horizon(self):
        return self.decoder.pred_horizon

    @property
    def full_horizon(self):
        return self.decoder.full_horizon

    @property
    def far_future_times(self):
        return self.decoder.far_future_times

    @property
    def action_dim(self):
        return self.decoder.action_dim
    
    @property
    def flow_dim(self):
        return self.decoder.flow_dim
    
    @property
    def flow_spec_dim(self):
        return self.context_encoder.flow_spec_dim

    def forward(
        self,
        context_tokens : UMAImageContext,
        masked_tokens : UMANormalizedMaskedTokens,
        pred_t : torch.Tensor,
        *additional_context_items : Union[Tuple[UMAContextItem, ...], UMAContextItem],
        prompt_tuning : bool = False,
    ) -> Tuple[UMANormalizedMaskedTokens, Tuple[torch.Tensor, torch.Tensor]]:
        context_output, (spec_tokens, spec_tokens_valids) = self.context_encoder.forward(context_tokens, self.missing_gripper_embedding, *additional_context_items)

        if prompt_tuning:
            spec_token = self.task_latent[None].expand(context_output[0].shape[0], -1, -1) # (B, N_spec, token_dim)
            spec_token_valid = torch.ones(context_output[0].shape[0], dtype=torch.bool, device=context_output[0].device) # (B, )
        else:
            spec_token = spec_tokens[:, 0] # (B, N_spec, token_dim)
            spec_token_valid = spec_tokens_valids[:, 0] # (B, )

        context_output = UMAContextEncoder.stack_spec_tokens(
            spec_token,
            spec_token_valid,
            context_output
        )
        
        token_output = self.decoder.forward(
            context_output,
            masked_tokens,
            pred_t=pred_t,
        )
        spec_contrastive_features = self.contrastive_projection_v2(spec_tokens.flatten(-2, -1)) # (B, S, contrastive_projection_dim)
        motion_readout_pred = self.motion_readout_v2(
            spec_tokens.detach() if self.motion_readout_detach else spec_tokens
        ) # (B, S, 3)
        return token_output, (spec_contrastive_features, spec_tokens_valids, motion_readout_pred)

    def compute_auxilary_losses(
        self,
        norm_context : UMAImageContext,
        output : Tuple[UMAMaskedTokens, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        norm_goal_output : UMAMaskedTokens,
        is_flow_matching : bool = False,
        contrastive_temperature : float = 0.07,
        contrastive_hard_negative_mask : Optional[torch.Tensor] = None,
        motion_readout_targets : Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs
    ):
        losses = self.compute_policy_losses(
            output[0],
            norm_goal_output
        )
        spec_contrastive_features, spec_tokens_valids, motion_readout_pred = output[1]
        losses.update(self.compute_contrastive_loss(
            spec_contrastive_features,
            spec_tokens_valids,
            temperature=contrastive_temperature,
            hard_negative_mask=contrastive_hard_negative_mask,
        ))
        if motion_readout_targets is not None:
            targets, target_valids = motion_readout_targets
            losses.update(self.compute_motion_readout_loss(
                motion_readout_pred,
                targets,
                target_valids & spec_tokens_valids.bool(),
            ))

        return losses

    def compute_motion_readout_loss(
        self,
        motion_readout_pred : torch.Tensor, # (B, S, 3), predicts displacement * SPEC_DISPLACEMENT_SCALE
        targets : torch.Tensor, # (B, S, 3), mean spec displacement in meters
        target_valids : torch.Tensor, # (B, S)
    ):
        scaled_targets = targets * SPEC_DISPLACEMENT_SCALE
        # The prediction's norm is bounded (NormalizedMeanMotionReadout); bound the
        # target the same way so an outlier spec cannot carry an irreducible loss.
        bound = self.motion_readout_v2.max_scaled_displacement
        scaled_targets = scaled_targets * torch.clamp(
            bound / torch.clamp(scaled_targets.norm(dim=-1, keepdim=True), min=1e-6), max=1.0
        )
        losses = {
            "motion_readout_loss": masked_mean_huber_loss(
                motion_readout_pred,
                scaled_targets,
                valid_mask=target_valids,
                delta=2.0,
            )
        }
        with torch.no_grad():
            error_m = (motion_readout_pred / SPEC_DISPLACEMENT_SCALE - targets).norm(dim=-1)
            denominator = torch.clamp(target_valids.sum(), min=1).float()
            losses["motion_readout_error_m"] = (
                torch.where(target_valids, error_m, 0.0).sum() / denominator
            ).detach()
            pred_norm_m = (motion_readout_pred / SPEC_DISPLACEMENT_SCALE).norm(dim=-1)
            target_norm_m = targets.norm(dim=-1)
            losses["motion_readout_pred_norm_max_m"] = torch.where(
                target_valids, pred_norm_m, 0.0
            ).max().detach()
            losses["motion_readout_target_norm_max_m"] = torch.where(
                target_valids, target_norm_m, 0.0
            ).max().detach()
        return losses

    def compute_contrastive_loss(
        self,
        spec_contrastive_features : torch.Tensor, # (B, S, contrastive_projection_dim)
        spec_tokens_valids : torch.Tensor, # (B, S)
        temperature : float = 0.07,
        hard_negative_mask : Optional[torch.Tensor] = None, # (S, ), True for scene-matched hard negatives
    ):
        B, S, _ = spec_contrastive_features.shape
        features = F.normalize(spec_contrastive_features, dim=-1).flatten(0, 1) # (B*S, token_dim)
        valid_mask = spec_tokens_valids.flatten(0, 1).bool() # (B*S, )
        batch_ids = torch.arange(B, device=features.device).repeat_interleave(S) # (B*S, )
        if hard_negative_mask is None:
            negative_ids = torch.zeros(B * S, dtype=torch.bool, device=features.device)
        else:
            negative_ids = hard_negative_mask.to(features.device)[None, :].expand(B, S).flatten(0, 1) # (B*S, )

        # Invalid tokens are masked out of every row and column instead of being
        # indexed away (boolean indexing and empty-batch branches would sync the
        # host every step). Masked logits use a large finite negative, not -inf,
        # so rows without positives stay finite and get zero gradient.
        logits = torch.matmul(features, features.T) / temperature # (B*S, B*S)
        self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        pair_valid = valid_mask[:, None] & valid_mask[None, :]
        # Hard negatives share the anchor's scene but have a different trajectory
        # (time-reversed / frozen): they enter the denominator like any other
        # negative but are never positives and never act as anchors themselves.
        # Without them, scene identity alone solves the contrastive task and the
        # spec embedding need not encode the trajectory at all.
        positive_mask = (
            (batch_ids[:, None] == batch_ids[None, :])
            & ~self_mask
            & ~negative_ids[None, :]
            & ~negative_ids[:, None]
            & pair_valid
        )
        denominator_mask = ~self_mask & pair_valid
        anchor_mask = positive_mask.any(dim=-1)
        masked = torch.finfo(logits.dtype).min / 2
        log_denominator = torch.logsumexp(logits.masked_fill(~denominator_mask, masked), dim=-1)
        log_positive = torch.logsumexp(logits.masked_fill(~positive_mask, masked), dim=-1)
        loss = torch.where(anchor_mask, -(log_positive - log_denominator), 0.0)
        n_anchors = anchor_mask.sum()
        # No anchors (e.g. at most one valid token): zero loss, still attached to the graph.
        return {"contrastive_loss": loss.sum() / n_anchors.clamp(min=1) + spec_contrastive_features.sum() * 0.0}

    def compute_policy_losses(
        self,
        norm_output : UMAMaskedTokens, 
        norm_goal_output : UMAMaskedTokens, 
    ):
        losses : Dict[str, torch.Tensor] = {}
        for key, obs_item in norm_goal_output.items():
            if key.endswith("_mean"):
                continue

            output_item = norm_output.get(key)
            goal_obs_item = norm_goal_output.get(key)
            
            mask = torch.logical_and(
                obs_item['pred_mask'], 
                obs_item['valid_mask']
            )
            
            output_data = output_item['token_data']
            
            target_data = goal_obs_item['token_data'].detach()
            huber_delta = 2.0

            if key == "action":
                losses["action_position_loss"] = masked_mean_huber_loss(
                    output_data[:, :self.obs_horizon + self.pred_horizon, ..., :3],
                    target_data[:, :self.obs_horizon + self.pred_horizon, ..., :3],
                    valid_mask=mask[:, :self.obs_horizon + self.pred_horizon],
                    delta=huber_delta
                )
                # Rotations are trained in the current-timestep frame, not as absolute 6D.
                losses["action_rotation_loss"] = masked_mean_huber_loss(
                    output_data[:, :self.obs_horizon + self.pred_horizon, ..., 3:-1],
                    target_data[:, :self.obs_horizon + self.pred_horizon, ..., 3:-1],
                    valid_mask=mask[:, :self.obs_horizon + self.pred_horizon],
                    delta=huber_delta
                )
                losses['action_gripper_loss'] = masked_mean_huber_loss(
                    output_data[:, :self.obs_horizon + self.pred_horizon, ..., -1],
                    target_data[:, :self.obs_horizon + self.pred_horizon, ..., -1],
                    valid_mask=mask[:, :self.obs_horizon + self.pred_horizon],
                    delta=huber_delta
                )
                if len(self.far_future_times) > 0:
                    losses["far_action_position_loss"] = masked_mean_huber_loss(
                        output_data[:, self.obs_horizon + self.pred_horizon:, ..., :3],
                        target_data[:, self.obs_horizon + self.pred_horizon:, ..., :3],
                        valid_mask=mask[:, self.obs_horizon + self.pred_horizon:],
                        delta=huber_delta
                    )
                    losses["far_action_rotation_loss"] = masked_mean_huber_loss(
                        output_data[:, self.obs_horizon + self.pred_horizon:, ..., 3:-1],
                        target_data[:, self.obs_horizon + self.pred_horizon:, ..., 3:-1],
                        valid_mask=mask[:, self.obs_horizon + self.pred_horizon:],
                        delta=huber_delta
                    )
                    losses['far_action_gripper_loss'] = masked_mean_huber_loss(
                        output_data[:, self.obs_horizon + self.pred_horizon:, ..., -1],
                        target_data[:, self.obs_horizon + self.pred_horizon:, ..., -1],
                        valid_mask=mask[:, self.obs_horizon + self.pred_horizon:],
                        delta=huber_delta
                    )
            elif key == "flow":
                losses["flow_position_loss"] = masked_mean_huber_loss(
                    output_data[:, :self.obs_horizon + self.pred_horizon, ..., :-1],
                    target_data[:, :self.obs_horizon + self.pred_horizon, ..., :-1],
                    valid_mask=mask[:, :self.obs_horizon + self.pred_horizon],
                    delta=huber_delta
                )
                losses["flow_visibility_loss"] = masked_mean_huber_loss(
                    output_data[:, :self.obs_horizon + self.pred_horizon, ..., -1],
                    target_data[:, :self.obs_horizon + self.pred_horizon, ..., -1],
                    valid_mask=mask[:, :self.obs_horizon + self.pred_horizon],
                    delta=huber_delta
                )
                if len(self.far_future_times) > 0:
                    losses["far_flow_position_loss"] = masked_mean_huber_loss(
                        output_data[:, self.obs_horizon + self.pred_horizon:, ..., :-1],
                        target_data[:, self.obs_horizon + self.pred_horizon:, ..., :-1],
                        valid_mask=mask[:, self.obs_horizon + self.pred_horizon:],
                        delta=huber_delta
                    )
                    losses["far_flow_visibility_loss"] = masked_mean_huber_loss(
                        output_data[:, self.obs_horizon + self.pred_horizon:, ..., -1],
                        target_data[:, self.obs_horizon + self.pred_horizon:, ..., -1],
                        valid_mask=mask[:, self.obs_horizon + self.pred_horizon:],
                        delta=huber_delta
                    )
            else:
                losses[key + "_loss"] = masked_mean_rmse_loss(
                    output_data,
                    target_data,
                    valid_mask=mask,
                )

        return losses
    
class UMAActor(PolicyActor[Dict[str, Any], Dict[str, Any]]):
    def __init__(
        self,
        feature_extractor: UMATransformer,
        actor_config : PolicyActorConfig,
        additional_times : List[int],
        max_language_len : int = 256,
        pointcloud_n_points : int = 2048,
        max_gripper_velocity : float = 0.05, # In meters per timestep
        max_flow_velocity : float = 0.05, # In meters per timestep
        action_clip_position: float = 0.2, # Max distance from current position to clip action to, in meters
        flow_clip_position: float = 0.2, # Max distance from current position to clip flow to, in meters
        encoder_type : EncoderType = "siglip2",
        *,
        action_position_std : Optional[List[float]] = None, # per token position (m); overrides the velocity-based std
        hide_proprio_history : bool = False, # True to never show past end-effector states to the model
        hide_current_pose : bool = False, # True to hide the current end-effector pose from the action tokens
        num_training_steps : int = 1_000_000,
        num_warmup_steps : int = 1000,
        timesteps_test : int = 10,
        max_train_timestep : float = 1.0,
        loss_weights : Dict[str, float] = {
            "action_position_loss": 1.0,
            "action_rotation_loss": 1.0,
            "action_gripper_loss": 1.0,
            "far_action_position_loss": 1.0,
            "far_action_rotation_loss": 1.0,
            "far_action_gripper_loss": 1.0,
            "flow_position_loss": 1.0,
            "flow_visibility_loss": 1.0,
            "far_flow_position_loss": 1.0,
            "far_flow_visibility_loss": 1.0,
            "contrastive_loss": 1.0,
        },
        is_flow_matching : bool = True,
        detach_tracking : bool = False,
        apply_se3_augmentation : bool = False,
        se3_translation_std : float = 0.1,
        se3_mirror_prob : float = 0.5,
        se3_rotation_max_angle : float = float(np.pi / 6),
        contrastive_hard_negative_specs : bool = True,
        prompt_tuning : bool = False,
    ):
        nets = {
            "feature_extractor": feature_extractor,
        }
        super().__init__(
            nets=nets,
            feature_extractor=feature_extractor,
            loss_weights=loss_weights
        )

        self.feature_extractor : UMATransformer = feature_extractor
        if actor_config.use_ema:
            self._init_emanet()

        if prompt_tuning:
            assert hasattr(self.feature_extractor, 'task_latent'), "Feature extractor must have 'task_latent' attribute for prompt tuning"
            for param in self.feature_extractor.parameters():
                param.requires_grad_(False)
            self.feature_extractor.task_latent.requires_grad_(True)
        else:
            self.feature_extractor.requires_grad_(True)
            self.feature_extractor.task_latent.requires_grad_(False)

        self.aux_loss_fn = feature_extractor.compute_auxilary_losses
        self.policy_loss_fn = feature_extractor.compute_policy_losses
        self.act_horizon = actor_config.act_horizon
        self.pred_horizon = actor_config.pred_horizon
        self.obs_horizon = actor_config.obs_horizon
        self.additional_times = additional_times
        self.max_language_len = max_language_len
        self.pointcloud_n_points = pointcloud_n_points
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.learning_rate = actor_config.learning_rate
        self.weight_decay = actor_config.weight_decay
        self.timesteps_test = timesteps_test
        # Flow-matching training timesteps are drawn from (0.001, max_train_timestep] (t = 1 is clean data):
        # excluding the near-clean range helps learning (0.9 excludes t > 0.9).
        assert 0.001 < max_train_timestep <= 1.0, f"max_train_timestep must be in (0.001, 1], got {max_train_timestep}"
        self.max_train_timestep = float(max_train_timestep)
        self._init_optimizer()

        self.max_gripper_velocity = max_gripper_velocity
        self.max_flow_velocity = max_flow_velocity
        # Optional per-position action std (m), e.g. the measured RMS displacement of the
        # actions: the velocity-based std max_velocity * sqrt(|t| + 1) leaves action targets
        # at roughly 0.1-0.4 against unit flow-matching noise.
        self.action_position_std = None if action_position_std is None else torch.tensor(list(action_position_std), dtype=torch.float32)
        self.action_clip_position = action_clip_position
        self.flow_clip_position = flow_clip_position
        self.is_flow_matching = is_flow_matching
        self.hide_proprio_history = hide_proprio_history
        self.feature_extractor.decoder.hide_current_pose = hide_current_pose
        self.detach_tracking = detach_tracking
        self.encoder_type = encoder_type
        self.apply_se3_augmentation = apply_se3_augmentation
        self.se3_translation_std = se3_translation_std
        self.se3_mirror_prob = se3_mirror_prob
        self.se3_rotation_max_angle = se3_rotation_max_angle
        self.contrastive_hard_negative_specs = contrastive_hard_negative_specs
        self.rng = torch.Generator(device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.rng.manual_seed(torch.initial_seed())  # follows the (per-rank) global seed
        self.prompt_tuning = prompt_tuning

    def predict_executable_action_with_nets(
        self, 
        nets : Dict[str, nn.Module], 
        context : UMAImageContext, 
        obs : UMAMaskedTokens, 
        rng : Optional[torch.Generator] = None, 
        **kwargs
    ):
        return self.get_output_with_nets(
            nets,
            context,
            obs,
            rng=rng,
            **kwargs
        )
    
    def get_motion_latent(
        self,
        keypoint_trajectory : UMAFlowSpecInputContext,
        use_ema : bool = False,
        *,
        accelerator : Optional["Accelerator"] = None,
        **kwargs
    ) -> torch.Tensor:
        if use_ema and self.ema_model is not None:
            self._sync_emanet()
            model = self.ema_nets
        else:
            model = self.nets if accelerator is None else {
                key: accelerator.unwrap_model(net) for key, net in self.nets.items()
            }
        return self.get_motion_latent_with_nets(
            model,
            keypoint_trajectory
        )

    def get_motion_latent_with_nets(
        self,
        nets : Dict[str, nn.Module],
        keypoint_trajectory : UMAFlowSpecInputContext,
        **kwargs
    ) -> torch.Tensor:
        with torch.no_grad():
            motion_latent, _ = nets['feature_extractor'].context_encoder.specification_encoder.forward_keypoint(
                keypoint_trajectory,
                nets['feature_extractor'].context_encoder.pointcloud_encoder,
            )
        return motion_latent

    def get_output_and_task_latent(
        self,
        context : UMAImageContext,
        obs : UMAMaskedTokens,
        rng : Optional[torch.Generator] = None,
        start_pred_t : float = 1e-3,
        latent : Optional[torch.Tensor] = None,
        use_ema : bool = False,
        *,
        accelerator : Optional["Accelerator"] = None,
        **kwargs
    ) -> Tuple[TensorDict, torch.Tensor]:
        if use_ema and self.ema_model is not None:
            self._sync_emanet()
            model = self.ema_nets
        else:
            model = self.nets if accelerator is None else {
                key: accelerator.unwrap_model(net) for key, net in self.nets.items()
            }
        return self.get_output_with_nets_and_task_latent(
            model,
            context,
            obs,
            rng=rng,
            start_pred_t=start_pred_t,
            latent=latent,
        )

    @torch.no_grad()
    def get_output_with_nets_and_task_latent(
        self,
        nets : Dict[str, nn.Module],
        context : UMAImageContext,
        obs : UMAMaskedTokens,
        rng : Optional[torch.Generator] = None,
        start_pred_t : float = 1e-3,
        latent : Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[TensorDict, torch.Tensor]:
        pred_t = start_pred_t

        if self.is_flow_matching:
            pred_dt = 1.0 / self.timesteps_test
        else:
            pred_t = 0.0
            pred_dt = 1.0
        
        batch_size = context['image']['token_data'].shape[0]
        device = context['image']['token_data'].device

        with torch.no_grad():
            model_context, model_obs, _ = self.convert_image_to_inputs(
                context,
                obs,
            )
            normalized_obs = self.normalize(model_obs)
            normalized_obs = self.generate_inference_action_noise(
                normalized_obs,
                rng=rng,
                pred_t=pred_t
            )

            pred_t_vec = torch.full(
                (batch_size, ),
                pred_t,
                device=device,
                dtype=torch.float32
            )

            normalized_next_obs : UMANormalizedMaskedTokens = copy.deepcopy(normalized_obs)

            context_output, (_, _) = nets['feature_extractor'].context_encoder.forward(
                model_context,
                self.feature_extractor.missing_gripper_embedding,
            )
            task_latent = latent if latent is not None else nets['feature_extractor'].task_latent[None].expand(batch_size, -1, -1)
            context_output = UMAContextEncoder.stack_spec_tokens(
                task_latent,
                torch.ones((batch_size, ), dtype=torch.bool, device=device),
                context_output
            )
        
        while pred_t <= PRED_T_KNOWN - pred_dt / 2:
            flow_field_pred: UMAMaskedTokens = nets['feature_extractor'].decoder.forward(
                context_output,
                normalized_next_obs,
                pred_t=pred_t_vec,
            )

            for key, obs_item in normalized_next_obs.items():
                if key.endswith("_mean"):
                    continue

                if self.is_flow_matching:
                    flow_pred = flow_field_pred.get(key)['token_data']
                    flow_pred = torch.clamp(flow_pred, min=-2.0, max=2.0)
                    normalized_next_obs[key] = dict_replace(
                        obs_item,
                        token_data=torch.where(
                            obs_item['pred_mask'][..., None],
                            obs_item['token_data'] + flow_pred * pred_dt,
                            obs_item['token_data']
                        ),
                    )
                else:
                    target_pred = flow_field_pred.get(key)['token_data']
                    normalized_next_obs[key] = dict_replace(
                        obs_item,
                        token_data=torch.where(
                            obs_item['pred_mask'][..., None],
                            target_pred,
                            obs_item['token_data']
                        ),
                    )
            
            pred_t += pred_dt
            pred_t_vec += pred_dt

        out_obs = self.unnormalize(normalized_next_obs)
        out_obs = self.convert_outputs(
            context,
            out_obs,
        )

        action_pred = out_obs['action']['token_data'][:, self.obs_horizon:self.obs_horizon+self.act_horizon]
        output = TensorDict.from_dict({
            "executable_action": action_pred,
            "future_flows": out_obs['flow']['token_data'][:, self.obs_horizon:self.obs_horizon+self.act_horizon],
            "future_task_flows": out_obs['flow']['token_data'][:, self.obs_horizon:],
            **out_obs
        }, batch_size=(action_pred.shape[0], ))
        return output, task_latent

    @torch.no_grad()
    def get_output_with_nets(
        self,
        nets : Dict[str, nn.Module],
        context : UMAImageContext,
        obs : UMAMaskedTokens,
        rng : Optional[torch.Generator] = None,
        start_pred_t : float = 1e-3,
        **kwargs
    ) -> TensorDict:
        pred_t = start_pred_t

        if self.is_flow_matching:
            pred_dt = 1.0 / self.timesteps_test
        else:
            pred_t = 0.0
            pred_dt = 1.0
        
        batch_size = context['image']['token_data'].shape[0]
        device = context['image']['token_data'].device

        with torch.no_grad():
            model_context, model_obs, _ = self.convert_image_to_inputs(
                context,
                obs,
            )
            normalized_obs = self.normalize(model_obs)
            normalized_obs = self.generate_inference_action_noise(
                normalized_obs,
                rng=rng,
                pred_t=pred_t
            )

            pred_t_vec = torch.full(
                (batch_size, ),
                pred_t,
                device=device,
                dtype=torch.float32
            )

            normalized_next_obs : UMANormalizedMaskedTokens = copy.deepcopy(normalized_obs)

            context_output, (spec_tokens, spec_token_valids) = nets['feature_extractor'].context_encoder.forward(
                model_context,
                self.feature_extractor.missing_gripper_embedding,
            )
            context_output = UMAContextEncoder.stack_spec_tokens(
                spec_tokens[:, 0],
                spec_token_valids[:, 0],
                context_output
            )
        
        while pred_t <= PRED_T_KNOWN - pred_dt / 2:
            flow_field_pred: UMAMaskedTokens = nets['feature_extractor'].decoder.forward(
                context_output,
                normalized_next_obs,
                pred_t=pred_t_vec,
            )

            for key, obs_item in normalized_next_obs.items():
                if key.endswith("_mean"):
                    continue

                if self.is_flow_matching:
                    flow_pred = flow_field_pred.get(key)['token_data']
                    normalized_next_obs[key] = dict_replace(
                        obs_item,
                        token_data=torch.where(
                            obs_item['pred_mask'][..., None],
                            obs_item['token_data'] + flow_pred * pred_dt,
                            obs_item['token_data']
                        ),
                    )
                else:
                    target_pred = flow_field_pred.get(key)['token_data']
                    normalized_next_obs[key] = dict_replace(
                        obs_item,
                        token_data=torch.where(
                            obs_item['pred_mask'][..., None],
                            target_pred,
                            obs_item['token_data']
                        ),
                    )
            
            pred_t += pred_dt
            pred_t_vec += pred_dt

        out_obs = self.unnormalize(normalized_next_obs)
        out_obs = self.convert_outputs(
            context,
            out_obs,
        )

        action_pred = out_obs['action']['token_data'][:, self.obs_horizon:self.obs_horizon+self.act_horizon]
        output = TensorDict.from_dict({
            "executable_action": action_pred,
            "future_flows": out_obs['flow']['token_data'][:, self.obs_horizon:self.obs_horizon+self.act_horizon],
            "future_task_flows": out_obs['flow']['token_data'][:, self.obs_horizon:],
            **out_obs
        }, batch_size=(action_pred.shape[0], ))
        return output

    def _additional_times_on(self, device : Union[torch.device, str]) -> torch.Tensor:
        """``additional_times`` as a tensor on ``device``, built once (not a host-to-device copy per step)."""
        cache = self.__dict__.setdefault("_additional_times_cache", {})
        key = str(device)
        if key not in cache:
            cache[key] = torch.tensor(self.additional_times, device=device)
        return cache[key]

    def sample_timesteps(self, size : Union[int, Sequence[int]], device : Union[torch.device, str], dtype : torch.dtype):
        """
        https://github.com/huggingface/lerobot/blob/a5727e37b4ff405f9d8de424e07dcc441aa2c82f/lerobot/common/policies/pi0/modeling_pi0.py#L501
        """
        time_beta = sample_beta_on_device(1.0, 1.5, size, dtype=dtype, device=device)
        # Never exactly 0 (pure noise) or above max_train_timestep (default 1: pure data).
        time = time_beta * (self.max_train_timestep - 0.001) + 0.001
        return time

    def generate_inference_action_noise(
        self,
        obs : UMANormalizedMaskedTokens,
        rng : Optional[torch.Generator] = None,
        pred_t : float = 0.0,
    ) -> UMANormalizedMaskedTokens:
        B = obs['action']['token_data'].shape[0] if obs['action'] is not None else obs['flow']['token_data'].shape[0]
        input_obs = {}

        for key, token in obs.items():
            if key.endswith("_mean"):
                input_obs[key] = token
                continue

            if self.is_flow_matching:
                noise = torch.randn(
                    token['token_data'].shape,
                    device=token['token_data'].device,
                    dtype=token['token_data'].dtype,
                    generator=rng,
                )
                noised_input = (1.0 - pred_t) * noise + pred_t * token['token_data']
                fill_value = noise
                pred_value = noised_input
            else:
                # Direct prediction uses deterministic (zero) mask tokens, not sampled noise.
                fill_value = torch.zeros_like(token['token_data'])
                pred_value = fill_value

            input_obs[key] = dict_replace(
                token,
                token_data=torch.where(
                    torch.logical_not(token['valid_mask'][..., None]), 
                    fill_value,
                    torch.where(
                        token['pred_mask'][..., None],
                        pred_value,
                        token['token_data']
                    )
                ),
            )
        return input_obs

    def generate_timesteps_and_noisy_actions(
        self,
        obs : UMANormalizedMaskedTokens,
    ) -> Tuple[UMANormalizedMaskedTokens, UMANormalizedMaskedTokens, torch.Tensor]:
        B = obs['action']['token_data'].shape[0] if obs['action'] is not None else obs['flow']['token_data'].shape[0]

        device = obs['action']['token_data'].device if obs['action'] is not None else obs['flow']['token_data'].device
        dtype = obs['action']['token_data'].dtype if obs['action'] is not None else obs['flow']['token_data'].dtype

        if self.is_flow_matching:
            timesteps = self.sample_timesteps(
                B,
                device=device,
                dtype=dtype
            ) # (B, )
        else:
            timesteps = torch.zeros(B, device=device, dtype=dtype) # (B, )

        input_obs = {}
        goal_obs = {}

        for key, token in obs.items():
            if key.endswith("_mean"):
                input_obs[key] = token
                goal_obs[key] = token
                continue

            if self.is_flow_matching:
                token_noise = torch.randn_like(token['token_data'])
                timesteps_full = timesteps.reshape(
                    B,
                    *([1] * (len(token['token_data'].shape) - 1)),
                )
                noisy_data = (1.0 - timesteps_full) * token_noise + timesteps_full * token['token_data']
                fill_value = token_noise
                pred_value = noisy_data
            else:
                # Direct prediction uses deterministic (zero) mask tokens, not sampled noise.
                fill_value = torch.zeros_like(token['token_data'])
                pred_value = fill_value
            
            input_obs[key] = dict_replace(
                token,
                token_data=torch.where(
                    torch.logical_not(token['valid_mask'][..., None]),
                    fill_value,
                    torch.where(
                        token['pred_mask'][..., None],
                        pred_value,
                        token['token_data']
                    ),
                ),
            )
            
            if self.is_flow_matching:
                residual = token['token_data'] - token_noise
                if token['pred_mask'] is not None:
                    residual = torch.where(
                        token['pred_mask'][..., None] & token['valid_mask'][..., None],
                        residual, 
                        0
                    )
                goal_obs[key] = dict_replace(
                    token,
                    token_data=residual
                )
            else:
                goal_obs[key] = token

        return input_obs, goal_obs, timesteps

    def compute_losses(self, context : UMAImageContext, obs : UMAMaskedTokens, ref_action, additional_context_items : Sequence[Union[UMAFlowSpecContextItem, UMAContextItem]], *args, **kwargs):
        with torch.no_grad():
            model_context, model_obs, model_additional_context_items = self.convert_image_to_inputs(
                context,
                obs,
                *additional_context_items
            )
            if self.apply_se3_augmentation:
                B = model_context['flow_spec_context']['initial_pointcloud'].shape[0]
                device = model_context['flow_spec_context']['initial_pointcloud'].device
                dtype = model_context['flow_spec_context']['initial_pointcloud'].dtype
                flow_spec_context_indices = [
                    i for i, item in enumerate(model_additional_context_items)
                    if 'token_data' not in item.keys()
                ]

                model_additional_context_items = list(model_additional_context_items)
                if len(flow_spec_context_indices) > 0:
                    aug_matrices = random_se3_augmentation_matrix(
                        batch_size=B * len(flow_spec_context_indices),
                        device=device,
                        dtype=dtype,
                        translation_std=self.se3_translation_std,
                        mirror_prob=self.se3_mirror_prob,
                        max_rotation_angle=self.se3_rotation_max_angle,
                    ).reshape(len(flow_spec_context_indices), B, 4, 4)

                    for aug_i, item_i in enumerate(flow_spec_context_indices):
                        model_additional_context_items[item_i] = UMAFlowSpecInputContext.transform_3d(
                            model_additional_context_items[item_i], aug_matrices[aug_i]
                        )

            # Scene-matched hard negatives for the contrastive loss: same initial
            # scene, semantically different trajectory. Appended after the SE3
            # augmentation so they stay aligned with the (unaugmented) main spec.
            model_additional_context_items = list(model_additional_context_items)
            contrastive_hard_negative_flags = [False] * (1 + len(model_additional_context_items))
            if self.contrastive_hard_negative_specs and not self.prompt_tuning:
                main_spec_context = model_context['flow_spec_context']
                model_additional_context_items.append(spec_time_reverse(main_spec_context))
                model_additional_context_items.append(spec_freeze(main_spec_context))
                contrastive_hard_negative_flags.extend([True, True])

            # Motion-readout targets, ordered like the spec tokens: main spec
            # first, then each additional item. Language items get an invalid
            # target; keypoint items (including hard negatives) get the mean
            # displacement of their own (possibly transformed) trajectory.
            motion_targets = []
            motion_target_valids = []
            for item in (model_context['flow_spec_context'], *model_additional_context_items):
                if 'token_data' not in item.keys():
                    target_i, target_valid_i = spec_mean_displacement(item)
                else:
                    ref = model_context['flow_spec_context']['flow_spec']
                    target_i = torch.zeros(ref.shape[0], 3, device=ref.device, dtype=ref.dtype)
                    target_valid_i = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)
                motion_targets.append(target_i)
                motion_target_valids.append(target_valid_i)
            motion_targets = torch.stack(motion_targets, dim=1) # (B, S, 3)
            motion_target_valids = torch.stack(motion_target_valids, dim=1) # (B, S)
            contrastive_hard_negative_mask = torch.tensor(
                contrastive_hard_negative_flags, dtype=torch.bool
            ).pin_memory().to(motion_targets.device, non_blocking=True)

            norm_obs = self.normalize(model_obs)
            input_norm_obs, goal_norm_obs, pred_t = self.generate_timesteps_and_noisy_actions(
                norm_obs
            )
        
        output = self.nets['feature_extractor'].forward(
            model_context,
            input_norm_obs,
            pred_t,
            *model_additional_context_items,
            prompt_tuning=self.prompt_tuning,
        )

        losses = self.aux_loss_fn(
            model_context,
            output,
            goal_norm_obs,
            *args,
            is_flow_matching=self.is_flow_matching,
            contrastive_hard_negative_mask=contrastive_hard_negative_mask,
            motion_readout_targets=(motion_targets, motion_target_valids),
            **kwargs
        ) if not self.prompt_tuning else self.policy_loss_fn(
            output[0],
            goal_norm_obs,
        )
        total_loss = 0
        for key, weight in self.loss_weights.items():
            if key in losses:
                total_loss += losses[key] * weight
        losses['total_loss'] = total_loss
        losses.update(self._action_metrics(output[0], input_norm_obs, norm_obs, pred_t))
        return losses

    @torch.no_grad()
    def _action_metrics(
        self,
        norm_output : UMANormalizedMaskedTokens,
        input_norm_obs : UMANormalizedMaskedTokens,
        norm_obs : UMANormalizedMaskedTokens,
        pred_t : torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Rotation error and gripper side agreement of the predicted *clean* action, in fp32.

        With flow matching the network predicts the velocity ``x1 - noise``; the
        clean action estimate is ``x_t + (1 - t) * v_hat``
        (``x_t = (1 - t) * noise + t * x1``). Without, the output is the action.
        Rotations are compared as 6D in the current-timestep frame (the
        normalization rescales positions only).
        """
        if norm_output.get('action') is None or norm_obs.get('action') is None:
            return {}
        token = norm_obs['action']
        with torch.autocast(device_type=token['token_data'].device.type, enabled=False):
            pred = norm_output['action']['token_data'].float()
            if self.is_flow_matching:
                t = pred_t.float().reshape(-1, *([1] * (pred.ndim - 1)))
                pred = input_norm_obs['action']['token_data'].float() + (1.0 - t) * pred
            target = token['token_data'].float()
            mask = torch.logical_and(token['pred_mask'], token['valid_mask'])
            metrics = {}
            horizon = self.obs_horizon + self.pred_horizon
            spans = [("", slice(None, horizon))]
            if len(self.additional_times) > 0:
                spans.append(("far_", slice(horizon, None)))
            for prefix, span in spans:
                metrics[f"{prefix}action_rotation_angle_error_deg"] = masked_rotation_angle_error_deg(
                    pred[:, span, ..., 3:-1], target[:, span, ..., 3:-1], valid_mask=mask[:, span],
                )
                metrics[f"{prefix}action_gripper_same_sign_prob"] = masked_same_sign_prob(
                    # Gripper is 0 (open) .. 1 (closed): compare sides of the midpoint.
                    pred[:, span, ..., -1] - 0.5, target[:, span, ..., -1] - 0.5, valid_mask=mask[:, span],
                )
            # Degenerate 6D rotations (which the angle metric clamps).
            metrics["action_rotation_degenerate_frac_pred"] = masked_rotation_6d_degenerate_frac(
                pred[:, :horizon, ..., 3:-1], valid_mask=mask[:, :horizon],
            )
            metrics["action_rotation_degenerate_frac_target"] = masked_rotation_6d_degenerate_frac(
                target[:, :horizon, ..., 3:-1], valid_mask=mask[:, :horizon],
            )
        return metrics

    def _normalization(self, absolute_timesteps : torch.Tensor, sqrt_timesteps : torch.Tensor):
        """Position std and clip, in normalized units, per token position: ``(flow_std, action_std, action_clip,
        flow_clip)``. ``action_position_std`` applies to actions only. Clipping is the same in metres either way:
        ``clip_position * (|t| + 1)``."""
        if self.action_position_std is None:
            flow_std = self.max_flow_velocity * sqrt_timesteps[None, :, None, None]
            action_std = self.max_gripper_velocity * sqrt_timesteps[None, :, None]
            action_clip = (self.action_clip_position / self.max_gripper_velocity) * sqrt_timesteps[None, :, None] if self.action_clip_position > 0 else None
            flow_clip = (self.flow_clip_position / self.max_flow_velocity) * sqrt_timesteps[None, :, None, None] if self.flow_clip_position > 0 else None
            return flow_std, action_std, action_clip, flow_clip
        std = self.action_position_std.to(absolute_timesteps.device)
        if std.shape[0] != absolute_timesteps.shape[0]:
            raise ValueError(f"action_position_std has {std.shape[0]} entries for {absolute_timesteps.shape[0]} token positions")
        action_clip = (self.action_clip_position * absolute_timesteps / std)[None, :, None] if self.action_clip_position > 0 else None
        # Tracked points keep the velocity-based std: occluded points carry up to 0.2 m of augmentation noise,
        # which a millimetre-scale std would blow up.
        flow_std = self.max_flow_velocity * sqrt_timesteps[None, :, None, None]
        flow_clip = (self.flow_clip_position / self.max_flow_velocity) * sqrt_timesteps[None, :, None, None] if self.flow_clip_position > 0 else None
        return flow_std, std[None, :, None], action_clip, flow_clip

    def normalize(
        self,
        obs : UMAMaskedTokens,
        clip : bool = True,
    ) -> UMANormalizedMaskedTokens:
        self._assert_current_timestep_valid(obs)

        flow_current = obs['flow']['token_data'][:, self.obs_horizon - 1]
        action_current = obs['action']['token_data'][:, self.obs_horizon - 1]
        
        full_timesteps = torch.concat([
            torch.arange(-self.obs_horizon + 1, self.pred_horizon + 1, device=flow_current.device),
            self._additional_times_on(flow_current.device)
        ], dim=0) # (L, )
        absolute_timesteps = torch.abs(full_timesteps).float() + 1 # (L, )
        # sqrt-time std: real displacement grows sub-linearly with horizon (objects
        # move, then stop), so max_velocity * sqrt(|t| + 1) keeps target magnitudes
        # near-uniform across the whole horizon, including far-future tokens. A
        # constant std would put far-future targets at O(|t|) and swamp the
        # flow-matching noise scale; a linear std would shrink them to O(0.1) and
        # weaken their supervision. The clip preserves the same limit in meters as
        # the constant-std formulation: clip_position * (|t| + 1).
        sqrt_timesteps = torch.sqrt(absolute_timesteps)
        flow_std, action_std, action_clip, flow_clip = self._normalization(absolute_timesteps, sqrt_timesteps)

        # Normalize after any frame conversion: positions stay anchored at the current
        # timestep, and rotations are expressed relative to the current end-effector
        # orientation in the same frame.
        action_relative = action_rotation_to_relative(
            obs['action']['token_data'],
            action_current[..., 3:ACTION_POSE_DIM],
        )
        action_valid = obs['action']['valid_mask']
        if self.hide_proprio_history:
            action_valid = action_valid.clone()
            action_valid[:, :self.obs_horizon - 1] = False

        obs = UMANormalizedMaskedTokens(
            action=UMATokenItem.normalize(
                dict_replace(obs['action'], token_data=action_relative, valid_mask=action_valid),
                mean=action_current[:, None, ..., :3],
                std=action_std[..., :3],
                clip=action_clip if clip else None,
            ),
            flow=UMATokenItem.normalize(
                obs['flow'],
                mean=flow_current[:, None, ..., :3],
                std=flow_std[..., :3],
                clip=flow_clip if clip else None,
            ),
            action_mean=action_current[..., :ACTION_POSE_DIM],
            flow_mean=flow_current[..., :3],
        )
        
        return obs

    def _assert_current_timestep_valid(
        self,
        obs : UMAMaskedTokens,
    ) -> None:
        current_idx = self.obs_horizon - 1
        for key in ("action", "flow"):
            token = obs.get(key)
            if token is None:
                continue

            valid_mask = token['valid_mask'].bool()
            if valid_mask.ndim < 2:
                raise AssertionError(
                    f"{key} valid_mask must include a time dimension; got shape {tuple(valid_mask.shape)}"
                )
            if valid_mask.shape[1] <= current_idx:
                raise AssertionError(
                    f"{key} valid_mask time dimension {valid_mask.shape[1]} is too short for current index {current_idx}"
                )

            current_valid = valid_mask[:, current_idx]
            any_valid = valid_mask.any(dim=1)
            # Checked on the device (no host sync per step); a violation fails the
            # next synchronizing CUDA call with this message.
            torch._assert_async(
                (current_valid == any_valid).all(),
                f"{key} current timestep validity mismatch: expected "
                f"valid_mask[:, {current_idx}] == valid_mask.any(dim=1) so every active "
                f"trajectory/keypoint has a valid normalization anchor",
            )
     
    def unnormalize(
        self,
        obs : UMANormalizedMaskedTokens,
        clip : bool = True,
    ) -> UMAMaskedTokens:
        flow_current = obs['flow_mean']
        action_current = obs['action_mean']

        full_timesteps = torch.concat([
            torch.arange(-self.obs_horizon + 1, self.pred_horizon + 1, device=flow_current.device),
            self._additional_times_on(flow_current.device)
        ], dim=0) # (L, )
        absolute_timesteps = (torch.abs(full_timesteps) + 1).float() # (L, )
        # Must mirror normalize(): sqrt-time std, clip equivalent to
        # clip_position * (|t| + 1) meters.
        sqrt_timesteps = torch.sqrt(absolute_timesteps)
        flow_std, action_std, action_clip, flow_clip = self._normalization(absolute_timesteps, sqrt_timesteps)

        action = UMATokenItem.unnormalize(
            obs['action'],
            mean=action_current[:, None, ..., :3],
            std=action_std[..., :3],
            clip=action_clip if clip else None,
        )
        # Rotations back from relative-to-current to absolute (still in the camera frame; convert_outputs returns
        # the actions to the robot base frame).
        action = dict_replace(
            action,
            token_data=action_rotation_to_absolute(
                action['token_data'],
                action_current[..., 3:ACTION_POSE_DIM],
            ),
        )

        obs = UMAMaskedTokens(
            action=action,
            flow=UMATokenItem.unnormalize(
                obs['flow'],
                mean=flow_current[:, None, ..., :3],
                std=flow_std[..., :3],
                clip=flow_clip if clip else None
            ),
        )
        return obs
    
    # no_grad, not inference_mode: training feeds these outputs to the (compiled) network, and autograd
    # cannot save inference tensors for backward. A clone inside the network does not help under
    # torch.compile, whose partitioner saves the graph input itself.
    @torch.no_grad()
    def convert_image_to_inputs(
        self,
        context : UMAImageContext,
        masked_tokens : UMANormalizedMaskedTokens,
        *additional_context_items : Union[UMAFlowSpecContextItem, UMAContextItem],
    ):
        input_context = UMAInputContext.from_image_context(
            context,
            n_points=self.pointcloud_n_points,
            max_language_length=self.max_language_len,
            encoder_type=self.encoder_type,
            rng=self.rng,
        )
        # The model works in the camera frame: tracked flows already are; the absolute action poses (robot base
        # frame) are put into the camera frame here, and normalize() then expresses rotations relative to the
        # current timestep in that frame.
        input_masked_tokens = dict_replace(
            masked_tokens,
            action=UMATokenItem.transform_3d(
                masked_tokens['action'],
                context['extrinsics']['token_data'],
                is_action=True,
            )
        )
        input_additional_context_items = []
        for item in additional_context_items:
            if "token_data" not in item.keys():
                input_additional_context_items.append(
                    UMAFlowSpecInputContext.from_flow_spec_context(
                        item,
                        n_points=self.pointcloud_n_points,
                        encoder_type=self.encoder_type,
                        rng=self.rng,
                    )
                )
            else:
                tokenized_language = tokenize_languages(
                    item['token_data'].tolist(),
                    max_language_length=self.max_language_len,
                    device=item['valid_mask'].device,
                    encoder_type=self.encoder_type,
                )
                input_additional_context_items.append(
                    dict_replace(
                        tokenized_language,
                        valid_mask=tokenized_language['valid_mask'] & item['valid_mask'][:, None].bool(),
                    )
                )
        return input_context, input_masked_tokens, input_additional_context_items
    
    @torch.inference_mode()
    def convert_outputs(
        self,
        context : UMAImageContext,
        outputs : UMAMaskedTokens,
    ) -> UMAMaskedTokens:
        # Executable actions go back to the robot base frame; flows stay in the camera frame.
        return dict_replace(
            outputs,
            action=UMATokenItem.transform_3d(
                outputs['action'],
                context['extrinsics']['token_data'],
                is_action=True,
                inv=True,
            )
        )
