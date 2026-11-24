from typing import Type, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    and_masks,
    or_masks
)

from xbarray.transformations.pointcloud.pytorch import (
    pixel_coordinate_and_depth_to_world,
    bilinear_interpolate
)
from uma.network.attention import TransformerBlock, FlexAttention, Attention
from uma.network.positional_embedding import TimestepPosEmb
from uma.network.dynamic_tanh import DynamicTanh
from uma.network.spatio_temporal_transformer import SpatioTemporalTransformer
from uma.tokenizers.language import EncoderType, tokenize_patch_features, tokenize_languages, get_image_embedding_dim, get_language_embedding_dim, upsample_patch_features
from uma.tokenizers.image_network import KeypointFeatureExtractor, ImageTokenizer
from .pointcloud_fusion import PointCloudPatchFusion
from .input_output import (
    UMAContextItem,
    UMAFlowSpecInputContext,
)

__all__ = [
    "SPEC_DISPLACEMENT_SCALE",
    "spec_first_valid_anchor",
    "KeypointSpecificationEncoder",
    "LanguageSpecificationEncoder",
    "SpecificationEncoder",
]

# Scale applied to spec displacements (meters) so they enter the network at
# O(1) instead of being a small fraction of the input feature variance.
SPEC_DISPLACEMENT_SCALE = 10.0

def spec_first_valid_anchor(
    flow_spec : torch.Tensor, # (B, T_fs, M_fs, 3)
    valid_mask : torch.Tensor, # (B, T_fs, M_fs)
) -> torch.Tensor:
    """Position of each keypoint at its first valid timestep, shape (B, 1, M_fs, 3).

    Keypoints with no valid timestep fall back to index 0; their tokens are
    masked out downstream so the value never matters.
    """
    first_valid_idx = torch.argmax(valid_mask.int(), dim=1) # (B, M_fs)
    return torch.gather(
        flow_spec,
        dim=1,
        index=first_valid_idx[:, None, :, None].expand(-1, 1, -1, flow_spec.shape[-1])
    ) # (B, 1, M_fs, 3)

class KeypointSpecificationEncoder(nn.Module):
    def __init__(
        self,
        n_spec_tokens : int,
        dim : int,
        num_layers_stt: int,
        num_layers_agg : int,
        encoder_type: EncoderType,
        flow_spec_dim : int,
        point_feature_dim : int,
        visual_feature_dim : int,
        flow_spec_time_embedding_dim: int = 64,
    ) -> None:
        super().__init__()

        self.encoder_type = encoder_type

        self.flow_spec_time_encoding = TimestepPosEmb(
            flow_spec_time_embedding_dim,
            min_period=0.004,
            max_period=4
        )
        self.pointcloud_fusion = PointCloudPatchFusion(
            point_feature_dim=point_feature_dim,
            visual_feature_dim=visual_feature_dim,
            token_dim=dim,
        )
        # Input per token: absolute position (kept in the same frame as the
        # initial pointcloud for cross-attention grounding) + displacement from
        # the keypoint's first valid position, scaled to O(1). The displacement
        # carries the task-relevant motion direction explicitly.
        self.flow_spec_projection = nn.Sequential(
            nn.Linear(2 * flow_spec_dim + flow_spec_time_embedding_dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.stt = SpatioTemporalTransformer(
            dim=dim,
            num_layers=num_layers_stt,
            num_heads=dim // 64,
            has_context=True, 
            in_dim_context=dim,
        )
        self.agg_sa = nn.ModuleList([
            TransformerBlock(
                Attention(
                    dim=dim,
                    num_heads=dim // 64,
                ),
                has_context=False,
                norm_layer=nn.LayerNorm,
            )
            for i_layer in range(num_layers_agg)
        ])
        self.agg_ca = nn.ModuleList([
            TransformerBlock(
                Attention(
                    dim=dim,
                    num_heads=dim // 64,
                ),
                has_context=True,
                norm_layer=nn.LayerNorm,
            )
            for i_layer in range(num_layers_agg)
        ])
        self.agg_token = nn.Parameter(torch.randn((n_spec_tokens, dim)))
        nn.init.trunc_normal_(self.agg_token, std=0.02)
    
    def forward_agg(
        self,
        context_features : torch.Tensor, # (B, N, dim)
        context_valids : torch.Tensor, # (B, N)
    ) -> torch.Tensor:
        B, N, _ = context_features.shape
        N_spec = self.agg_token.shape[0]
        agg_token = self.agg_token.unsqueeze(0).expand(B, -1, -1) # (B, n_spec_tokens, dim)

        for sa_layer, ca_layer in zip(self.agg_sa, self.agg_ca):
            agg_token = ca_layer(
                agg_token,
                context=context_features,
                attention_mask=context_valids[:, None, None, :].expand(-1, self.agg_sa[0].num_heads, N_spec, -1) # (B, num_heads, n_spec_tokens, N)
            )
            agg_token = sa_layer(
                agg_token
            )
        return agg_token # (B, N_spec, dim)

    def forward(
        self,
        flow_spec_context : UMAFlowSpecInputContext,
        pointcloud_encoder : nn.Module,
    ):
        pointcloud_features, pointcloud_valids = self.pointcloud_fusion(
            pointcloud_encoder,
            flow_spec_context['initial_pointcloud'],
            flow_spec_context['initial_patch_features']['token_data'],
            flow_spec_context['initial_image_size'],
            flow_spec_context['initial_intrinsics'],
            # Specs are built in the camera frame; an SE(3)-augmented spec carries the projection back to it.
            None if flow_spec_context['pointcloud_in_cam_frame'] else flow_spec_context['initial_extrinsics'],
            image_valid_mask=flow_spec_context['initial_patch_features']['valid_mask'],
        ) # (B, N_groups, token_dim), (B, N_groups)

        flow_spec_features = flow_spec_context['flow_spec']  # (B, T_fs, M_fs, flow_spec_dim (usually = 3))
        B, T_fs, M_fs, _ = flow_spec_features.shape
        flow_spec_times = torch.clamp(flow_spec_context['valid_mask'].any(dim=-1).float().cumsum(dim=1) - 1, min=0) # (B, T_fs)
        flow_spec_times = flow_spec_times / torch.clamp(flow_spec_times[:, -1:], min=1.0) # Normalize to [0, 1]
        flow_spec_time_embedding = self.flow_spec_time_encoding.compute_embedding_with_position(flow_spec_times)[..., None, :].expand(-1, -1, M_fs, -1) # (B, T_fs, M_fs, flow_spec_time_embedding_dim)

        flow_spec_anchor = spec_first_valid_anchor(
            flow_spec_features, flow_spec_context['valid_mask']
        ) # (B, 1, M_fs, flow_spec_dim)
        flow_spec_displacement = (flow_spec_features - flow_spec_anchor) * SPEC_DISPLACEMENT_SCALE

        flow_spec_features = self.flow_spec_projection(torch.concat([
            flow_spec_features,
            flow_spec_displacement,
            flow_spec_time_embedding,
        ], dim=-1)) # (B, T_fs, M_fs, dim)
        flow_spec_aggregated_features, flow_spec_keypoints_features, flow_spec_time_features, raw_features, raw_mask = self.stt.forward(
            flow_spec_features,
            flow_spec_context['valid_mask'],
            context=pointcloud_features,
            context_mask=pointcloud_valids,
        ) # (B, token_dim), (B, M_fs, token_dim), (B, T_fs, token_dim), (B, T_fs + 1, M_fs + 1, token_dim), (B, T_fs + 1, M_fs + 1)

        context_features = torch.concat([
            flow_spec_aggregated_features[:, None, :], # (B, 1, token_dim)
            flow_spec_time_features, # (B, T_fs, token_dim)
        ], dim=1) # (B, T_fs + 1, token_dim)
        context_valids = torch.concat([
            torch.ones(B, 1, device=context_features.device, dtype=torch.bool), # (B, 1)
            flow_spec_context['valid_mask'].any(dim=-1), # (B, T_fs)
        ], dim=1) # (B, T_fs + 1)
        flow_spec_aggregated_features = self.forward_agg(
            context_features,
            context_valids
        )
        return (
            flow_spec_aggregated_features, # (B, N_spec, token_dim)
            torch.any(
                flow_spec_context['valid_mask'].flatten(1), dim=-1
            ) # (B, )
        )
    
class LanguageSpecificationEncoder(nn.Module):
    def __init__(
        self,
        n_spec_tokens : int,
        dim : int,
        num_layers_agg: int,
        encoder_type: EncoderType,
    ) -> None:
        super().__init__()

        self.encoder_type = encoder_type

        self.agg_sa = nn.ModuleList([
            TransformerBlock(
                Attention(
                    dim=dim,
                    num_heads=dim // 64,
                ),
                has_context=False,
                norm_layer=nn.LayerNorm,
            )
            for i_layer in range(num_layers_agg)
        ])
        self.agg_ca = nn.ModuleList([
            TransformerBlock(
                Attention(
                    dim=dim,
                    num_heads=dim // 64,
                    in_dim_context=get_language_embedding_dim(encoder_type),
                ),
                has_context=True,
                norm_layer=nn.LayerNorm,
            )
            for i_layer in range(num_layers_agg)
        ])
        self.agg_token = nn.Parameter(torch.randn((n_spec_tokens, dim)))
        nn.init.trunc_normal_(self.agg_token, std=0.02)
        self.cls_token = nn.Parameter(torch.randn(get_language_embedding_dim(encoder_type)))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    @torch.compile
    def forward_agg(
        self,
        context_features : torch.Tensor, # (B, N, dim)
        context_valids : torch.Tensor, # (B, N)
    ) -> torch.Tensor:
        B, N, _ = context_features.shape
        N_spec = self.agg_token.shape[0]
        agg_token = self.agg_token.unsqueeze(0).expand(B, -1, -1) # (B, n_spec_tokens, dim)

        for sa_layer, ca_layer in zip(self.agg_sa, self.agg_ca):
            agg_token = ca_layer(
                agg_token,
                context=context_features,
                attention_mask=context_valids[:, None, None, :].expand(-1, self.agg_sa[0].num_heads, N_spec, -1) # (B, num_heads, n_spec_tokens, N)
            )
            agg_token = sa_layer(
                agg_token
            )
        return agg_token # (B, N_spec, dim)

    @torch.compile
    def forward(
        self,
        language_token : UMAContextItem, # (B, T_lang, language_token_dim)
    ) -> Tuple[
        torch.Tensor, # data
        torch.Tensor, # valid mask
    ]:
        language_features, language_valids = language_token['token_data'], language_token['valid_mask'] # (B, T_lang, language_embedding_dim), (B, T_lang)
        B = language_features.shape[0]
        
        context_features = torch.cat([
            self.cls_token[None, None, :].expand(B, -1, -1), # (B, 1, language_embedding_dim)
            language_features, # (B, T_lang, language_embedding_dim)
        ], dim=1) # (B, T_lang + 1, language_embedding_dim)
        context_valids = torch.cat([
            torch.ones(B, 1, device=language_valids.device, dtype=language_valids.dtype), # (B, 1)
            language_valids, # (B, T_lang)
        ], dim=1).bool() # (B, T_lang + 1)
        language_aggregated_features = self.forward_agg(
            context_features,
            context_valids
        ) # (B, N_spec, dim)
        return (
            language_aggregated_features, # (B, N_spec, dim)
            torch.any(language_valids, dim=-1) # (B, )
        )

class SpecificationEncoder(nn.Module):
    def __init__(
        self,
        dim : int,
        n_spec_tokens : int,
        num_layers_stt : int,
        num_layers_agg : int,
        num_layers_language : int,
        encoder_type: EncoderType,
        flow_spec_dim : int,
        point_feature_dim : int,
        visual_feature_dim : int,
        flow_spec_time_embedding_dim: int = 64,
        max_language_length: int = 64,
    ) -> None:
        super().__init__()

        self.keypoint_spec_encoder = KeypointSpecificationEncoder(
            n_spec_tokens=n_spec_tokens,
            dim=dim,
            num_layers_stt=num_layers_stt,
            num_layers_agg=num_layers_agg,
            encoder_type=encoder_type,
            flow_spec_dim=flow_spec_dim,
            point_feature_dim=point_feature_dim,
            visual_feature_dim=visual_feature_dim,
            flow_spec_time_embedding_dim=flow_spec_time_embedding_dim,
        )
        self.language_spec_encoder = LanguageSpecificationEncoder(
            n_spec_tokens=n_spec_tokens,
            dim=dim,
            num_layers_agg=num_layers_language,
            encoder_type=encoder_type,
        )
    
    def forward_keypoint(
        self,
        flow_spec_context : UMAFlowSpecInputContext, # (B, T_fs, M_fs, flow_spec_dim)
        pointcloud_encoder : nn.Module,
    ):
        return self.keypoint_spec_encoder.forward(
            flow_spec_context=flow_spec_context,
            pointcloud_encoder=pointcloud_encoder,
        )
    
    def forward_language(
        self,
        language_token : UMAContextItem, # (B, T_lang, language_token_dim)
    ):
        return self.language_spec_encoder.forward(
            language_token=language_token,
        )
    
    def forward(
        self,
        flow_spec_context : Optional[UMAFlowSpecInputContext],
        language_token : Optional[UMAContextItem], # (B, T_lang, language_token_dim)
        pointcloud_encoder : nn.Module,
    ):
        if flow_spec_context is not None:
            keypoint_features, keypoint_valids = self.forward_keypoint(
                flow_spec_context=flow_spec_context,
                pointcloud_encoder=pointcloud_encoder,
            ) # (B, token_dim), (B, )
        else:
            keypoint_features, keypoint_valids = None, None
        
        if language_token is not None:
            language_features, language_valids = self.forward_language(
                language_token=language_token,
            ) # (B, language_embedding_dim), (B, )
        else:
            language_features, language_valids = None, None
        
        if keypoint_features is not None and language_features is not None:
            return torch.where(
                language_valids[:, None, None],
                language_features,
                keypoint_features,
            ), torch.where(
                language_valids,
                True,
                keypoint_valids,
            )
        elif keypoint_features is not None:
            return keypoint_features, keypoint_valids
        elif language_features is not None:
            return language_features, language_valids
        else:
            raise ValueError("At least one of flow_spec or language_token must be provided")
