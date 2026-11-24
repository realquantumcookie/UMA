from typing import Optional, Union, Tuple, Any, Generic, TypeVar, List, Dict, Mapping, Callable, TYPE_CHECKING
from typing_extensions import TypedDict
import math
import torch
from torch.nn import functional as F
from torchvision.transforms import v2
import numpy as np
from tensordict import TensorDict, TensorClass, NonTensorData
import functools

from unienv_interface.backends.pytorch import PyTorchComputeBackend, PyTorchArrayType, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType
from unienv_interface.space import *
from xbarray.transformations.pointcloud.pytorch import (
    depth_image_to_world,
    random_point_sampling,
    farthest_point_sampling,
    world_to_pixel_coordinate_and_depth,
    bilinear_interpolate,
    pointcloud_jitter
)
from xbarray.transformations.rotation_conversions.pytorch import *

EncoderType = str

DEFAULT_FLOW_SPEC_DIM = 3
DEFAULT_FLOW_DIM = 4 # x, y, z, visibility_mask
ACTION_POSITION_DIM = 3
ACTION_ROTATION_6D_DIM = 6
ACTION_POSE_DIM = ACTION_POSITION_DIM + ACTION_ROTATION_6D_DIM
DEFAULT_ACTION_DIM = ACTION_POSE_DIM + 1 # position (3), 6d rotation (6), gripper (1)

def depth_jitter(
    depth : torch.Tensor,
    std : float = 0.01,
    rng : Optional[PyTorchRNGType] = None,
) -> torch.Tensor:
    noise = torch.randn(
        depth.shape,
        device=depth.device,
        dtype=depth.dtype,
        generator=rng,
    ) * std
    return torch.where(
        torch.isnan(depth) | (depth <= 0),
        depth,
        depth + noise
    )

def random_se3_augmentation_matrix(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    translation_std: float = 0.1,
    mirror_prob: float = 0.5,
    max_rotation_angle: float = math.pi / 6,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate random SE(3) transformation matrices with optional mirroring for
    contrastive pointcloud augmentation.

    Each matrix is a 4x4 homogeneous transform. The rotation is yaw-only (about
    the gravity-aligned +z axis of the spec frame) and limited to
    ±max_rotation_angle: arbitrary SO(3) rotations (and mirroring) change task
    semantics, so using them to generate contrastive positives teaches the spec
    encoder to discard motion direction.

    Mirroring reflects the pointcloud across a randomly chosen set of coordinate
    axes (determinant = -1), which is NOT an SE(3) element. It flips task
    semantics (e.g. push-left vs push-right), so mirror_prob should stay 0 when
    the transformed specs are used as contrastive positives.

    Args:
        batch_size: Number of matrices to generate.
        device: Target torch device.
        dtype: Target torch dtype.
        translation_std: Standard deviation of random translation (meters).
        mirror_prob: Probability of applying mirroring to each batch element.
        max_rotation_angle: Maximum magnitude of the random yaw (radians).
        rng: Optional torch random generator.

    Returns:
        (B, 4, 4) homogeneous transformation matrices.
    """
    # --- Random yaw rotation about +z, uniform in [-max_rotation_angle, max_rotation_angle] ---
    angle = (torch.rand(batch_size, device=device, dtype=dtype, generator=rng) * 2.0 - 1.0) * max_rotation_angle
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    R = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
    R[:, 0, 0] = cos_a
    R[:, 0, 1] = -sin_a
    R[:, 1, 0] = sin_a
    R[:, 1, 1] = cos_a
    R[:, 2, 2] = 1.0

    # --- Random translation ---
    translation = torch.randn(batch_size, 3, device=device, dtype=dtype, generator=rng) * translation_std

    # --- Build 4x4 homogeneous matrix ---
    T = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = translation
    T[:, 3, 3] = 1.0

    # --- Optional mirroring (X and Y axes only, never mirror Z) ---
    if mirror_prob > 0.0:
        apply_mirror = torch.rand(batch_size, device=device, dtype=dtype, generator=rng) < mirror_prob
        if apply_mirror.any():
            # Only mirror X (idx 0) and Y (idx 1); Z (idx 2) is never mirrored
            mirror_axes = torch.rand(batch_size, 2, device=device, dtype=dtype, generator=rng) < 0.5
            # Guarantee at least one mirrored axis when mirroring is applied
            needs_axis = apply_mirror & (~mirror_axes.any(dim=-1))
            if needs_axis.any():
                random_axis = torch.randint(0, 2, (needs_axis.sum().item(),), device=device, generator=rng)
                mirror_axes[needs_axis, random_axis] = True

            # Build full 3-axis mirror_sign with Z always +1
            mirror_sign = torch.ones(batch_size, 3, device=device, dtype=dtype)
            mirror_sign[:, :2] = torch.where(mirror_axes, -1.0, 1.0)
            mirror_sign = torch.where(apply_mirror.unsqueeze(-1), mirror_sign, torch.ones_like(mirror_sign))
            M = torch.diag_embed(mirror_sign).to(dtype=dtype)

            # Apply mirroring to rotation and translation
            T[:, :3, :3] = torch.matmul(M, T[:, :3, :3])
            T[:, :3, 3] = T[:, :3, 3] * mirror_sign

    return T


def dict_replace(
    original_dict : Dict[str, Any],
    **replacements : Any
) -> Dict[str, Any]:
    new_dict = original_dict.copy() if isinstance(original_dict, dict) else dict(original_dict)
    for key, value in replacements.items():
        new_dict[key] = value
    return new_dict

def _is_token_item(value: Any) -> bool:
    return isinstance(value, Mapping) and 'token_data' in value and 'pred_mask' in value and 'valid_mask' in value

def _is_context_item(value: Any) -> bool:
    return isinstance(value, Mapping) and 'token_data' in value and 'valid_mask' in value and 'pred_mask' not in value

def resize_intrinsics(
    intrinsics : torch.Tensor,
    original_height : int,
    original_width : int,
    target_height : int,
    target_width : int,
) -> torch.Tensor:
    scale_y = (target_height - 1) / (original_height - 1)
    scale_x = (target_width - 1) / (original_width - 1)
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] = intrinsics[..., 0, :] * scale_x
    intrinsics[..., 1, :] = intrinsics[..., 1, :] * scale_y
    return intrinsics

def transform_3d(
    token_data : torch.Tensor,
    transformation_matrix : torch.Tensor,
    is_action : bool = False,
    action_rotation_relative : bool = False,
    inv : bool = False,
) -> torch.Tensor:
    """
    Transform the 3D points in the token data using the given transformation matrix.
    Assumes the first 3 dimensions of the token data are the 3D points.
    The transformation matrix should be of shape (..., 3, 4) or (..., 4, 4).
    Input:
        token_item: UMATokenItem with token_data of shape (..., D), where D >= 3
        transformation_matrix: torch.Tensor of shape (..., 3, 4) or (..., 4, 4)
    Output:
        UMATokenItem with transformed token_data
    """
    batch_shape = tuple(transformation_matrix.shape[:-2])
    assert token_data.shape[:len(batch_shape)] == batch_shape, (
        "token_data leading dims must match transformation_matrix batch dims"
    )
    data_sequence_shape = tuple(token_data.shape[len(batch_shape):-1])
    B_flat = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

    ones = torch.ones(tuple(token_data.shape[:-1]) + (1,), device=token_data.device, dtype=token_data.dtype)
    point_data_homogeneous = torch.cat([
        token_data[..., :3],
        ones,
    ], dim=-1)  # (..., 4)
    point_data_homogeneous_flat = point_data_homogeneous.reshape(B_flat, -1, 4)  # (B_flat, N, 4)

    transformation_matrix_flat = transformation_matrix.reshape(B_flat, *transformation_matrix.shape[-2:])  # (B_flat, 3, 4) or (B_flat, 4, 4)
    if transformation_matrix_flat.shape[-2:] == (3, 4):
        bottom_row = transformation_matrix_flat.new_zeros((B_flat, 1, 4))
        bottom_row[..., 3] = 1
        transformation_matrix_flat = torch.cat([
            transformation_matrix_flat,
            bottom_row,
        ], dim=-2)  # (B_flat, 4, 4)
    elif transformation_matrix_flat.shape[-2:] != (4, 4):
        raise ValueError(
            f"transformation_matrix must have shape (..., 3, 4) or (..., 4, 4), got {tuple(transformation_matrix.shape)}"
        )
    
    if inv:
        transformation_matrix_flat = _rigid_inverse(transformation_matrix_flat)

    transformed_points_flat = torch.matmul(
        point_data_homogeneous_flat,
        transformation_matrix_flat.transpose(-1, -2),
    )  # (B_flat, N, 4)
    transformed_points = transformed_points_flat.reshape(batch_shape + data_sequence_shape + (4,))
    transformed_xyz = transformed_points[..., :3] / transformed_points[..., 3:4]  # (..., 3)

    if is_action:
        rotation_matrix = rotation_6d_to_matrix(token_data[..., 3:9]) # (..., L, 3, 3)
        flat_rotation_matrix = rotation_matrix.reshape(B_flat, -1, 3, 3) # (B_flat, L, 3, 3)
        if action_rotation_relative:
            # Relative action rotations are invariant to the external frame change.
            transformed_rotation_6d = token_data[..., 3:9]
        else:
            transformed_rotation_matrix = transformation_matrix_flat[:, None, :3, :3] @ flat_rotation_matrix # (B_flat, L, 3, 3)
            transformed_rotation_6d = matrix_to_rotation_6d(
                transformed_rotation_matrix.reshape(batch_shape + (-1, 3, 3))
            ) # (..., L, 6)
        new_token_data = torch.concat([
            transformed_xyz,
            transformed_rotation_6d,
            token_data[..., 9:],
        ], dim=-1)
    else:
        new_token_data = torch.concat([
            transformed_xyz,
            token_data[..., 3:],
        ], dim=-1)

    new_token_data = torch.nan_to_num(
        new_token_data,
        nan=0.0
    )
    return new_token_data

def _rigid_inverse(transform: torch.Tensor) -> torch.Tensor:
    """
    Invert a rigid homogeneous transform [R|t] using rotation transpose.
    Handles both (..., 3, 4) and (..., 4, 4) shapes.
    For mixed precision safety, computes in float32 when the input is not float32/float64.
    """
    original_dtype = transform.dtype
    if original_dtype not in (torch.float32, torch.float64):
        transform = transform.float()

    if transform.shape[-2:] == (3, 4):
        R = transform[..., :3, :3]
        t = transform[..., :3, 3]
        R_inv = R.transpose(-1, -2)
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
        out = torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1)
    elif transform.shape[-2:] == (4, 4):
        R = transform[..., :3, :3]
        t = transform[..., :3, 3]
        R_inv = R.transpose(-1, -2)
        t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)
        bottom = transform.new_zeros((*transform.shape[:-2], 1, 4))
        bottom[..., 3] = 1.0
        out = torch.cat([
            torch.cat([R_inv, t_inv.unsqueeze(-1)], dim=-1),
            bottom,
        ], dim=-2)
    else:
        raise ValueError(
            f"transform must have shape (..., 3, 4) or (..., 4, 4), got {tuple(transform.shape)}"
        )

    if original_dtype not in (torch.float32, torch.float64):
        out = out.to(original_dtype)
    return out

def transform_extrinsics(
    extrinsics : torch.Tensor,
    transformation_matrix : torch.Tensor,
    inv : bool = False,
) -> torch.Tensor:
    """Transform world-to-camera extrinsics under a frame change."""
    batch_shape = tuple(transformation_matrix.shape[:-2])
    assert extrinsics.shape[:len(batch_shape)] == batch_shape, (
        "extrinsics leading dims must match transformation_matrix batch dims"
    )
    B_flat = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

    transformation_matrix_flat = transformation_matrix.reshape(B_flat, *transformation_matrix.shape[-2:])
    if transformation_matrix_flat.shape[-2:] == (3, 4):
        bottom_row = transformation_matrix_flat.new_zeros((B_flat, 1, 4))
        bottom_row[..., 3] = 1
        transformation_matrix_flat = torch.cat([transformation_matrix_flat, bottom_row], dim=-2)
    elif transformation_matrix_flat.shape[-2:] != (4, 4):
        raise ValueError(
            f"transformation_matrix must have shape (..., 3, 4) or (..., 4, 4), got {tuple(transformation_matrix.shape)}"
        )

    extrinsics_flat = extrinsics.reshape(B_flat, *extrinsics.shape[-2:])
    if extrinsics_flat.shape[-2:] == (3, 4):
        bottom_row = extrinsics_flat.new_zeros((B_flat, 1, 4))
        bottom_row[..., 3] = 1
        extrinsics_flat = torch.cat([extrinsics_flat, bottom_row], dim=-2)
    elif extrinsics_flat.shape[-2:] != (4, 4):
        raise ValueError(f"extrinsics must have shape (..., 3, 4) or (..., 4, 4), got {tuple(extrinsics.shape)}")

    if inv:
        transformed_extrinsics = extrinsics_flat @ transformation_matrix_flat
    else:
        transformed_extrinsics = extrinsics_flat @ _rigid_inverse(transformation_matrix_flat)
    return transformed_extrinsics.reshape(batch_shape + (4, 4))[..., :3, :]

def safe_rotation_6d_to_matrix(d6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """``rotation_6d_to_matrix`` with clamped norms: finite for a degenerate 6D
    (a zero half, or parallel halves), where the plain Gram-Schmidt divides by
    zero. Identical for a valid 6D rotation."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / torch.linalg.vector_norm(a1, dim=-1, keepdim=True).clamp_min(eps)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = b2 / torch.linalg.vector_norm(b2, dim=-1, keepdim=True).clamp_min(eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_degenerate(d6: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``(...)`` bool: the 6D rotation is non-finite, has a zero first half, or a
    second half parallel to it (no rotation matrix)."""
    d6 = d6.detach().float()
    a1, a2 = d6[..., :3], d6[..., 3:]
    norm1 = torch.linalg.vector_norm(a1, dim=-1)
    b1 = a1 / norm1.clamp_min(eps).unsqueeze(-1)
    norm2 = torch.linalg.vector_norm(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    return ~torch.isfinite(d6).all(dim=-1) | (norm1 < eps) | (norm2 < eps)

def _broadcast_rotation_matrix(
    reference_rotation_matrix: torch.Tensor,
    target_ndim: int,
) -> torch.Tensor:
    while reference_rotation_matrix.ndim < target_ndim:
        reference_rotation_matrix = reference_rotation_matrix.unsqueeze(-3)
    return reference_rotation_matrix

def action_rotation_to_relative(
    token_data: torch.Tensor,
    reference_rotation_6d: torch.Tensor,
) -> torch.Tensor:
    """Express 6D action rotations in the current-timestep frame.

    Computed in fp32 (also under autocast: a bf16 matmul quantizes the rotation
    targets by up to ~0.07°). Invalid (zero) tokens give finite values through
    the clamped Gram-Schmidt; their masks keep them out of the losses.
    """
    if token_data.shape[-1] < ACTION_POSE_DIM:
        return token_data

    with torch.autocast(device_type=token_data.device.type, enabled=False):
        action_rotation_matrix = safe_rotation_6d_to_matrix(
            token_data[..., ACTION_POSITION_DIM:ACTION_POSE_DIM].float()
        )
        reference_rotation_matrix = _broadcast_rotation_matrix(
            safe_rotation_6d_to_matrix(reference_rotation_6d.float()),
            action_rotation_matrix.ndim,
        )
        relative_rotation_matrix = reference_rotation_matrix.transpose(-1, -2) @ action_rotation_matrix
        relative_rotation_6d = matrix_to_rotation_6d(relative_rotation_matrix).to(token_data.dtype)
    return torch.nan_to_num(
        torch.cat([
            token_data[..., :ACTION_POSITION_DIM],
            relative_rotation_6d,
            token_data[..., ACTION_POSE_DIM:],
        ], dim=-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def action_rotation_to_absolute(
    token_data: torch.Tensor,
    reference_rotation_6d: torch.Tensor,
) -> torch.Tensor:
    """Recover absolute 6D rotations from the current-timestep frame.

    Computed in fp32 (also under autocast: a bf16 matmul quantizes the rotation
    targets by up to ~0.07°). Invalid (zero) tokens give finite values through
    the clamped Gram-Schmidt; their masks keep them out of the losses.
    """
    if token_data.shape[-1] < ACTION_POSE_DIM:
        return token_data

    with torch.autocast(device_type=token_data.device.type, enabled=False):
        action_rotation_matrix = safe_rotation_6d_to_matrix(
            token_data[..., ACTION_POSITION_DIM:ACTION_POSE_DIM].float()
        )
        reference_rotation_matrix = _broadcast_rotation_matrix(
            safe_rotation_6d_to_matrix(reference_rotation_6d.float()),
            action_rotation_matrix.ndim,
        )
        absolute_rotation_matrix = reference_rotation_matrix @ action_rotation_matrix
        absolute_rotation_6d = matrix_to_rotation_6d(absolute_rotation_matrix).to(token_data.dtype)
    return torch.nan_to_num(
        torch.cat([
            token_data[..., :ACTION_POSITION_DIM],
            absolute_rotation_6d,
            token_data[..., ACTION_POSE_DIM:],
        ], dim=-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

DataT = TypeVar("DataT")
class UMATokenItem(Generic[DataT], TypedDict):
    token_data : Union[torch.Tensor, Any]
    pred_mask : torch.Tensor
    valid_mask : torch.Tensor

    @staticmethod
    def to_device(
        token_item : "UMATokenItem",
        device : PyTorchDeviceType,
    ) -> "UMATokenItem":
        return UMATokenItem(
            token_data=token_item['token_data'].to(device) if isinstance(token_item['token_data'], torch.Tensor) else token_item['token_data'],
            pred_mask=token_item['pred_mask'].to(device),
            valid_mask=token_item['valid_mask'].to(device),
        )

    @staticmethod
    def pad(
        token_item : "UMATokenItem",
        target_sequence_shape : Tuple[int, ...],
    ) -> "UMATokenItem":
        current_sequence_shape = token_item["pred_mask"].shape
        assert len(current_sequence_shape) == len(target_sequence_shape), "Current sequence shape and target sequence shape must have the same number of dimensions"
        
        new_token_data = token_item['token_data']
        new_token_pred_mask = token_item['pred_mask']
        new_token_valid_mask = token_item['valid_mask']

        for i, (current_size, target_size) in enumerate(zip(current_sequence_shape, target_sequence_shape)):
            if current_size >= target_size:
                continue
            new_token_data = PyTorchComputeBackend.pad_dim(
                new_token_data,
                dim=i,
                target_size=target_size
            )
            new_token_pred_mask = PyTorchComputeBackend.pad_dim(
                new_token_pred_mask,
                dim=i,
                target_size=target_size,
                pad_value=False,
            )
            new_token_valid_mask = PyTorchComputeBackend.pad_dim(
                new_token_valid_mask,
                dim=i,
                target_size=target_size,
                pad_value=False,
            )
        return UMATokenItem(
            token_data=new_token_data,
            pred_mask=new_token_pred_mask,
            valid_mask=new_token_valid_mask,
        )

    @staticmethod
    def create(
        token_data : Union[torch.Tensor, Any],
        pred_mask : Optional[torch.Tensor] = None,
        valid_mask : Optional[torch.Tensor] = None,
        sequence_shape : Optional[Tuple[int, ...]] = None,
        empty : bool = False,
        device : Optional[PyTorchDeviceType] = None,
    ) -> "UMATokenItem":
        assert sequence_shape is not None or pred_mask is not None or valid_mask is not None, "Either sequence_shape or pred_mask or valid_mask must be provided"
        
        if device is None and hasattr(token_data, 'device'):
            device = token_data.device

        if sequence_shape is None:
            if pred_mask is not None:
                sequence_shape = pred_mask.shape
            else:
                sequence_shape = valid_mask.shape
        if pred_mask is None:
            pred_mask = torch.zeros(sequence_shape, dtype=torch.bool, device=device)
        if valid_mask is None:
            valid_mask = torch.ones(sequence_shape, dtype=torch.bool, device=device) \
                if not empty else torch.zeros(sequence_shape, dtype=torch.bool, device=device)
        
        assert token_data.shape[:len(sequence_shape)] == sequence_shape, "Token data shape does not match sequence shape"
        assert pred_mask.shape == sequence_shape, "Pred mask shape does not match sequence shape"
        assert valid_mask.shape == sequence_shape, "Valid mask shape does not match sequence shape"
        return UMATokenItem(
            token_data=token_data,
            pred_mask=pred_mask,
            valid_mask=valid_mask,
        )
    
    @staticmethod
    def create_space(
        sequence_shape : Tuple[int, ...],
        data_space : Optional[Space[Any, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]] = None,
        data_shape : Optional[Tuple[int, ...]] = None,
        data_dtype : PyTorchDtypeType = torch.float32,
        device : Optional[PyTorchDeviceType] = None,
    ) -> DictSpace:
        assert data_shape is not None or data_space is not None, "Either data_shape or data_space must be provided"
        return DictSpace(
            PyTorchComputeBackend,
            {
                "token_data": data_space if data_space is not None else BoxSpace(
                    PyTorchComputeBackend,
                    low=-float('inf'),
                    high=float('inf'),
                    shape=sequence_shape + data_shape,
                    dtype=data_dtype,
                    device=device,
                ),
                "pred_mask": BinarySpace(
                    PyTorchComputeBackend,
                    shape=sequence_shape,
                    dtype=torch.bool,
                    device=device,
                ),
                "valid_mask": BinarySpace(
                    PyTorchComputeBackend,
                    shape=sequence_shape,
                    dtype=torch.bool,
                    device=device,
                ),
            },
            device=device
        )
    
    @staticmethod
    def _clip_normalize(
        normalized : Union[torch.Tensor, "UMATokenItem"],
        min_val=-1.0,
        max_val=1.0,
    ) -> Union[torch.Tensor, "UMATokenItem"]:
        if _is_token_item(normalized):
            return dict_replace(
                normalized,
                token_data=torch.clamp(normalized['token_data'], min=min_val, max=max_val),
            )

        return torch.clamp(normalized, min=min_val, max=max_val)

    @staticmethod
    def normalize(
        token_item : "UMATokenItem",
        mean : torch.Tensor,
        std : torch.Tensor,
        clip : Optional[torch.Tensor] = None,
    ) -> "UMATokenItem":
        if isinstance(std, torch.Tensor):
            # Device-side check: no host sync per step.
            torch._assert_async((std > 1e-6).all(), "Standard deviation must be greater than 1e-6 for normalization")
        else:
            assert std > 1e-6, "Standard deviation must be greater than 1e-6 for normalization"
        dim_last = mean.shape[-1]
        normalized_token_data = (token_item['token_data'][..., :dim_last] - mean) / std
        
        if clip is not None:
            normalized_token_data = torch.clamp(normalized_token_data, min=-clip, max=clip)

        return UMATokenItem(
            token_data=torch.concat([
                normalized_token_data,
                token_item['token_data'][..., dim_last:]
            ], dim=-1),
            pred_mask=token_item['pred_mask'],
            valid_mask=token_item['valid_mask'],
        )
    
    @staticmethod
    def unnormalize(
        token_item : "UMATokenItem",
        mean : torch.Tensor,
        std : torch.Tensor,
        clip : Optional[torch.Tensor] = None,
    ) -> "UMATokenItem":
        dim_last = mean.shape[-1]

        normalized_token_data = token_item['token_data'][..., :dim_last]
        if clip is not None:
            normalized_token_data = torch.clamp(normalized_token_data, min=-clip, max=clip)
        unnormalized_token_data = normalized_token_data * std + mean
        return UMATokenItem(
            token_data=torch.concat([
                unnormalized_token_data,
                token_item['token_data'][..., dim_last:]
            ], dim=-1),
            pred_mask=token_item['pred_mask'],
            valid_mask=token_item['valid_mask'],
        )
    
    @staticmethod
    def transform_3d(
        token_item : "UMATokenItem",
        transformation_matrix : torch.Tensor,
        is_action : bool = False,
        action_rotation_relative : bool = False,
        inv : bool = False,
    ) -> "UMATokenItem":
        return dict_replace(
            token_item,
            token_data=transform_3d(
                token_item['token_data'],
                transformation_matrix,
                is_action=is_action,
                action_rotation_relative=action_rotation_relative,
                inv=inv,
            )
        )
    
    @staticmethod
    def apply_recursive(
        value : Mapping[str, Any],
        callable : Callable[["UMATokenItem"], Any],
    ) -> Mapping[str, Any]:
        new_dict = {}
        for key, item in value.items():
            if _is_token_item(item):
                new_dict[key] = callable(item)
            elif isinstance(item, Mapping):
                new_dict[key] = UMATokenItem.apply_recursive(item, callable)
            else:
                new_dict[key] = item
        try:
            return type(value)(new_dict)
        except:
            return new_dict

DataT = TypeVar("DataT")
class UMAContextItem(Generic[DataT], TypedDict):
    token_data : Union[torch.Tensor, Any]
    valid_mask : torch.Tensor

    @staticmethod
    def to_device(
        token_item : "UMAContextItem",
        device : PyTorchDeviceType,
    ) -> "UMAContextItem":
        return UMAContextItem(
            token_data=token_item['token_data'].to(device) if isinstance(token_item['token_data'], torch.Tensor) else token_item['token_data'],
            valid_mask=token_item['valid_mask'].to(device),
        )

    @staticmethod
    def pad(
        token_item : "UMAContextItem",
        target_sequence_shape : Tuple[int, ...],
    ) -> "UMAContextItem":
        current_sequence_shape = token_item["valid_mask"].shape
        assert len(current_sequence_shape) == len(target_sequence_shape), "Current sequence shape and target sequence shape must have the same number of dimensions"
        
        new_token_data = token_item['token_data']
        new_token_valid_mask = token_item['valid_mask']

        for i, (current_size, target_size) in enumerate(zip(current_sequence_shape, target_sequence_shape)):
            if current_size >= target_size:
                continue
            new_token_data = PyTorchComputeBackend.pad_dim(
                new_token_data,
                dim=i,
                target_size=target_size
            )
            new_token_valid_mask = PyTorchComputeBackend.pad_dim(
                new_token_valid_mask,
                dim=i,
                target_size=target_size,
                pad_value=False,
            )
        return UMAContextItem(
            token_data=new_token_data,
            valid_mask=new_token_valid_mask,
        )

    @staticmethod
    def create(
        token_data : Union[torch.Tensor, Any],
        valid_mask : Optional[torch.Tensor] = None,
        sequence_shape : Optional[Tuple[int, ...]] = None,
        empty : bool = False,
        device : Optional[PyTorchDeviceType] = None,
    ) -> "UMAContextItem":
        assert sequence_shape is not None or valid_mask is not None, "Either sequence_shape or valid_mask must be provided"
        
        if device is None and hasattr(token_data, 'device'):
            device = token_data.device

        if sequence_shape is None:
            sequence_shape = valid_mask.shape
        if valid_mask is None:
            valid_mask = torch.ones(sequence_shape, dtype=torch.bool, device=device) \
                if not empty else torch.zeros(sequence_shape, dtype=torch.bool, device=device)
        
        assert token_data.shape[:len(sequence_shape)] == sequence_shape, "Token data shape does not match sequence shape"
        assert valid_mask.shape == sequence_shape, "Valid mask shape does not match sequence shape"
        return UMAContextItem(
            token_data=token_data,
            valid_mask=valid_mask,
        )
    
    @staticmethod
    def create_space(
        sequence_shape : Tuple[int, ...],
        data_space : Optional[Space[Any, PyTorchDeviceType, PyTorchDtypeType, PyTorchRNGType]] = None,
        data_shape : Optional[Tuple[int, ...]] = None,
        data_dtype : PyTorchDtypeType = torch.float32,
        device : Optional[PyTorchDeviceType] = None,
    ) -> DictSpace:
        assert data_shape is not None or data_space is not None, "Either data_shape or data_space must be provided"
        return DictSpace(
            PyTorchComputeBackend,
            {
                "token_data": data_space if data_space is not None else BoxSpace(
                    PyTorchComputeBackend,
                    low=-float('inf'),
                    high=float('inf'),
                    shape=sequence_shape + data_shape,
                    dtype=data_dtype,
                    device=device,
                ),
                "valid_mask": BinarySpace(
                    PyTorchComputeBackend,
                    shape=sequence_shape,
                    dtype=torch.bool,
                    device=device,
                ),
            },
            device=device
        )
    
    @staticmethod
    def transform_3d(
        token_item : "UMAContextItem",
        transformation_matrix : torch.Tensor,
        is_action : bool = False,
        inv : bool = False,
    ) -> "UMAContextItem":
        """
        Transform the 3D points in the token data using the given transformation matrix.
        Assumes the first 3 dimensions of the token data are the 3D points.
        The transformation matrix should be of shape (..., 3, 4) or (..., 4, 4).
        Input:
            token_item: UMAContextItem with token_data of shape (..., D), where D >= 3
            transformation_matrix: torch.Tensor of shape (..., 3, 4) or (..., 4, 4)
        Output:
            UMAContextItem with transformed token_data
        """
        batch_shape = tuple(transformation_matrix.shape[:-2])
        assert token_item["token_data"].shape[:len(batch_shape)] == batch_shape, (
            "token_item['token_data'] leading dims must match transformation_matrix batch dims"
        )
        data_sequence_shape = tuple(token_item["token_data"].shape[len(batch_shape):-1])
        B_flat = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1

        valid_mask = token_item["valid_mask"]
        ones = torch.ones_like(valid_mask, dtype=token_item["token_data"].dtype).unsqueeze(-1)
        point_data_homogeneous = torch.cat([
            token_item["token_data"][..., :3],
            ones,
        ], dim=-1)  # (..., 4)
        point_data_homogeneous_flat = point_data_homogeneous.reshape(B_flat, -1, 4)  # (B_flat, N, 4)

        transformation_matrix_flat = transformation_matrix.reshape(B_flat, *transformation_matrix.shape[-2:])  # (B_flat, 3, 4) or (B_flat, 4, 4)
        if transformation_matrix_flat.shape[-2:] == (3, 4):
            bottom_row = transformation_matrix_flat.new_zeros((B_flat, 1, 4))
            bottom_row[..., 3] = 1
            transformation_matrix_flat = torch.cat([
                transformation_matrix_flat,
                bottom_row,
            ], dim=-2)  # (B_flat, 4, 4)
        elif transformation_matrix_flat.shape[-2:] != (4, 4):
            raise ValueError(
                f"transformation_matrix must have shape (..., 3, 4) or (..., 4, 4), got {tuple(transformation_matrix.shape)}"
            )
        
        if inv:
            transformation_matrix_flat = _rigid_inverse(transformation_matrix_flat)

        transformed_points_flat = torch.matmul(
            point_data_homogeneous_flat,
            transformation_matrix_flat.transpose(-1, -2),
        )  # (B_flat, N, 4)
        transformed_points = transformed_points_flat.reshape(batch_shape + data_sequence_shape + (4,))
        transformed_xyz = transformed_points[..., :3] / transformed_points[..., 3:4]  # (..., 3)

        if is_action:
            rotation_matrix = rotation_6d_to_matrix(token_item["token_data"][..., 3:9]) # (..., L, 3, 3)
            flat_rotation_matrix = rotation_matrix.reshape(B_flat, -1, 3, 3) # (B_flat, L, 3, 3)
            transformed_rotation_matrix = transformation_matrix_flat[:, None, :3, :3] @ flat_rotation_matrix # (B_flat, L, 3, 3)
            transformed_rotation_6d = matrix_to_rotation_6d(
                transformed_rotation_matrix.reshape(batch_shape + (-1, 3, 3))
            ) # (..., L, 6)
            new_token_data = torch.concat([
                transformed_xyz,
                transformed_rotation_6d,
                token_item["token_data"][..., 9:],
            ], dim=-1)
        else:
            new_token_data = torch.concat([
                transformed_xyz,
                token_item["token_data"][..., 3:],
            ], dim=-1)

        new_token_data = torch.where(
            torch.logical_or(
                token_item['valid_mask'].reshape(*token_item['valid_mask'].shape, *[1] * (new_token_data.ndim - token_item['valid_mask'].ndim)),
                ~torch.isnan(new_token_data)
            ),
            new_token_data,
            0
        )
        return UMAContextItem(
            token_data=new_token_data,
            valid_mask=valid_mask,
        )
    
    @staticmethod
    def apply_recursive(
        value : Mapping[str, Any],
        callable : Callable[["UMAContextItem"], Any],
    ) -> Mapping[str, Any]:
        new_dict = {}
        for key, item in value.items():
            if _is_context_item(item):
                new_dict[key] = callable(item)
            elif isinstance(item, Mapping):
                new_dict[key] = UMAContextItem.apply_recursive(item, callable)
            else:
                new_dict[key] = item
        try:
            return type(value)(new_dict)
        except:
            return new_dict

class UMAFlowSpecContextItem(TypedDict):
    """
    Flow specification for the task, expressed in camera frame.
    Shape (B, T_fs_max, M_fs_max, flow_spec_dim) for the data, shape (B, T_fs_max, M_fs_max) for valid_mask and pred_mask
    """
    flow_spec: PyTorchArrayType
    
    """
    Initial frame image (normalized to [0, 1])
    Shape (B, 3, H, W) for data
    """
    initial_image : PyTorchArrayType

    """
    Initial frame depth (Optional)
    Shape (B, H, W) for data (0 or nan is invalid depth)
    """
    initial_depth : PyTorchArrayType

    """
    Initial frame camera intrinsics (Optional)
    Shape (B, 3, 3) for data
    """
    initial_intrinsics : PyTorchArrayType

    """
    Initial frame camera extrinsics (world-to-camera) (Optional)
    Shape (B, 3, 4) for data
    """
    initial_extrinsics : PyTorchArrayType

    """
    Valid mask for the flow specification
    Shape (B, T_fs_max, M_fs_max)
    """
    valid_mask : torch.Tensor
    
    @staticmethod
    def create(
        flow_spec : PyTorchArrayType,
        initial_image : PyTorchArrayType,
        initial_depth : PyTorchArrayType,
        initial_intrinsics : PyTorchArrayType,
        initial_extrinsics : PyTorchArrayType,
        valid_mask : torch.Tensor,
    ) -> "UMAFlowSpecContextItem":
        return UMAFlowSpecContextItem(
            flow_spec=flow_spec,
            initial_image=initial_image,
            initial_depth=initial_depth,
            initial_intrinsics=initial_intrinsics,
            initial_extrinsics=initial_extrinsics,
            valid_mask=valid_mask,
        )
    
    @staticmethod
    def create_empty(
        batch_shape : Tuple[int, ...],
        flow_spec_T_fs_max : int,
        flow_spec_M_max : int,
        image_height : int,
        image_width : int,
        flow_spec_dim : int = DEFAULT_FLOW_SPEC_DIM,
        device : Optional[PyTorchDeviceType] = None,
    ) -> "UMAFlowSpecContextItem":
        return UMAFlowSpecContextItem(
            flow_spec=torch.zeros(
                batch_shape + (flow_spec_T_fs_max, flow_spec_M_max, flow_spec_dim),
                dtype=torch.float32,
                device=device
            ),
            initial_image=torch.zeros(
                batch_shape + (3, image_height, image_width),
                dtype=torch.float32,
                device=device
            ),
            initial_depth=torch.zeros(
                batch_shape + (image_height, image_width),
                dtype=torch.float32,
                device=device
            ),
            initial_intrinsics=torch.zeros(
                batch_shape + (3, 3),
                dtype=torch.float32,
                device=device
            ),
            initial_extrinsics=torch.zeros(
                batch_shape + (3, 4),
                dtype=torch.float32,
                device=device
            ),
            valid_mask=torch.zeros(
                batch_shape + (flow_spec_T_fs_max, flow_spec_M_max),
                dtype=torch.bool,
                device=device
            ),
        )

    @staticmethod
    def create_space(
        batch_shape : Tuple[int, ...],
        flow_spec_T_fs_max : int,
        flow_spec_M_max : int,
        image_height : int,
        image_width : int,
        flow_spec_dim : int = DEFAULT_FLOW_SPEC_DIM,
        device : Optional[PyTorchDeviceType] = None,
    ) -> DictSpace:
        return DictSpace(
            PyTorchComputeBackend,
            {
                "flow_spec": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float('inf'),
                    high=float('inf'),
                    shape=batch_shape + (flow_spec_T_fs_max, flow_spec_M_max, flow_spec_dim),
                    dtype=torch.float32,
                    device=device,
                ),
                "initial_image": BoxSpace(
                    PyTorchComputeBackend,
                    low=0.0,
                    high=1.0,
                    shape=batch_shape + (3, image_height, image_width),
                    dtype=torch.float32,
                    device=device,
                ),
                "initial_depth": BoxSpace(
                    PyTorchComputeBackend,
                    low=0.0,
                    high=torch.inf,
                    shape=batch_shape + (image_height, image_width),
                    dtype=torch.float32,
                    device=device,
                ),
                "initial_intrinsics": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float('inf'),
                    high=float('inf'),
                    shape=batch_shape + (3, 3),
                    dtype=torch.float32,
                    device=device,
                ),
                "initial_extrinsics": BoxSpace(
                    PyTorchComputeBackend,
                    low=-float('inf'),
                    high=float('inf'),
                    shape=batch_shape + (3, 4),
                    dtype=torch.float32,
                    device=device,
                ),
                "valid_mask": BinarySpace(
                    PyTorchComputeBackend,
                    shape=batch_shape + (flow_spec_T_fs_max, flow_spec_M_max),
                    dtype=torch.bool,
                    device=device,
                ),
            },
            device=device
        )
    
    @staticmethod
    @torch.no_grad()
    def resize_image(
        context : "UMAFlowSpecContextItem",
        target_height : int,
        target_width : int,
    ) -> "UMAFlowSpecContextItem":
        original_height, original_width = context['initial_image'].shape[-2:]

        if (original_height, original_width) == (target_height, target_width):
            return context

        resized_image = F.interpolate(
            context['initial_image'],
            size=(target_height, target_width),
            mode='bilinear',
            antialias=True,
        )
        resized_depth = F.interpolate(
            context['initial_depth'].unsqueeze(1), # (B, 1, H, W)
            size=(target_height, target_width),
            mode='nearest',
        ).squeeze(1) # (B, H, W)
        resized_intrinsics = resize_intrinsics(
            context['initial_intrinsics'],
            original_height=original_height,
            original_width=original_width,
            target_height=target_height,
            target_width=target_width,
        )
        return dict_replace(
            context,
            initial_image=resized_image,
            initial_depth=resized_depth,
            initial_intrinsics=resized_intrinsics,
        )

    @staticmethod
    @torch.no_grad()
    def perform_augmentation(
        context : "UMAFlowSpecContextItem",
    ) -> "UMAFlowSpecContextItem":
        if not hasattr(UMAFlowSpecContextItem, "_augmentation_transform"):
            transform_list = []
            transform_list.append(v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08
                    )
                ],
                p=0.5
            ))
            transform_list.append(v2.RandomGrayscale(p=0.1))
            transforms = v2.Compose(transform_list)
            UMAFlowSpecContextItem._augmentation_transform = transforms
        else:
            transforms : v2.Compose = UMAFlowSpecContextItem._augmentation_transform

        depth_transforms = functools.partial(
            depth_jitter,
            std=0.01,
        )

        # RandomGrayscale returns a channel-expanded view (stride 0), which
        # pin_memory() rejects ("more than one element ... refers to a single
        # memory location"): materialize it.
        return dict_replace(
            context,
            initial_image=transforms(context['initial_image']).contiguous(),
            initial_depth=depth_transforms(context['initial_depth']),
        )

    @staticmethod
    def to_device(
        flow_spec_context_item : "UMAFlowSpecContextItem",
        device : PyTorchDeviceType,
    ) -> "UMAFlowSpecContextItem":
        return UMAFlowSpecContextItem(
            flow_spec=flow_spec_context_item['flow_spec'].to(device),
            initial_image=flow_spec_context_item['initial_image'].to(device),
            initial_depth=flow_spec_context_item['initial_depth'].to(device),
            initial_intrinsics=flow_spec_context_item['initial_intrinsics'].to(device),
            initial_extrinsics=flow_spec_context_item['initial_extrinsics'].to(device),
            valid_mask=flow_spec_context_item['valid_mask'].to(device),
        )

class UMAImageContext(TypedDict):
    """
    Defines the context input for UMA model.
    """

    """
    Image Data in the current timestep (normalized to [0, 1])
    Shape (B, 3, H, W) for data, shape (B,) for valid_mask and pred_mask
    """
    image : UMAContextItem[PyTorchArrayType]

    """
    Gripper Image in the current timestep (normalized to [0, 1])
    Shape (B, 3, H, W) for data, shape (B,) for valid_mask and pred_mask
    """
    gripper_image : UMAContextItem[PyTorchArrayType]

    """
    Depth Data in the current timestep
    Shape (B, H, W) for data (0 or nan is invalid depth), shape (B,) for valid_mask and pred_mask
    """
    depth : UMAContextItem[PyTorchArrayType]

    """
    Camera intrinsics for the current timestep
    Shape (B, 3, 3) for data, shape (B,) for valid_mask and pred_mask
    """
    intrinsics : UMAContextItem[PyTorchArrayType]

    """
    Camera extrinsics (world-to-camera).
    Shape (B, 3, 4) for data, shape (B,) for valid_mask and pred_mask
    """
    extrinsics : UMAContextItem[PyTorchArrayType]

    """
    language instructions stored in ndarray format
    shape (B, ), each element is a string with length <= max_language_length
    """
    language_instructions : UMAContextItem[Union[str, np.ndarray]]

    """
    Keypoint flow specification for the task, expressed in the initial camera frame (the model's frame).
    Shape (B, T_fs_max, M_fs_max, flow_spec_dim) for the data, shape (B, T_fs_max, M_fs_max) for valid_mask
    """
    flow_spec_context : UMAFlowSpecContextItem

    @staticmethod
    def create(
        image : UMAContextItem[PyTorchArrayType],
        depth : UMAContextItem[PyTorchArrayType],
        intrinsics : UMAContextItem[PyTorchArrayType],
        extrinsics : Optional[UMAContextItem[PyTorchArrayType]] = None,
        gripper_image : Optional[UMAContextItem[PyTorchArrayType]] = None,
        language_instructions : Optional[UMAContextItem[Union[str, np.ndarray]]] = None,
        flow_spec_context : Optional[UMAFlowSpecContextItem] = None,
        flow_spec_T_fs_max : Optional[int] = None,
        flow_spec_M_max : Optional[int] = None,
        flow_spec_dim : int = DEFAULT_FLOW_SPEC_DIM,
    ) -> "UMAImageContext":
        device = image['token_data'].device
        batch_shape = image['token_data'].shape[:-3] # Exclude (C, H, W)
        H, W = image['token_data'].shape[-2:]

        if extrinsics is None:
            extrinsics = UMAContextItem.create(
                token_data=torch.zeros(
                    batch_shape + (3, 4),
                    dtype=torch.float32,
                    device=device
                ),
                sequence_shape=batch_shape,
                empty=True
            )

        if gripper_image is None:
            gripper_image = UMAContextItem.create(
                token_data=torch.zeros(
                    batch_shape + (3, H, W),
                    dtype=torch.float32,
                    device=device
                ),
                sequence_shape=batch_shape,
                empty=True
            )
        
        if language_instructions is None:
            language_instructions = UMAContextItem.create(
                token_data=np.array([''] * int(np.prod(batch_shape)), dtype=object).reshape(batch_shape),
                sequence_shape=batch_shape,
                empty=True,
                device=device,
            )

        if flow_spec_context is None:
            flow_spec_context = UMAFlowSpecContextItem.create_empty(
                batch_shape=batch_shape,
                flow_spec_T_fs_max=flow_spec_T_fs_max if flow_spec_T_fs_max is not None else 1,
                flow_spec_M_max=flow_spec_M_max if flow_spec_M_max is not None else 1,
                image_height=H,
                image_width=W,
                flow_spec_dim=flow_spec_dim,
                device=device,
            )
        return UMAImageContext(
            image=image,
            gripper_image=gripper_image,
            depth=depth,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            language_instructions=language_instructions,
            flow_spec_context=flow_spec_context,
        )
    
    @staticmethod
    def create_space(
        batch_shape : Tuple[int, ...],
        image_height : int,
        image_width : int,
        flow_spec_T_fs_max : int,
        flow_spec_M_max : int,
        max_language_length : int,
        flow_spec_dim : int = DEFAULT_FLOW_SPEC_DIM,
        device : Optional[PyTorchDeviceType] = None,
    ) -> DictSpace:
        single_text_space = TextSpace(
            PyTorchComputeBackend,
            max_length=max_language_length,
        )
        if batch_shape == ():
            batched_text_space = single_text_space
        else:
            batched_text_space = BatchedSpace(
                single_text_space,
                batch_shape
            )

        return DictSpace(
            PyTorchComputeBackend,
            {
                "image": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_shape=(3, image_height, image_width),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "depth": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_shape=(image_height, image_width),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "intrinsics": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_shape=(3, 3),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "extrinsics": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_shape=(3, 4),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "gripper_image": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_shape=(3, image_height, image_width),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "language_instructions": UMAContextItem.create_space(
                    sequence_shape=batch_shape,
                    data_space=batched_text_space,
                    device=device,
                ),
                "flow_spec_context": UMAFlowSpecContextItem.create_space(
                    batch_shape=batch_shape,
                    flow_spec_T_fs_max=flow_spec_T_fs_max,
                    flow_spec_M_max=flow_spec_M_max,
                    image_height=image_height,
                    image_width=image_width,
                    flow_spec_dim=flow_spec_dim,
                    device=device,
                ),
            },
            device=device
        )

    @staticmethod
    def to_device(
        image_context : "UMAImageContext",
        device : PyTorchDeviceType,
    ) -> "UMAImageContext":
        new_dict = {}
        for key, value in image_context.items():
            if key == 'flow_spec_context':
                new_dict[key] = UMAFlowSpecContextItem.to_device(value, device)
            else:
                new_dict[key] = UMAContextItem.to_device(value, device)
        return UMAImageContext(new_dict)

    @staticmethod
    @torch.no_grad
    def obtain_full_rgb_pointcloud(
        image_context : "UMAImageContext",
        pointcloud_workspace : Optional[BoxSpace] = None,
        crop_robot_mask : bool = False,
    ) -> PyTorchArrayType:
        """Coloured camera-frame pointcloud of every pixel, (B, N, 3 + 3 + 1): xyz, rgb, valid.

        ``pointcloud_workspace`` (robot base frame) crops the points of the batch entries with valid extrinsics.
        """
        batch_shape = image_context['image']['valid_mask'].shape
        pcd_points = depth_image_to_world(
            depth_image=image_context['depth']['token_data'],
            intrinsic_matrix=image_context['intrinsics']['token_data'],
            extrinsic_matrix=None,
        ).reshape(*batch_shape, -1, 4)  # (B, N, 4)
    
        pcd_valids = pcd_points[..., -1] != 0  # (B, N)
        pcd_points = pcd_points[..., :3]  # (B, N, 3)

        # Apply workspace cropping only for batches with valid extrinsics
        if pointcloud_workspace is not None:
            # Get per-batch extrinsics validity mask (B,)
            extrinsics_valid = image_context['extrinsics']['valid_mask']  # (B,)

            # Only compute workspace filtering for batches with valid extrinsics
            if extrinsics_valid.any():
                pcd_points_world = depth_image_to_world(
                    depth_image=image_context['depth']['token_data'],
                    intrinsic_matrix=image_context['intrinsics']['token_data'],
                    extrinsic_matrix=image_context['extrinsics']['token_data']
                ).reshape(*batch_shape, -1, 4)

                workspace_low = pointcloud_workspace.low.reshape(*([1] * len(batch_shape)), 1, 3).to(pcd_points_world.device) # (B, 1, 3)
                workspace_high = pointcloud_workspace.high.reshape(*([1] * len(batch_shape)), 1, 3).to(pcd_points_world.device) # (B, 1, 3)

                # Compute workspace mask
                workspace_mask = torch.logical_and(
                    pcd_points_world[..., :3] >= workspace_low,
                    pcd_points_world[..., :3] <= workspace_high,
                ).all(dim=-1)  # (B, N)

                # Apply workspace mask only to batches with valid extrinsics
                pcd_valids = torch.where(
                    extrinsics_valid.reshape(*batch_shape, 1),  # (B, 1)
                    pcd_valids & workspace_mask,
                    pcd_valids
                )

        pcd_valids &= image_context['depth']['token_data'].reshape(*batch_shape, -1) > 0.1  # (B, N)
        pcd_valids &= image_context['depth']['token_data'].reshape(*batch_shape, -1) < 1.8  # (B, N)
        if crop_robot_mask:
            pcd_valids &= torch.logical_not(image_context['robot_mask']['token_data'].reshape(*batch_shape, -1))  # (B, N)

        pcd_rgb = image_context['image']['token_data'].movedim(-3, -1).reshape(*batch_shape, -1, 3)  # (B, N, 3)
        full_pointcloud = torch.concat([
            pcd_points,
            pcd_rgb,
            pcd_valids.unsqueeze(-1)
        ], dim=-1)  # (B, N, 3 + 3 + 1)
        return full_pointcloud

    @staticmethod
    def resize_image(
        image_context : "UMAImageContext",
        target_height : int,
        target_width : int,
    ) -> "UMAImageContext":
        original_height, original_width = image_context['image']['token_data'].shape[-2:]
        if original_height != target_height or original_width != target_width:
            image_context = dict_replace(
                image_context,
                image=dict_replace(
                    image_context['image'],
                    token_data=F.interpolate(
                        image_context['image']['token_data'],  # (B, 3, H, W)
                        size=(target_height, target_width),
                        mode='bilinear',
                        antialias=True,
                    ),  # (B, H, W, 3)
                ),
                depth=dict_replace(
                    image_context['depth'],
                    token_data=F.interpolate(
                        image_context['depth']['token_data'].unsqueeze(1),  # (B, 1, H, W)
                        size=(target_height, target_width),
                        mode='nearest',
                    ).squeeze(1),  # (B, H, W)
                ),
                intrinsics=dict_replace(
                    image_context['intrinsics'],
                    token_data=resize_intrinsics(
                        image_context['intrinsics']['token_data'],
                        original_height=original_height,
                        original_width=original_width,
                        target_height=target_height,
                        target_width=target_width,
                    )
                ),
                flow_spec_context=UMAFlowSpecContextItem.resize_image(
                    image_context['flow_spec_context'],
                    target_height=target_height,
                    target_width=target_width,
                ),
            )

        # Resize gripper image and depth if valid
        gripper_orig_h, gripper_orig_w = image_context['gripper_image']['token_data'].shape[-2:]
        if gripper_orig_h != target_height or gripper_orig_w != target_width:
            image_context = dict_replace(
                image_context,
                gripper_image=dict_replace(
                    image_context['gripper_image'],
                    token_data=F.interpolate(
                        image_context['gripper_image']['token_data'],
                        size=(target_height, target_width),
                        mode='bilinear',
                        antialias=True,
                    ),
                ),
            )
        return image_context
    
    @staticmethod
    @torch.no_grad()
    def perform_augmentation(
        image_context : "UMAImageContext",
    ) -> "UMAImageContext":
        if not hasattr(UMAImageContext, "_augmentation_transform"):
            transform_list = []
            transform_list.append(v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.32, contrast=0.32, saturation=0.32, hue=0.08
                    )
                ],
                p=0.5
            ))
            transform_list.append(v2.RandomGrayscale(p=0.1))
            transforms = v2.Compose(transform_list)
            UMAImageContext._augmentation_transform = transforms
        else:
            transforms : v2.Compose = UMAImageContext._augmentation_transform

        depth_transforms = functools.partial(
            depth_jitter,
            std=0.01,
        )

        # .contiguous(): see UMAFlowSpecContextItem.perform_augmentation.
        return dict_replace(
            image_context,
            image=dict_replace(
                image_context['image'],
                token_data=transforms(
                    image_context['image']['token_data']
                ).contiguous()
            ),
            depth=dict_replace(
                image_context['depth'],
                token_data=depth_transforms(
                    image_context['depth']['token_data']
                )
            ),
            gripper_image=dict_replace(
                image_context['gripper_image'],
                token_data=transforms(
                    image_context['gripper_image']['token_data']
                ).contiguous()
            ),
            flow_spec_context=UMAFlowSpecContextItem.perform_augmentation(
                image_context['flow_spec_context']
            ),
        )

class UMAFlowSpecInputContext(TypedDict):
    """
    Defines the flow specification context input for UMA model.
    """
    """
    Flow specification for the task, in the same frame as initial_pointcloud.
    Shape (B, T_fs_max, M_fs_max, 3) for the data, shape (B, T_fs_max, M_fs_max) for valid_mask and pred_mask
    """
    flow_spec: PyTorchArrayType
    
    """
    Initial frame pointcloud
    Shape (B, N, 4) for data (x, y, z, valid)
    """
    initial_pointcloud : PyTorchArrayType

    """
    Low-resolution initial-frame patch features from the frozen visual encoder.
    Shape (B, H_p, W_p, C)
    """
    initial_patch_features : UMAContextItem[PyTorchArrayType]

    """
    Whether flow_spec and initial_pointcloud are in the (initial) camera frame. True as built by
    from_flow_spec_context; False after transform_3d (SE(3) augmentation), whose initial_extrinsics then map the
    transformed frame back to the camera for projecting points into the image.
    """
    pointcloud_in_cam_frame : bool

    """
    Original image size corresponding to the patch grid.
    """
    initial_image_size : Tuple[int, int]

    """
    Initial frame camera intrinsics.
    Shape (B, 3, 3) for data.
    """
    initial_intrinsics : PyTorchArrayType

    """
    Initial frame camera extrinsics (world-to-camera).
    Shape (B, 3, 4) for data.
    """
    initial_extrinsics : PyTorchArrayType

    """
    Valid mask for the flow specification
    Shape (B, T_fs_max, M_fs_max)
    """
    valid_mask : torch.Tensor
    
    @staticmethod
    @torch.no_grad()  # not inference_mode: training backpropagates through layers fed by these outputs
    def from_flow_spec_context(
        flow_spec_context : UMAFlowSpecContextItem,
        n_points : int,
        encoder_type : EncoderType,
        rng : torch.Generator,
    ) -> "UMAFlowSpecInputContext":
        from uma.tokenizers.language import tokenize_patch_features

        image_patch_features = tokenize_patch_features(
            flow_spec_context['initial_image'], 
            device=flow_spec_context['initial_image'].device,
            upscale=False, 
            encoder_type=encoder_type
        ) # (B, H_p, W_p, C)

        batch_shape = flow_spec_context['initial_image'].shape[:-3]
        device = flow_spec_context['initial_image'].device
        image_size = tuple(flow_spec_context['initial_image'].shape[-2:])

        # Camera-frame pointcloud of the initial frame, like the flow spec.
        pointcloud = depth_image_to_world(
            flow_spec_context['initial_depth'],
            intrinsic_matrix=flow_spec_context['initial_intrinsics'],
            extrinsic_matrix=None,
        )  # (B, H, W, 4)
        pointcloud = pointcloud.flatten(1, 2)  # (B, N, 4)
        rng, pointcloud_indices, pointcloud_sampled_valid = random_point_sampling(
            pointcloud,
            num_samples=n_points,
            rng=rng,
            points_valid=pointcloud[..., -1] != 0
        ) # (B, n_points)
        initial_pointcloud = torch.gather(
            pointcloud,
            dim=1,
            index=pointcloud_indices.unsqueeze(-1).expand(-1, -1, 4)
        ) # (B, n_points, 4)
        initial_pointcloud = initial_pointcloud.clone()
        initial_pointcloud[..., -1] = pointcloud_sampled_valid.to(initial_pointcloud.dtype)

        return UMAFlowSpecInputContext(
            flow_spec=flow_spec_context['flow_spec'],
            initial_pointcloud=initial_pointcloud,
            initial_patch_features=UMAContextItem.create(
                token_data=image_patch_features,
                valid_mask=torch.ones(batch_shape, dtype=torch.bool, device=device),
            ),
            pointcloud_in_cam_frame=True,
            initial_image_size=image_size,
            initial_intrinsics=flow_spec_context['initial_intrinsics'].to(device),
            initial_extrinsics=flow_spec_context['initial_extrinsics'].to(device),
            valid_mask=flow_spec_context['valid_mask'],
        )

    @staticmethod
    def transform_3d(
        token_item : "UMAFlowSpecInputContext",
        transformation_matrix : torch.Tensor,
        inv : bool = False,
    ) -> "UMAFlowSpecInputContext":
        """
        Transform the flow specification and associated pointcloud/extrinsics.
        After transformation, pointcloud_in_cam_frame is set to False because the
        data is now expressed in a transformed frame.  When the original data was
        already in camera frame, the projection extrinsics after transformation must
        derive from the identity camera-frame extrinsics, not the stored
        world-to-camera initial_extrinsics.
        """
        if token_item['pointcloud_in_cam_frame']:
            # Original spec / pointcloud live in the camera frame.  After applying
            # the frame change the correct projection extrinsics are obtained by
            # transforming identity (camera-frame) extrinsics, not the original
            # world-to-camera matrix.
            batch_shape = token_item['initial_extrinsics'].shape[:-2]
            source_extrinsics = torch.eye(
                4,
                dtype=token_item['initial_extrinsics'].dtype,
                device=token_item['initial_extrinsics'].device,
            )[:3]  # (3, 4)
            source_extrinsics = source_extrinsics.view(
                (1,) * len(batch_shape) + (3, 4)
            ).expand(batch_shape + (3, 4))
        else:
            source_extrinsics = token_item['initial_extrinsics']
        return dict_replace(
            token_item,
            initial_pointcloud=transform_3d(
                token_item['initial_pointcloud'],
                transformation_matrix,
                inv=inv,
                is_action=False,
            ),
            initial_extrinsics=transform_extrinsics(
                source_extrinsics,
                transformation_matrix,
                inv=inv,
            ),
            flow_spec=transform_3d(
                token_item['flow_spec'],
                transformation_matrix,
                inv=inv
            ),
            pointcloud_in_cam_frame=False,
        )

    @staticmethod
    def to_device(
        flow_spec_input_context : "UMAFlowSpecInputContext",
        device : PyTorchDeviceType,
    ) -> "UMAFlowSpecInputContext":
        new_dict = {}
        for key, value in flow_spec_input_context.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            elif _is_context_item(value):
                new_dict[key] = UMAContextItem.to_device(value, device)
            else:
                new_dict[key] = value
        return UMAFlowSpecInputContext(new_dict)

class UMAInputContext(TypedDict):
    """
    Defines the context input for UMA model.
    """

    """
    Pointcloud in the current timestep obtained from the depth image, in the camera frame
    Shape (B, N, 4) for data (x, y, z, valid)
    """
    pointcloud : torch.Tensor

    """
    Low-resolution patch features from the current RGB image.
    Shape (B, H_p, W_p, C) for data.
    """
    pointcloud_patch_features : UMAContextItem[PyTorchArrayType]

    """
    Original image size corresponding to the patch grid.
    """
    pointcloud_image_size : Tuple[int, int]

    """
    Intrinsics used for projecting pointcloud queries back into the image.
    """
    pointcloud_intrinsics : PyTorchArrayType

    """
    Gripper Patch Features in the current timestep
    Shape (B, H', W', C) for data, shape (B,) for valid_mask
    """
    gripper_features : UMAContextItem[PyTorchArrayType]

    """
    Tokenized language instructions
    shape (B, max_language_length, C) for data, shape (B, max_language_length) for valid_mask
    """
    tokenized_language_instructions : UMAContextItem[PyTorchArrayType]

    """
    Keypoint flow specification for the task
    """
    flow_spec_context : UMAFlowSpecInputContext
    
    @staticmethod
    @torch.no_grad()  # not inference_mode: training backpropagates through layers fed by these outputs
    def from_image_context(
        image_context : UMAImageContext,
        n_points : int,
        max_language_length : int,
        encoder_type : EncoderType,
        rng : torch.Generator,
    ) -> "UMAInputContext":
        from uma.tokenizers.language import tokenize_languages, tokenize_patch_features

        image_patch_features = tokenize_patch_features(
            image_context['image']['token_data'], 
            device=image_context['image']['token_data'].device,
            upscale=False, 
            encoder_type=encoder_type
        ) # (B, H_p, W_p, C)
        pointcloud = depth_image_to_world(
            depth_image=image_context['depth']['token_data'],
            intrinsic_matrix=image_context['intrinsics']['token_data'],
            extrinsic_matrix=None,
        )  # (B, H, W, 4), camera frame
        pointcloud = pointcloud.reshape(image_context['image']['token_data'].shape[0], -1, 4)  # (B, N, 4)
        rng, pointcloud_indices, pointcloud_sampled_valid = random_point_sampling(
            pointcloud,
            num_samples=n_points,
            rng=rng,
            points_valid=pointcloud[..., -1] != 0
        ) # (B, n_points)
        pointcloud = torch.gather(
            pointcloud,
            dim=1,
            index=pointcloud_indices.unsqueeze(-1).expand(-1, -1, 4)
        ) # (B, n_points, 4)
        pointcloud = pointcloud.clone()
        pointcloud[..., -1] = pointcloud_sampled_valid.to(pointcloud.dtype)

        pointcloud_image_valid = (
            image_context['image']['valid_mask']
            & image_context['depth']['valid_mask']
            & image_context['intrinsics']['valid_mask']
        )

        gripper_features = UMAContextItem.create(
            token_data=tokenize_patch_features(
                image_context['gripper_image']['token_data'], 
                device=image_context['gripper_image']['token_data'].device,
                upscale=False, 
                encoder_type=encoder_type
            ), # (B, H', W', C)
            valid_mask=image_context['gripper_image']['valid_mask'],
        )
        tokenized_language_instructions = tokenize_languages(
            image_context['language_instructions']['token_data'].tolist(),
            max_language_length=max_language_length,
            device=image_context['language_instructions']['valid_mask'].device,
            encoder_type=encoder_type,
        )
        tokenized_language_instructions = dict_replace(
            tokenized_language_instructions,
            valid_mask=(
                tokenized_language_instructions['valid_mask'] &
                image_context['language_instructions']['valid_mask'][:, None].bool()
            ),
        )

        flow_spec_context = UMAFlowSpecInputContext.from_flow_spec_context(
            image_context['flow_spec_context'],
            n_points=n_points,
            encoder_type=encoder_type,
            rng=rng,
        )
        return UMAInputContext(
            pointcloud=pointcloud,
            pointcloud_patch_features=UMAContextItem.create(
                token_data=image_patch_features,
                valid_mask=pointcloud_image_valid,
            ),
            pointcloud_image_size=tuple(image_context['image']['token_data'].shape[-2:]),
            pointcloud_intrinsics=image_context['intrinsics']['token_data'],
            gripper_features=gripper_features,
            tokenized_language_instructions=tokenized_language_instructions,
            flow_spec_context=flow_spec_context,
        )

    @staticmethod
    def to_device(
        input_context : "UMAInputContext",
        device : PyTorchDeviceType,
    ) -> "UMAInputContext":
        new_dict = {}
        for key, value in input_context.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            elif _is_context_item(value):
                new_dict[key] = UMAContextItem.to_device(value, device)
            elif key == 'flow_spec_context':
                new_dict[key] = UMAFlowSpecInputContext.to_device(value, device)
            else:
                new_dict[key] = value
        return UMAInputContext(new_dict)

class UMAMaskedTokens(TypedDict):
    """
    Action Data
    Shape (B, obs_horizon + pred_horizon + far_future_horizon, action_dim)
    """
    action: UMATokenItem[PyTorchArrayType]

    """
    Flow Data
    Shape (B, obs_horizon + pred_horizon + far_future_horizon, M_flow, flow_dim)
    """
    flow: UMATokenItem[PyTorchArrayType]

    @staticmethod
    def create(
        action : Optional[UMATokenItem[PyTorchArrayType]],
        flow : Optional[UMATokenItem[PyTorchArrayType]],
        flow_M_max : Optional[int] = None,
        flow_dim : int = DEFAULT_FLOW_DIM,
        action_dim : int = DEFAULT_ACTION_DIM,
    ) -> "UMAMaskedTokens":
        assert action is not None or flow is not None, "Either action or flow must be provided"
        batch_shape = action['token_data'].shape[:-2] if action is not None else flow['token_data'].shape[:-3]
        total_horizon = action['token_data'].shape[-2] if action is not None else flow['token_data'].shape[-3]
        device = action['token_data'].device if action is not None else flow['token_data'].device
        
        if action is None:
            action = UMATokenItem.create(
                token_data=torch.zeros(
                    batch_shape + (total_horizon, action_dim),
                    dtype=torch.float32,
                    device=device
                ),
                sequence_shape=batch_shape + (total_horizon, ),
                empty=True
            )
        if flow is None:
            assert flow_M_max is not None, "If flow is not provided, flow_M_max must be provided"
            flow = UMATokenItem.create(
                token_data=torch.zeros(
                    batch_shape + (total_horizon, flow_M_max, flow_dim),
                    dtype=torch.float32,
                    device=device
                ),
                sequence_shape=batch_shape + (total_horizon, flow_M_max),
                empty=True
            )
        return UMAMaskedTokens(
            action=action,
            flow=flow,
        )
    
    @staticmethod
    def create_space(
        batch_shape : Tuple[int, ...],
        total_horizon : int,
        flow_M_max : int,
        flow_dim : int = DEFAULT_FLOW_DIM,
        action_dim : int = DEFAULT_ACTION_DIM,
        device : Optional[PyTorchDeviceType] = None,
    ) -> DictSpace:
        return DictSpace(
            PyTorchComputeBackend,
            {
                "action": UMATokenItem.create_space(
                    sequence_shape=batch_shape + (total_horizon, ),
                    data_shape=(action_dim, ),
                    data_dtype=torch.float32,
                    device=device,
                ),
                "flow": UMATokenItem.create_space(
                    sequence_shape=batch_shape + (total_horizon, flow_M_max),
                    data_shape=(flow_dim, ),
                    data_dtype=torch.float32,
                    device=device,
                ),
            },
            device=device
        )
    
    @staticmethod
    def to_device(
        masked_tokens : "UMAMaskedTokens",
        device : PyTorchDeviceType,
    ) -> "UMAMaskedTokens":
        new_dict = {}
        for key, value in masked_tokens.items():
            new_dict[key] = UMATokenItem.to_device(value, device)
        return UMAMaskedTokens(new_dict)

    @staticmethod
    def transform_3d(
        masked_tokens : "UMAMaskedTokens",
        transformation_matrix : torch.Tensor,
        inv : bool = False,
        transform_flow : bool = True,
        transform_action : bool = True,
        action_rotation_relative : bool = False,
    ) -> "UMAMaskedTokens":
        return UMAMaskedTokens(
            action=UMATokenItem.transform_3d(
                masked_tokens['action'],
                transformation_matrix,
                is_action=True,
                action_rotation_relative=action_rotation_relative,
                inv=inv
            ) if transform_action else masked_tokens['action'],
            flow=UMATokenItem.transform_3d(
                masked_tokens['flow'],
                transformation_matrix,
                inv=inv,
            ) if transform_flow else masked_tokens['flow'],
        )
