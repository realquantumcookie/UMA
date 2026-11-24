from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from xbarray.transformations.pointcloud.pytorch import world_to_pixel_coordinate_and_depth

from uma.network.pointnet import PointNet2Encoder


def project_pointcloud_to_image(
    query_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3-D query points into image pixel coordinates."""
    if query_xyz.shape[-1] != 3:
        raise ValueError(f"Expected query_xyz to end with 3 coordinates, got {tuple(query_xyz.shape)}")
    pixel_coordinates, depth = world_to_pixel_coordinate_and_depth(
        query_xyz,
        intrinsics,
        extrinsics,
    )
    return pixel_coordinates, depth


def sample_patch_features_at_pixels(
    patch_features: torch.Tensor,
    pixel_coordinates: torch.Tensor,
    image_size: Tuple[int, int],
    valid_mask: Optional[torch.Tensor] = None,
    image_valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample low-resolution patch features at pixel coordinates.

    Uses bilinear sampling with align_corners=False and a pixel->grid conversion
    that matches the same convention.
    """
    if patch_features.ndim != 4:
        raise ValueError(f"patch_features must have shape (B, H, W, C), got {tuple(patch_features.shape)}")

    batch_size, num_points = pixel_coordinates.shape[:2]
    feature_dim = patch_features.shape[-1]

    if num_points == 0:
        empty_features = patch_features.new_zeros((batch_size, 0, feature_dim))
        empty_valid = pixel_coordinates.new_zeros((batch_size, 0), dtype=torch.bool)
        return empty_features, empty_valid

    H, W = int(image_size[0]), int(image_size[1])
    safe_pixels = torch.nan_to_num(pixel_coordinates, nan=0.0, posinf=0.0, neginf=0.0)
    grid_x = (safe_pixels[..., 0] + 0.5) * (2.0 / max(float(W), 1.0)) - 1.0
    grid_y = (safe_pixels[..., 1] + 0.5) * (2.0 / max(float(H), 1.0)) - 1.0

    # Clamp grid to the safe interpolation domain of the low-res patch feature map
    # so that valid in-bounds image projections never sample outside the patch grid.
    patch_H, patch_W = patch_features.shape[1], patch_features.shape[2]
    safe_grid_x_min = -1.0 + 1.0 / max(float(patch_W), 1.0)
    safe_grid_x_max = 1.0 - 1.0 / max(float(patch_W), 1.0)
    safe_grid_y_min = -1.0 + 1.0 / max(float(patch_H), 1.0)
    safe_grid_y_max = 1.0 - 1.0 / max(float(patch_H), 1.0)
    grid_x = torch.clamp(grid_x, safe_grid_x_min, safe_grid_x_max)
    grid_y = torch.clamp(grid_y, safe_grid_y_min, safe_grid_y_max)

    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = torch.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(2)  # (B, N, 1, 2)

    sampled_valid = torch.isfinite(pixel_coordinates).all(dim=-1)
    sampled_valid &= (pixel_coordinates[..., 0] >= 0) & (pixel_coordinates[..., 0] <= (W - 1))
    sampled_valid &= (pixel_coordinates[..., 1] >= 0) & (pixel_coordinates[..., 1] <= (H - 1))
    if valid_mask is not None:
        sampled_valid &= valid_mask.bool()
    if image_valid_mask is not None:
        sampled_valid &= image_valid_mask.bool()[:, None]

    if feature_dim == 0:
        sampled_features = patch_features.new_zeros((batch_size, num_points, 0))
        return sampled_features, sampled_valid

    sampled_features = F.grid_sample(
        patch_features.permute(0, 3, 1, 2),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )  # (B, C, N, 1)
    sampled_features = sampled_features.permute(0, 2, 3, 1).squeeze(2)  # (B, N, C)
    sampled_features = torch.nan_to_num(sampled_features, nan=0.0)
    sampled_features = torch.where(sampled_valid[..., None], sampled_features, 0.0)
    return sampled_features, sampled_valid


class PointCloudPatchFusion(nn.Module):
    """Fuse geometry-only PointNet++ features with sampled patch features."""

    def __init__(
        self,
        point_feature_dim: int,
        visual_feature_dim: int,
        token_dim: int,
        projection_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = projection_hidden_dim or token_dim * 2
        self.fusion_projection = nn.Sequential(
            nn.Linear(point_feature_dim + visual_feature_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(
        self,
        pointcloud_encoder: PointNet2Encoder,
        pointcloud: torch.Tensor,
        patch_features: torch.Tensor,
        image_size: Tuple[int, int],
        intrinsics: torch.Tensor,
        extrinsics: Optional[torch.Tensor] = None,
        *,
        image_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``extrinsics`` maps the pointcloud's frame to the camera frame for the projection into the image; ``None``
        when the pointcloud is already in the camera frame."""
        if pointcloud.shape[-1] < 4:
            raise ValueError(f"pointcloud must have shape (B, N, 4), got {tuple(pointcloud.shape)}")

        point_xyz = pointcloud[..., :3]
        point_valids = pointcloud[..., -1] != 0

        point_features, query_xyz, query_valids = pointcloud_encoder(
            point_xyz,
            mask=point_valids,
        )

        pixel_coordinates, depth = project_pointcloud_to_image(
            query_xyz,
            intrinsics,
            extrinsics,
        )
        projected_valids = torch.isfinite(pixel_coordinates).all(dim=-1)
        projected_valids &= torch.isfinite(depth)
        projected_valids &= depth > 0

        sampled_visual_features, sampled_valids = sample_patch_features_at_pixels(
            patch_features,
            pixel_coordinates,
            image_size,
            valid_mask=projected_valids,
            image_valid_mask=image_valid_mask,
        )

        fused_features = torch.cat([
            point_features,
            sampled_visual_features,
            query_xyz,
        ], dim=-1)
        token_features = self.fusion_projection(fused_features)
        token_features = torch.nan_to_num(token_features, nan=0.0)

        final_valids = query_valids & sampled_valids
        token_features = torch.where(final_valids[..., None], token_features, 0.0)
        return token_features, final_valids
