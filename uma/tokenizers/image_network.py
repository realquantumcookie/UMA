import torch
from torch import nn
from .language import EncoderType, tokenize_patch_features, upsample_patch_features, get_image_embedding_dim

from xbarray.transformations.pointcloud.pytorch import (
    bilinear_interpolate,
    world_to_pixel_coordinate_and_depth
)

class ImageTokenizer(nn.Module):
    def __init__(
        self,
        encoder_type : EncoderType,
    ) -> None:
        super().__init__()
        self.encoder_type = encoder_type
    
    @torch.compiler.disable
    def forward(
        self,
        image : torch.Tensor,
        image_valid : torch.Tensor,
        unknown_image_embedding : torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tokenize the input image into patch features and their normalized 2D patch coordinates.
        Args:
            image: Tensor of shape (B, 3, H, W) with pixel values in the range [0, 1].
            image_valid: Tensor of shape (B, ) containing the validity mask for each image (1 for valid, 0 for invalid).
            unknown_image_embedding: Tensor of shape (B, C) or (1, C) containing the image embedding from the unknown image encoder, used for invalid images.
        Returns:
            A tuple containing:
                - tokens: Tensor of shape (B, H', W', C + 2): the patch features concatenated with the patch's (x, y) grid coordinates in [0, 1].
                - pure_patch_features: Tensor of shape (B, H', W', C) containing the patch features without coordinates.
                - position_encoding: Tensor of shape (H', W', 2) containing the (x, y) grid coordinates of each patch.
        """
        with torch.no_grad():
            patch_features = tokenize_patch_features(
                image,
                device=image.device,
                upscale=False,
                encoder_type=self.encoder_type,
            ) # (B, H_p, W_p, image_embedding_dim)
            
            B, H_p, W_p = patch_features.shape[:3]
            device = image.device

            # Build normalized 2D grid coordinates [0, 1]
            ys = torch.linspace(0, 1, H_p, device=device)
            xs = torch.linspace(0, 1, W_p, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H_p, W_p, 2)

            # The raw grid coordinates serve as the 2D positional encoding.
            pos_emb_2d = grid_coords

        # Invalid images get the unknown-image embedding at every patch.
        patch_features = torch.where(
            image_valid[:, None, None, None],
            patch_features,
            unknown_image_embedding[:, None, None, :]
        )
        return (
            torch.concat([
                patch_features, 
                pos_emb_2d[None, ...].expand(B, -1, -1, -1)
            ], dim=-1), # (B, H_p, W_p, C + 2)
            patch_features, # (B, H_p, W_p, C)
            pos_emb_2d, # (H_p, W_p, 2)
        )
    
    @staticmethod
    def add_2d_positional_encoding(patch_features : torch.Tensor) -> torch.Tensor:
        """
        Append normalized (x, y) patch grid coordinates in [0, 1] to the patch features.
        Args:
            patch_features: Tensor of shape (B, H', W', C) containing the tokenized patch features for each image.
        Returns:
            Tensor of shape (B, H', W', C + 2) containing the patch features concatenated with their grid coordinates.
        """
        H_p, W_p = patch_features.shape[1:3]
        device = patch_features.device

        # Build normalized 2D grid coordinates [0, 1]
        ys = torch.linspace(0, 1, H_p, device=device)
        xs = torch.linspace(0, 1, W_p, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # (H_p, W_p, 2)

        return torch.concat([
            patch_features, 
            grid_coords[None, ...].expand(patch_features.shape[0], -1, -1, -1)
        ], dim=-1)

class KeypointFeatureExtractor(nn.Module):
    def __init__(
        self,
        depth_threshold : float = 0.015,
    ) -> None:
        super().__init__()
        self.depth_threshold = depth_threshold
    
    @torch.compiler.disable
    def forward(
        self,
        keypoint_locations : torch.Tensor,
        keypoint_valids : torch.Tensor,
        unknown_image_embedding : torch.Tensor,
        feature_image : torch.Tensor,
        depth_image : torch.Tensor,
        intrinsics : torch.Tensor,
        extrinsics : torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract keypoint features from the input RGB-D image and camera parameters.
        Args:
            keypoint_locations: Tensor of shape (B, N, 3) containing the world coorindates of the keypoints.
            keypoint_valids: Tensor of shape (B, N) containing the validity of each keypoint (1 for valid, 0 for invalid).
            unknown_image_embedding: Tensor of shape (B, C) or (1, C) containing the image embedding from the unknown image encoder.
            feature_image: Tensor of shape (B, H, W, C) containing the dense feature for each pixel.
            depth_image: Tensor of shape (B, H, W) containing the depth image.
            intrinsics: Tensor of shape (B, 3, 3) containing the camera intrinsics.
            extrinsics: Optional tensor of shape (B, 4, 4) containing the camera extrinsics. If None, identity is assumed.
        Returns:
            A tuple containing:
                - keypoint_features: Tensor of shape (B, N, C) containing the extracted features for each keypoint.
        """
        with torch.no_grad():
            H, W = depth_image.shape[-2:]

            keypoint_locations_2d, keypoint_depths = world_to_pixel_coordinate_and_depth(
                keypoint_locations,
                intrinsics,
                extrinsics
            ) # (B, N, 2), (B, N)
            
            depth_valids = (depth_image > 0) & (~torch.isnan(depth_image))
            depth_image = torch.where(
                depth_valids,
                depth_image,
                torch.nan
            ) # (B, H, W)
            interpolated_depths = bilinear_interpolate(
                depth_image[..., None], # (B, H, W, 1)
                keypoint_locations_2d,
                index_dtype=torch.long,
            )[..., 0] # (B, N)
            keypoint_valids = keypoint_valids & (torch.abs(interpolated_depths - keypoint_depths) < self.depth_threshold) & (~torch.isnan(interpolated_depths))
            keypoint_valids = keypoint_valids & (keypoint_locations_2d[..., 0] >= 0) & (keypoint_locations_2d[..., 0] < W) & (keypoint_locations_2d[..., 1] >= 0) & (keypoint_locations_2d[..., 1] < H)
            
            interpolated_features = bilinear_interpolate(
                torch.where(
                    depth_valids[..., None],
                    feature_image,
                    torch.nan
                ),
                keypoint_locations_2d,
                index_dtype=torch.long,
            ) # (B, N, C)
            interpolated_features_valid = (interpolated_features != 0).any(dim=-1) # (B, N)
            keypoint_valids = keypoint_valids & interpolated_features_valid

        keypoint_features = torch.where(
            keypoint_valids[..., None],
            interpolated_features,
            unknown_image_embedding[:, None, :],
         ) # (B, N, C)
        return keypoint_features
