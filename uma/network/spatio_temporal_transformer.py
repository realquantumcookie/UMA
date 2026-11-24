import math
from typing import Type, Optional, Tuple
import torch
import torch.nn as nn

from .attention import TransformerBlock, Attention
from .axial_attention import AxialAttention

class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        dim : int,
        num_layers: int,

        # Attention kwargs
        num_heads : int = 8,
        qkv_bias : bool = True,
        qkv_scale : Optional[float] = None,
        proj_drop : float = 0.0,
        in_dim_src : Optional[int] = None,
        has_context : bool = False,
        in_dim_context : Optional[int] = None,
        
        # Transformer block kwargs
        mlp_ratio : float = 4.0,
        drop_mlp : float = 0.0,
        drop_path : float = 0.0,
        act_layer : Type[nn.Module] = nn.GELU,
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        pre_norm : bool = True,
    ) -> None:
        super().__init__()
        self.temporal_axis = 1
        self.spatial_axis = 2
        if has_context:
            self.context_blocks = nn.ModuleList([
                TransformerBlock(
                    Attention(
                        dim=dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qkv_scale=qkv_scale,
                        proj_drop=proj_drop,
                        in_dim_src=in_dim_src if i_layer == 0 else None,
                        in_dim_context=in_dim_context,
                    ),
                    has_context=True,
                    mlp_ratio=mlp_ratio,
                    drop_mlp=drop_mlp,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pre_norm=pre_norm
                ) for i_layer in range(num_layers)
            ])
        else:
            self.context_blocks = None
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(
                AxialAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_scale=qkv_scale,
                    proj_drop=proj_drop,
                    in_dim_src=in_dim_src if (i_layer == 0 and not has_context) else None,
                    in_dim_context=in_dim_src if (i_layer == 0 and not has_context) else None,
                    attn_axis=self.temporal_axis,
                ),
                has_context=False,
                mlp_ratio=mlp_ratio,
                drop_mlp=drop_mlp,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                pre_norm=pre_norm
            ) for i_layer in range(num_layers)
        ])
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(
                AxialAttention(
                    dim=dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_scale=qkv_scale,
                    proj_drop=proj_drop,
                    attn_axis=self.spatial_axis,
                ),
                has_context=False,
                mlp_ratio=mlp_ratio,
                drop_mlp=drop_mlp,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                pre_norm=pre_norm
            ) for _ in range(num_layers)
        ])
        self.cls_time = nn.Parameter(torch.zeros(in_dim_src or dim))
        nn.init.trunc_normal_(self.cls_time, std=0.02)
        self.cls_keypoint = nn.Parameter(torch.zeros(in_dim_src or dim))
        nn.init.trunc_normal_(self.cls_keypoint, std=0.02)
        self.cls_all = nn.Parameter(torch.zeros(in_dim_src or dim))
        nn.init.trunc_normal_(self.cls_all, std=0.02)

    def forward(
        self, 
        x : torch.Tensor,
        x_mask : torch.Tensor = None,
        context : Optional[torch.Tensor] = None,
        context_mask : Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the spatio-temporal transformer.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, T, K, C)` where:
                - B is the batch size
                - T is the temporal dimension (time steps)
                - K is the spatial dimension (keypoints)
                - C is the channel/embedding dimension
            x_mask (torch.Tensor): Valid mask of shape `(B, T, K)` where True indicates valid tokens and False indicates padding. If provided, it will be used to create appropriate attention masks for both temporal and spatial attention.
            context (torch.Tensor, optional): Context tensor of shape `(B, S, C)` where S is the context sequence length. If the transformer is initialized with `has_context=True`, this context will be attended to in each block.
            context_mask (torch.Tensor, optional): Valid mask for the context tensor of shape `(B, S)`. If provided, it will be used to create attention masks for the context attention in the transformer blocks.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - Global classification token: `(B, D)` - the all-class token at position (0, 0)
                - Spatial tokens: `(B, K, D)` - spatial classification tokens for each keypoint
                - Raw output tokens: `(B, (T+1)*(K+1), D)` - the output tokens before reshaping, which include the classification tokens and the original tokens
                - Raw valid mask: `(B, (T+1)*(K+1))` - the valid mask corresponding to the raw output tokens, which includes the masks for the original tokens and the classification tokens
        """
        B, T, K, C = x.shape
        
        target_cls_shape = (1,) * len(x.shape[:-1]) + (C,)
        def add_cls(x : torch.Tensor, x_mask : torch.Tensor, attn_axis : int, cls_token : torch.Tensor):
            cls_token = cls_token.reshape(target_cls_shape)
            current_target_expand_shape = list(target_cls_shape)
            for axis, axis_size in enumerate(x.shape):
                if axis != attn_axis and axis != attn_axis + len(x.shape):
                    current_target_expand_shape[axis] = axis_size
            cls_token = cls_token.expand(current_target_expand_shape)
            cls_mask = x_mask.any(dim=attn_axis, keepdim=True)
            x = torch.concat([
                cls_token,
                x
            ], dim=attn_axis)
            x_mask = torch.concat([
                cls_mask,
                x_mask
            ], dim=attn_axis if attn_axis >= 0 else attn_axis + len(x.shape))
            return x, x_mask
        
        x, x_mask = add_cls(x, x_mask, self.spatial_axis, self.cls_keypoint)
        x, x_mask = add_cls(x, x_mask, self.temporal_axis, self.cls_time)
        x[:, 0, 0, :] = self.cls_all[None, :]

        x_mask = x_mask.bool()
        T_plus_1, K_plus_1 = x.shape[1], x.shape[2]

        temporal_valid = x_mask.permute(0, 2, 1) # (B, K+1, T+1)
        temporal_attention_mask = (
            temporal_valid[..., :, None] & temporal_valid[..., None, :]
        ) # (B, K+1, T+1, T+1)
        temporal_eye = torch.eye(T_plus_1, dtype=torch.bool, device=x.device)
        temporal_attention_mask = (
            temporal_attention_mask | temporal_eye[None, None, :, :]
        ).reshape(B * K_plus_1, T_plus_1, T_plus_1)

        spatial_valid = x_mask # (B, T+1, K+1)
        spatial_attention_mask = (
            spatial_valid[..., :, None] & spatial_valid[..., None, :]
        ) # (B, T+1, K+1, K+1)
        spatial_eye = torch.eye(K_plus_1, dtype=torch.bool, device=x.device)
        spatial_attention_mask = (
            spatial_attention_mask | spatial_eye[None, None, :, :]
        ).reshape(B * T_plus_1, K_plus_1, K_plus_1)

        if self.context_blocks is not None:
            assert context is not None, "Context tensor must be provided when the transformer is initialized with `has_context=True`."
            if context_mask is not None:
                context_attention_mask = context_mask[:, None, :].expand(
                    B,
                    T_plus_1 * K_plus_1,
                    context.shape[1],
                )
            else:
                context_attention_mask = None

        for i, (temporal_block, spatial_block) in enumerate(zip(self.temporal_blocks, self.spatial_blocks)):
            if self.context_blocks is not None:
                context_block = self.context_blocks[i]
                x_shape = x.shape
                x = context_block.forward(
                    x.flatten(1, 2),
                    context=context,
                    attention_mask=context_attention_mask,
                ).reshape(x_shape[0], x_shape[1], x_shape[2], -1)

            # Temporal attention over the time axis for each keypoint column.
            x = temporal_block.forward(x, attention_mask=temporal_attention_mask)
            # Spatial attention over the keypoint axis for each timestep row.
            x = spatial_block.forward(x, attention_mask=spatial_attention_mask)

        return x[:, 0, 0], x[:, 0, 1:], x[:, 1:, 0], x, x_mask
