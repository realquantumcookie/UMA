from typing import Optional, Tuple, Any, Type
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, BlockMask, _score_mod_signature

class Mlp(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : Optional[int] = None,
        out_features : Optional[int] = None,
        act_layer : Type[nn.Module] = nn.GELU,
        drop : float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(
        self,
        dim : int,
        num_heads : int = 8,
        qkv_bias : bool = True,
        qkv_scale : Optional[float] = None,
        proj_drop : float = 0.0,
        in_dim_src : Optional[int] = None,
        in_dim_context : Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # None lets scaled_dot_product_attention use its default 1/sqrt(head_dim).
        self.scale = qkv_scale

        self.q_proj = nn.Linear(in_dim_src or dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(in_dim_context or dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(in_dim_context or dim, dim, bias=qkv_bias)

        self.in_dim_src = in_dim_src or dim
        self.in_dim_context = in_dim_context or dim

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    @property
    def dim(self) -> int:
        return self.num_heads * self.head_dim

    def forward(
        self, 
        src : torch.Tensor,
        context : Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Performs attention on the input tensor `x` with optional context.
        Args:
            src (torch.Tensor): The source tensor of shape (B, N, in_dim_src).
            context (Optional[torch.Tensor]): The context tensor of shape (B, L, in_dim_context).
            attention_mask (Optional[torch.Tensor]): A mask to block certain attention scores of shape (B, num_heads, N, L) or (B, N, L).
        Returns:
            torch.Tensor: The output tensor after applying attention, of shape (B, N, dim).
        """
        B, N, _ = src.shape

        if context is None:
            context = src
        
        L = context.shape[1]
        
        q = self.q_proj.forward(src).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        k = self.k_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        v = self.v_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        
        if attention_mask is not None and attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.scale
        ) # (B, num_heads, N, head_dim)

        x = x.transpose(1, 2).flatten(2) # (B, N, dim)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FlexAttention(Attention):
    def __init__(
        self,
        dim : int,
        num_heads : int = 8,
        qkv_bias : bool = True,
        qkv_scale : Optional[float] = None,
        proj_drop : float = 0.0,
        in_dim_src : Optional[int] = None,
        in_dim_context : Optional[int] = None,
        compile : bool = True,
    ):
        super().__init__(dim, num_heads, qkv_bias, qkv_scale, proj_drop, in_dim_src, in_dim_context)
        if compile:
            # Compile per attention module instead of sharing one global compiled
            # function across different mask types and sequence lengths.
            self.flex_attention_fn = torch.compile(flex_attention, dynamic=True)
        else:
            self.flex_attention_fn = flex_attention
        
    def forward(
        self, 
        src : torch.Tensor,
        context : Optional[torch.Tensor] = None,
        score_mod : Optional[_score_mod_signature] = None,
        block_mask: Optional[BlockMask] = None,
    ):
        """
        Performs flexible attention on the input tensor `x` with optional context.
        Args:
            src (torch.Tensor): The source tensor of shape (B, N, in_dim_src).
            context (Optional[torch.Tensor]): The context tensor of shape (B, L, in_dim_context).
            score_mod (Optional[_score_mod_signature]): A function to modify the attention scores
                (default is None, which means no modification).
            block_mask (Optional[BlockMask]): A mask to block certain attention scores (default is None).
        Returns:
            torch.Tensor: The output tensor after applying attention, of shape (B, N, dim).
        """
        B, N, _ = src.shape

        if context is None:
            context = src
        
        L = context.shape[1]
        
        q = self.q_proj.forward(src).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, N, head_dim)
        k = self.k_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        v = self.v_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        
        x = self.flex_attention_fn(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.scale,
        ) # (B, num_heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention : Attention,
        has_context: bool = False,
        mlp_ratio : float = 4.0,
        drop_mlp : float = 0.0,
        drop_path : float = 0.0,
        act_layer : Type[nn.Module] = nn.GELU,
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        pre_norm : bool = True,
    ):
        super().__init__()
        self.pre_norm_src = norm_layer(attention.in_dim_src) if pre_norm else nn.Identity()
        if has_context:
            self.pre_norm_context = norm_layer(attention.in_dim_context) if pre_norm else nn.Identity()
        else:
            self.pre_norm_context = None
        
        if attention.in_dim_src != attention.dim:
            self.in_proj = nn.Linear(attention.in_dim_src, attention.dim)
        else:
            self.in_proj = nn.Identity()
        
        self.attention = attention

        if mlp_ratio > 0:
            self.norm2 = norm_layer(attention.dim)
            self.mlp = Mlp(
                in_features=attention.dim,
                hidden_features=int(attention.dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop_mlp,
            )
        else:
            self.norm2 = None
            self.mlp = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @property
    def has_context(self) -> bool:
        return self.pre_norm_context is not None

    @property
    def num_heads(self) -> int:
        return self.attention.num_heads
    
    @property
    def head_dim(self) -> int:
        return self.attention.head_dim

    @property
    def dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def in_dim_src(self) -> int:
        return self.attention.in_dim_src

    @property
    def in_dim_context(self) -> int:
        return self.attention.in_dim_context

    def forward(
        self, 
        src : torch.Tensor,
        context : Optional[torch.Tensor] = None,
        **kwargs
    ):
        assert (context is not None) == self.has_context, "Context must be provided if the block is configured to use context."
        x = src
        y = self.attention(
            self.pre_norm_src(src), 
            context=self.pre_norm_context(context) if context is not None else None, 
            **kwargs
        )
        x = self.in_proj(x) + self.drop_path(y)
        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Extended modules (SwiGLU FFN, QK-normalized attention)
# ============================================================================

class MlpSwiGLU(nn.Module):
    """
    SwiGLU FFN block with the same interface as :class:`Mlp`:
      - fc1 projects to 2 * hidden_features (gate + value)
      - SiLU on gate, then elementwise multiply
      - fc2 projects back to out_features
      - dropout after the gating and after fc2, as in :class:`Mlp`
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features  # pass e.g. 4 * in_features explicitly for a wider FFN

        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)  # gate + value
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, v = self.fc1(x).chunk(2, dim=-1)  # (..., hidden_features) each
        x = F.silu(g) * v                    # SwiGLU gating
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionWithQKNorm(Attention):
    """
    Attention module with Query-Key normalization.
    Applies normalization (LayerNorm or RMSNorm) to queries and keys after projection, 
    improving training stability.
    
    Args:
        qk_norm_layer: Normalization layer to use (default: nn.LayerNorm).
                       Can be nn.RMSNorm for efficiency.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        in_dim_src: Optional[int] = None,
        in_dim_context: Optional[int] = None,
        qk_norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__(dim, num_heads, qkv_bias, qkv_scale, proj_drop, in_dim_src, in_dim_context)
        
        # Add normalization for queries and keys
        self.q_norm = qk_norm_layer(self.head_dim)
        self.k_norm = qk_norm_layer(self.head_dim)
    
    def forward(
        self,
        src: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with QK normalization.
        """
        B, N, _ = src.shape

        if context is None:
            context = src
        
        L = context.shape[1]
        
        # Project and reshape
        q = self.q_proj.forward(src).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Permute to (B, num_heads, N/L, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        if attention_mask is not None and attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)
        
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            scale=self.scale
        )

        x = x.transpose(1, 2).flatten(2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlexAttentionWithQKNorm(FlexAttention):
    """
    FlexAttention module with Query-Key normalization.
    Applies normalization (LayerNorm or RMSNorm) to queries and keys after projection, 
    improving training stability.
    
    Args:
        qk_norm_layer: Normalization layer to use (default: nn.LayerNorm).
                       Can be nn.RMSNorm for efficiency.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qkv_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        in_dim_src: Optional[int] = None,
        in_dim_context: Optional[int] = None,
        compile: bool = True,
        qk_norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__(dim, num_heads, qkv_bias, qkv_scale, proj_drop, in_dim_src, in_dim_context, compile)
        
        # Add normalization for queries and keys
        self.q_norm = qk_norm_layer(self.head_dim)
        self.k_norm = qk_norm_layer(self.head_dim)
    
    def forward(
        self,
        src: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        score_mod: Optional[_score_mod_signature] = None,
        block_mask: Optional[BlockMask] = None,
    ):
        """
        Forward pass with QK normalization.
        """
        B, N, _ = src.shape

        if context is None:
            context = src
        
        L = context.shape[1]
        
        # Project and reshape
        q = self.q_proj.forward(src).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim)
        v = self.v_proj.forward(context).reshape(B, L, self.num_heads, self.head_dim)
        
        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Permute to (B, num_heads, N/L, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        x = self.flex_attention_fn(
            q, k, v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

