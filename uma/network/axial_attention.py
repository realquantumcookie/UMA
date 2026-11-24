import math
from typing import Type, Optional, Tuple
import torch
import torch.nn as nn

from .attention import TransformerBlock, Attention

class AxialAttention(Attention):
    def __init__(self, *args, attn_axis : int, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_axis = attn_axis
    
    def forward(self, src, context = None, attention_mask = None):
        """Performs axial attention on `src` along `attn_axis`.

        Args:
            src (torch.Tensor): Input of shape `(B, ..., N, ..., C)` where `N` is
                the size at `attn_axis` and `C` is the embedding dim.
            context (None): Always None.
            attention_mask (torch.Tensor, optional): Optional self-attention mask
                with trailing shape `(N, N)`. Supported forms:
                `(N, N)`, `(B, N, N)`, or `(B, H, N, N)`,
                If provided with leading batch size `B`, it is repeated across
                non-attention axes to match the internally flattened batch.
        Returns:
            torch.Tensor: Output tensor with the same shape as `src`.
        """
        assert context is None, "Context is not supported in AxialAttention"
        
        if self.attn_axis < 0:
            attn_axis = len(src.shape) + self.attn_axis
        else:
            attn_axis = self.attn_axis

        # Permute the input tensor to bring the attention axis to the second to last dimension (embedding dimension) for attention computation
        permute_dims = list(range(len(src.shape)))
        permute_dims[attn_axis], permute_dims[-2] = permute_dims[-2], permute_dims[attn_axis]
        src = src.permute(permute_dims)
        # Reshape the input tensor to (batch_size * [all remaining dims], sequence_length, embedding_dim)
        batch_size, *sequence_dims, sequence_length, embedding_dim = src.shape
        flattened_batch_size = batch_size * math.prod(sequence_dims) if sequence_dims else batch_size
        if attention_mask is not None:
            # Broadcast batch-level masks across non-attention axes to match flattened batch.
            if attention_mask.ndim in (3, 4) and attention_mask.shape[0] == batch_size and flattened_batch_size != batch_size:
                extra_size = flattened_batch_size // batch_size
                if attention_mask.ndim == 3:
                    attention_mask = attention_mask[:, None].expand(
                        batch_size, extra_size, sequence_length, sequence_length
                    ).reshape(flattened_batch_size, sequence_length, sequence_length)
                else:
                    num_heads = attention_mask.shape[1]
                    attention_mask = attention_mask[:, None].expand(
                        batch_size, extra_size, num_heads, sequence_length, sequence_length
                    ).reshape(flattened_batch_size, num_heads, sequence_length, sequence_length)

        src = src.reshape(flattened_batch_size, sequence_length, embedding_dim)
        # Perform attention
        output = super().forward(src, context, attention_mask)
        # Reshape the output tensor back to the original shape with the attention axis in its original position
        output = output.reshape(batch_size, *sequence_dims, sequence_length, output.shape[-1])
        # Permute the output tensor back to the original dimension order
        output = output.permute(permute_dims)
        return output
