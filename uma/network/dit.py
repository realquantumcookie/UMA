from typing import Optional, Tuple, Type, Union, Any, Dict, List, Literal
import torch
from torch import nn
from .attention import Attention, FlexAttention, TransformerBlock, DropPath, Mlp
import copy

def modulate(
    x : torch.Tensor, 
    shift : torch.Tensor, 
    scale : torch.Tensor, 
    mask : Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C)
        shift (torch.Tensor): Shift tensor of shape (B, C)
        scale (torch.Tensor): Scale tensor of shape (B, C)
        mask (Optional[torch.Tensor]): Optional mask tensor of shape (B, N). Default is None.
            when provided, the modulation is only applied to the positions where mask is True.
    Returns:
        torch.Tensor: Modulated tensor of shape (B, N, C)
    """
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)

    modulated = x * (1 + scale) + shift # (B, N, C)
    if mask is None:
        return modulated
    else:
        return torch.where(
            mask.unsqueeze(-1),
            modulated,
            x
        )

def gate(
    x : torch.Tensor,
    gate : torch.Tensor,
    mask : Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C)
        gate (torch.Tensor): Gate tensor of shape (B, C)
        mask (Optional[torch.Tensor]): Optional mask tensor of shape (B, N). Default is None.
            when provided, the gating is only applied to the positions where mask is True.
    Returns:
        torch.Tensor: Gated tensor of shape (B, N, C)
    """
    gated = x * gate.unsqueeze(1) if gate.ndim == 2 else x * gate # (B, N, C)
    if mask is None:
        return gated
    else:
        return torch.where(
            mask.unsqueeze(-1),
            gated,
            x
        )

class DiTBlock(TransformerBlock):
    def __init__(
        self,
        attention : Attention,
        in_dim_condition : int,
        has_context: bool = False,
        mlp_ratio : float = 4.0,
        drop_mlp : float = 0.0,
        drop_path : float = 0.0,
        act_layer : Type[nn.Module] = nn.GELU,
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        pre_norm : bool = True,
        adaln_zero_init : bool = True,
    ):
        super().__init__(
            attention,
            has_context=has_context,
            mlp_ratio=mlp_ratio,
            drop_mlp=drop_mlp,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pre_norm=pre_norm
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(in_dim_condition or self.dim, self.dim * 6),
            nn.SiLU(),
            nn.Linear(self.dim * 6, self.dim * 6 if mlp_ratio > 0 else self.dim * 3, bias=True)
        )
        # Zero only the last layer (as in DiT): the block starts as the identity, and the first layer still gets
        # gradients. Zeroing the whole MLP would leave SiLU(0) = 0 between two zero layers, so neither weight would
        # train and the modulation could not depend on the condition.
        self.init_network_weights(self.adaLN_modulation[-1], zero_init=adaln_zero_init)
        
        self.in_dim_condition = in_dim_condition or self.dim

    @staticmethod
    def init_network_weights(m: Union[nn.Module, nn.Parameter], zero_init: bool):
        if isinstance(m, nn.Module):
            for p in m.parameters():
                if zero_init:
                    nn.init.zeros_(p)
                else:
                    nn.init.trunc_normal_(p, std=0.02)
            return m
        else:
            return nn.init.zeros_(m) if zero_init else nn.init.trunc_normal_(m, std=0.02)

    def forward(
        self, 
        src : torch.Tensor,
        condition : torch.Tensor,
        context : Optional[torch.Tensor] = None,
        **kwargs
    ):
        assert (context is not None) == self.has_context, "Context must be provided if the block is configured to use context."
        if self.mlp is not None:
            (
                shift_msa, scale_msa, gate_msa,
                shift_mlp, scale_mlp, gate_mlp
            ) = self.adaLN_modulation(condition).chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation(condition).chunk(3, dim=-1)
            shift_mlp = scale_mlp = gate_mlp = None

        x = src
        y = self.attention.forward(
            modulate(self.pre_norm_src(x), shift_msa, scale_msa), 
            context=self.pre_norm_context(context) if context is not None else None, 
            **kwargs
        )
        x = x + self.drop_path(gate(y, gate_msa))
        if self.mlp is not None:
            x = x + self.drop_path(gate(
                self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
                gate_mlp
            ))
        return x

class MaskedDiTBlock(DiTBlock):
    def __init__(
        self,
        attention : Attention,
        in_dim_condition : int,
        has_context: bool = False,
        mlp_ratio : float = 4.0,
        drop_mlp : float = 0.0,
        drop_path : float = 0.0,
        act_layer : Type[nn.Module] = nn.GELU,
        norm_layer : Type[nn.Module] = nn.LayerNorm,
        pre_norm : bool = True,
        adaln_zero_init : bool = True,
    ):
        super().__init__(
            attention,
            in_dim_condition=in_dim_condition,
            has_context=has_context,
            mlp_ratio=mlp_ratio,
            drop_mlp=drop_mlp,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pre_norm=pre_norm,
            adaln_zero_init=adaln_zero_init
        )

        self.adaLN_modulation_unmasked = nn.Parameter(torch.zeros((1, self.dim* (6 if mlp_ratio > 0 else 3))))
        self.adaLN_modulation_unmasked = self.init_network_weights(self.adaLN_modulation_unmasked, zero_init=adaln_zero_init)
    
    def forward(
        self, 
        src : torch.Tensor,
        condition : torch.Tensor,
        condition_mask : torch.Tensor,
        context : Optional[torch.Tensor] = None,
        **kwargs
    ):
        assert (context is not None) == self.has_context, "Context must be provided if the block is configured to use context."
        if self.mlp is not None:
            (
                shift_msa, scale_msa, gate_msa,
                shift_mlp, scale_mlp, gate_mlp
            ) = self.adaLN_modulation(condition).chunk(6, dim=-1)
            (
                shift_msa_unmasked, scale_msa_unmasked, gate_msa_unmasked,
                shift_mlp_unmasked, scale_mlp_unmasked, gate_mlp_unmasked
            ) = self.adaLN_modulation_unmasked.chunk(6, dim=-1)
        else:
            shift_msa, scale_msa, gate_msa = self.adaLN_modulation(condition).chunk(3, dim=-1)
            shift_mlp = scale_mlp = gate_mlp = None
            shift_msa_unmasked, scale_msa_unmasked, gate_msa_unmasked = self.adaLN_modulation_unmasked.chunk(3, dim=-1)
            shift_mlp_unmasked = scale_mlp_unmasked = gate_mlp_unmasked = None

        x = src
        y = self.attention.forward(
            modulate(
                self.pre_norm_src(x),
                torch.where(
                    condition_mask.unsqueeze(-1), shift_msa.unsqueeze(1), shift_msa_unmasked.unsqueeze(1)
                ),
                torch.where(
                    condition_mask.unsqueeze(-1), scale_msa.unsqueeze(1), scale_msa_unmasked.unsqueeze(1)
                )
            ),
            context=self.pre_norm_context(context) if context is not None else None,
            **kwargs
        )
        x = x + self.drop_path(gate(
            y,
            torch.where(
                condition_mask.unsqueeze(-1),
                gate_msa.unsqueeze(1),
                gate_msa_unmasked.unsqueeze(1),
            )
        ))
        if self.mlp is not None:
            x = x + self.drop_path(gate(
                self.mlp(modulate(
                    self.norm2(x),
                    torch.where(
                    condition_mask.unsqueeze(-1),
                        shift_mlp.unsqueeze(1),
                        shift_mlp_unmasked.unsqueeze(1)
                    ), 
                    torch.where(
                        condition_mask.unsqueeze(-1),
                        scale_mlp.unsqueeze(1),
                        scale_mlp_unmasked.unsqueeze(1)
                    )
                )),
                torch.where(
                    condition_mask.unsqueeze(-1),
                    gate_mlp.unsqueeze(1),
                    gate_mlp_unmasked.unsqueeze(1),
                )
            ))
        return x