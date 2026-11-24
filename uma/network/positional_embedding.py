from typing import Optional, Union, Type
import torch
from torch import nn
import math

class TimestepPosEmb(nn.Module):
    """
    https://github.com/huggingface/lerobot/blob/a5727e37b4ff405f9d8de424e07dcc441aa2c82f/lerobot/common/policies/pi0/modeling_pi0.py#L71
    """
    def __init__(
        self, 
        dim : int,
        min_period : float = 4e-3,
        max_period : float = 4.0,
        dtype : torch.dtype = torch.float32,
        device : Union[torch.device, str] = "cpu"
    ):
        if dim % 2 != 0:
            raise ValueError(f"Embedding dim ({dim}) must be divisible by 2")
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer("scaling_factor", __class__.compute_scaling_factor(
            dim // 2,
            min_period,
            max_period,
            dtype=dtype,
            device=device
        )) # (dim // 2, )
    
    @staticmethod
    def compute_scaling_factor(
        half_dim : int,
        min_period : float,
        max_period : float,
        dtype : torch.dtype = torch.float32,
        device : Union[torch.device, str] = "cpu"
    ) -> torch.Tensor:
        fraction = torch.linspace(0.0, 1.0, half_dim, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        return scaling_factor

    def compute_embedding_with_position(self, time : torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embedding for a given time tensor.
        Args:
            time (torch.Tensor): A tensor of shape (..., 1) representing the time.
        Returns:
            torch.Tensor: Positional embedding of shape (..., dim).
        """
        return self.forward(time)

    def forward(self, time : torch.Tensor):
        sin_input = torch.reshape(
            self.scaling_factor,
            [1] * len(time.shape) + [-1]
        ) * time[..., None]
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
        return pos_emb

class PosEmb3D(nn.Module):
    """
    Positional embedding for 3D coordinates.
    """
    
    # Reuse the compute_scaling_factor method from TimestepPosEmb
    compute_scaling_factor = TimestepPosEmb.compute_scaling_factor

    def __init__(
        self, 
        dim : int,
        min_period : float = 4e-3,
        max_period : float = 4.0,
        min_value : Union[float, torch.Tensor] = 0.0,
        max_value : Union[float, torch.Tensor] = 1.0,
        dtype : torch.dtype = torch.float32,
        device : Union[torch.device, str] = "cpu"
    ):
        if dim % 6 != 0:
            raise ValueError(f"Embedding dim ({dim}) must be divisible by 6")
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer("scaling_factor", __class__.compute_scaling_factor(
            dim // 6,
            min_period,
            max_period,
            dtype=dtype,
            device=device
        )) # (dim // 6, )
        if isinstance(min_value, float):
            self.min_value = min_value
        else:
            self.register_buffer("min_value", min_value)
        if isinstance(max_value, float):
            self.max_value = max_value
        else:
            self.register_buffer("max_value", max_value)

    def compute_embedding_with_position(self, pos : torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embedding for a given pos tensor.
        Args:
            pos (torch.Tensor): A tensor of shape (..., 3) representing the position.
        Returns:
            torch.Tensor: Positional embedding of shape (..., dim).
        """
        return self.forward(pos)

    def forward(self, pos : torch.Tensor):
        min_expanded = self.min_value if isinstance(self.min_value, float) else torch.reshape(
            self.min_value,
            [1] * len(pos.shape[:-1]) + [-1]
        )
        max_expanded = self.max_value if isinstance(self.max_value, float) else torch.reshape(
            self.max_value,
            [1] * len(pos.shape[:-1]) + [-1]
        )
        sin_input = torch.reshape(
            self.scaling_factor,
            [1] * len(pos.shape) + [-1]
        ) * ((pos - min_expanded) / (max_expanded - min_expanded))[..., None] # (..., 3, dim // 6)
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1) # (..., 3, dim // 3)
        pos_emb = torch.flatten(pos_emb, start_dim=-2) # (..., dim)
        return pos_emb

class PosEmb2D(nn.Module):
    """
    Positional embedding for 2D coordinates.
    """

    compute_scaling_factor = TimestepPosEmb.compute_scaling_factor

    def __init__(
        self,
        dim : int,
        min_period : float = 4e-3,
        max_period : float = 4.0,
        min_value : Union[float, torch.Tensor] = 0.0,
        max_value : Union[float, torch.Tensor] = 1.0,
        dtype : torch.dtype = torch.float32,
        device : Union[torch.device, str] = "cpu"
    ):
        if dim % 4 != 0:
            raise ValueError(f"Embedding dim ({dim}) must be divisible by 4")
        super().__init__()
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer("scaling_factor", __class__.compute_scaling_factor(
            dim // 4,
            min_period,
            max_period,
            dtype=dtype,
            device=device
        )) # (dim // 4, )
        if isinstance(min_value, float):
            self.min_value = min_value
        else:
            self.register_buffer("min_value", min_value)
        if isinstance(max_value, float):
            self.max_value = max_value
        else:
            self.register_buffer("max_value", max_value)

    def compute_embedding_with_position(self, pos : torch.Tensor) -> torch.Tensor:
        """
        Computes the positional embedding for a given pos tensor.
        Args:
            pos (torch.Tensor): A tensor of shape (..., 2) representing the position.
        Returns:
            torch.Tensor: Positional embedding of shape (..., dim).
        """
        return self.forward(pos)

    def forward(self, pos : torch.Tensor):
        min_expanded = self.min_value if isinstance(self.min_value, float) else torch.reshape(
            self.min_value,
            [1] * len(pos.shape[:-1]) + [-1]
        )
        max_expanded = self.max_value if isinstance(self.max_value, float) else torch.reshape(
            self.max_value,
            [1] * len(pos.shape[:-1]) + [-1]
        )
        sin_input = torch.reshape(
            self.scaling_factor,
            [1] * len(pos.shape) + [-1]
        ) * ((pos - min_expanded) / (max_expanded - min_expanded))[..., None] # (..., 2, dim // 4)
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1) # (..., 2, dim // 2)
        pos_emb = torch.flatten(pos_emb, start_dim=-2) # (..., dim)
        return pos_emb