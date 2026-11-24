from typing import Tuple, Optional, Union, Sequence
import torch
from torch import nn

__all__ = [
    "pad_dim_to_power_of_2",
    "pad_dim_to_multiples_of",
    "pad_tensors_dim_to_power_of_2",
    "pad_tensors_dim_to_multiples_of",
    "masked_mean_mse_loss",
    "masked_mean_rmse_loss",
    "masked_mean_huber_loss",
    "masked_bce_loss_with_logits",
    "masked_same_sign_prob",
]

def pad_dim_to_power_of_2(
    data : torch.Tensor,
    dim : int = -1,
    pad_value : Union[float, int, bool] = 0
) -> torch.Tensor:
    """
    Pads the specified dimension of a tensor to the next power of 2. 
    This is useful for optimizing performance with `torch.compile`'s limited cache size, when the input dimension is variable.
    """

    if dim < 0:
        dim += data.dim()
    
    current_size = data.size(dim)
    next_power_of_2 = 1 << (current_size - 1).bit_length()
    
    if current_size == next_power_of_2:
        return data
    
    pad_size = next_power_of_2 - current_size
    pad_shape = list(data.shape)
    pad_shape[dim] = pad_size
    
    padding_tensor = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)
    
    return torch.cat([data, padding_tensor], dim=dim)

def pad_dim_to_multiples_of(
    data : torch.Tensor,
    dim : int = -1,
    multiplier : int = 16,
    pad_value : Union[float, int, bool] = 0
) -> torch.Tensor:
    if dim < 0:
        dim += data.dim()

    current_size = data.size(dim)
    target_size = ((current_size + multiplier - 1) // multiplier) * multiplier

    if current_size == target_size:
        return data

    pad_size = target_size - current_size
    pad_shape = list(data.shape)
    pad_shape[dim] = pad_size

    padding_tensor = torch.full(pad_shape, pad_value, dtype=data.dtype, device=data.device)

    return torch.cat([data, padding_tensor], dim=dim)

def pad_tensors_dim_to_power_of_2(
    *datas : torch.Tensor,
    dim : int = -1,
    pad_value : Union[float, int, bool, Sequence[Union[float, int, bool]]] = 0
) -> Tuple[torch.Tensor, ...]:
    """
    Pads the specified dimension of multiple tensors to the next power of 2. 
    This is useful for optimizing performance with `torch.compile`'s limited cache size, when the input dimension is variable.
    """
    return tuple(pad_dim_to_power_of_2(data, dim=dim, pad_value=pad_value[i] if isinstance(pad_value, Sequence) else pad_value) for i, data in enumerate(datas))

def pad_tensors_dim_to_multiples_of(
    *datas : torch.Tensor,
    dim : int = -1,
    multiplier : int = 16,
    pad_value : Union[float, int, bool, Sequence[Union[float, int, bool]]] = 0
) -> Tuple[torch.Tensor, ...]:
    """
    Pads the specified dimension of multiple tensors to the next multiple of a given number.
    This is useful for optimizing performance with `torch.compile`'s limited cache size, when the input dimension is variable.
    """
    return tuple(pad_dim_to_multiples_of(data, dim=dim, multiplier=multiplier, pad_value=pad_value[i] if isinstance(pad_value, Sequence) else pad_value) for i, data in enumerate(datas))

def masked_mean_mse_loss(
    pred : torch.Tensor,
    target : torch.Tensor,
    weight : Optional[torch.Tensor] = None,
    valid_mask : Optional[torch.Tensor] = None,
    max_diff : Optional[float] = None,
) -> torch.Tensor:
    if any(dim == 0 for dim in pred.shape):
        return pred.sum() * 0.0

    pred_f = pred.float()
    target_f = target.detach().float()

    if weight is None and valid_mask is None:
        return nn.functional.mse_loss(pred_f, target_f, reduction='mean').to(pred.dtype)

    diff = pred_f - target_f
    if max_diff is not None:
        diff = torch.clamp(diff, -max_diff, max_diff)
    squared_error = diff ** 2

    expanded_valid_mask = None
    if valid_mask is not None:
        expanded_valid_mask = valid_mask.reshape(valid_mask.shape + (1,) * (len(pred.shape) - len(valid_mask.shape))) if len(valid_mask.shape) < len(pred.shape) else valid_mask
        squared_error = squared_error.masked_fill(~expanded_valid_mask, 0.0)

    expanded_weight = None
    if weight is not None:
        expanded_weight = weight.reshape(weight.shape + (1,) * (len(pred.shape) - len(weight.shape))) if len(weight.shape) < len(pred.shape) else weight
        expanded_weight = expanded_weight.float().expand_as(squared_error)
        if expanded_valid_mask is not None:
            expanded_weight = expanded_weight.masked_fill(~expanded_valid_mask, 0.0)
        squared_error = squared_error * expanded_weight.detach()

    if weight is not None:
        mean_squared_error = squared_error.sum() / torch.clamp(expanded_weight.sum(), min=1e-6)
    else:
        valid_denominator = expanded_valid_mask.expand_as(squared_error).sum()
        mean_squared_error = squared_error.sum() / torch.clamp(valid_denominator, min=1e-6).float()

    return mean_squared_error.to(pred.dtype)

def masked_mean_rmse_loss(
    pred : torch.Tensor,
    target : torch.Tensor,
    weight : Optional[torch.Tensor] = None,
    valid_mask : Optional[torch.Tensor] = None,
    max_diff : Optional[float] = None,
    eps : float = 1e-8,
) -> torch.Tensor:
    if any(dim == 0 for dim in pred.shape):
        return pred.sum() * 0.0

    pred_f = pred.float()
    target_f = target.detach().float()

    if weight is None and valid_mask is None:
        mean_squared_error = nn.functional.mse_loss(pred_f, target_f, reduction='mean')
        return torch.sqrt(mean_squared_error + eps).to(pred.dtype)

    diff = pred_f - target_f
    if max_diff is not None:
        diff = torch.clamp(diff, -max_diff, max_diff)
    squared_error = diff ** 2
    
    expanded_valid_mask = None
    if valid_mask is not None:
        expanded_valid_mask = valid_mask.reshape(valid_mask.shape + (1,) * (len(pred.shape) - len(valid_mask.shape))) if len(valid_mask.shape) < len(pred.shape) else valid_mask
        squared_error = squared_error.masked_fill(~expanded_valid_mask, 0.0)

    expanded_weight = None
    if weight is not None:
        expanded_weight = weight.reshape(weight.shape + (1,) * (len(pred.shape) - len(weight.shape))) if len(weight.shape) < len(pred.shape) else weight
        expanded_weight = expanded_weight.float().expand_as(squared_error)
        if expanded_valid_mask is not None:
            expanded_weight = expanded_weight.masked_fill(~expanded_valid_mask, 0.0)
        squared_error = squared_error * expanded_weight
    
    if weight is not None:
        mean_squared_error = squared_error.sum() / torch.clamp(expanded_weight.sum(), min=1e-6)
    else:
        mean_squared_error = squared_error.sum() / torch.clamp(expanded_valid_mask.sum(), min=1e-6).float()

    return torch.sqrt(mean_squared_error + eps).to(pred.dtype)

def masked_mean_huber_loss(
    pred : torch.Tensor,
    target : torch.Tensor,
    weight : Optional[torch.Tensor] = None,
    valid_mask : Optional[torch.Tensor] = None,
    max_diff : Optional[float] = None,
    delta : float = 2.0,
) -> torch.Tensor:
    if any(dim == 0 for dim in pred.shape):
        return pred.sum() * 0.0

    pred_f = pred.float()
    target_f = target.detach().float()
    diff = pred_f - target_f
    if max_diff is not None:
        diff = torch.clamp(diff, -max_diff, max_diff)

    abs_diff = diff.abs()
    quadratic = abs_diff.clamp(max=delta)  # not torch.minimum(x, new_tensor(delta)): no host-to-device copy
    linear = abs_diff - quadratic
    huber_loss = 0.5 * quadratic ** 2 + delta * linear

    if weight is None and valid_mask is None:
        return huber_loss.mean().to(pred.dtype)

    expanded_valid_mask = None
    if valid_mask is not None:
        expanded_valid_mask = valid_mask.reshape(valid_mask.shape + (1,) * (len(pred.shape) - len(valid_mask.shape))) if len(valid_mask.shape) < len(pred.shape) else valid_mask
        huber_loss = huber_loss.masked_fill(~expanded_valid_mask, 0.0)

    expanded_weight = None
    if weight is not None:
        expanded_weight = weight.reshape(weight.shape + (1,) * (len(pred.shape) - len(weight.shape))) if len(weight.shape) < len(pred.shape) else weight
        expanded_weight = expanded_weight.float().expand_as(huber_loss)
        if expanded_valid_mask is not None:
            expanded_weight = expanded_weight.masked_fill(~expanded_valid_mask, 0.0)
        huber_loss = huber_loss * expanded_weight.detach()

    if weight is not None:
        mean_huber_loss = huber_loss.sum() / torch.clamp(expanded_weight.sum(), min=1e-6)
    else:
        valid_denominator = expanded_valid_mask.expand_as(huber_loss).sum()
        mean_huber_loss = huber_loss.sum() / torch.clamp(valid_denominator, min=1e-6).float()

    return mean_huber_loss.to(pred.dtype)

def masked_bce_loss_with_logits(
    pred_logits : torch.Tensor,
    target : torch.Tensor,
    weight : Optional[torch.Tensor] = None,
    valid_mask : Optional[torch.Tensor] = None
) -> torch.Tensor:
    if any(dim == 0 for dim in pred_logits.shape):
        return 0

    if weight is None and valid_mask is None:
        return nn.functional.binary_cross_entropy_with_logits(pred_logits, target.detach(), reduction='mean')
    
    bce_loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, target.detach(), reduction='none')
    
    expanded_valid_mask = None
    if valid_mask is not None:
        expanded_valid_mask = valid_mask.reshape(valid_mask.shape + (1,) * (len(pred_logits.shape) - len(valid_mask.shape))) if len(valid_mask.shape) < len(pred_logits.shape) else valid_mask
        bce_loss = bce_loss.masked_fill(~expanded_valid_mask, 0.0)

    expanded_weight = None
    if weight is not None:
        expanded_weight = weight.reshape(weight.shape + (1,) * (len(pred_logits.shape) - len(weight.shape))) if len(weight.shape) < len(pred_logits.shape) else weight
        expanded_weight = expanded_weight.float().expand_as(bce_loss)
        if expanded_valid_mask is not None:
            expanded_weight = expanded_weight.masked_fill(~expanded_valid_mask, 0.0)
        bce_loss = bce_loss * expanded_weight.detach()

    if weight is not None:
        mean_bce_loss = bce_loss.sum() / torch.clamp(expanded_weight.sum(), min=1e-6)
    else:
        mean_bce_loss = bce_loss.sum() / torch.clamp(expanded_valid_mask.sum(), min=1e-6).float()

    return mean_bce_loss

def masked_same_sign_prob(
    pred_logits : torch.Tensor,
    target : torch.Tensor,
    weight : Optional[torch.Tensor] = None,
    valid_mask : Optional[torch.Tensor] = None
) -> torch.Tensor:
    if any(dim == 0 for dim in pred_logits.shape):
        return torch.tensor(0.0, device=pred_logits.device)

    correct = (torch.sign(pred_logits) == torch.sign(target.detach())).to(pred_logits.dtype)

    if weight is None and valid_mask is None:
        return correct.mean()

    expanded_valid_mask = None
    if valid_mask is not None:
        expanded_valid_mask = valid_mask.reshape(valid_mask.shape + (1,) * (len(pred_logits.shape) - len(valid_mask.shape))) if len(valid_mask.shape) < len(pred_logits.shape) else valid_mask
        correct = correct.masked_fill(~expanded_valid_mask, 0.0)

    expanded_weight = None
    if weight is not None:
        expanded_weight = weight.reshape(weight.shape + (1,) * (len(pred_logits.shape) - len(weight.shape))) if len(weight.shape) < len(pred_logits.shape) else weight
        expanded_weight = expanded_weight.float().expand_as(correct)
        if expanded_valid_mask is not None:
            expanded_weight = expanded_weight.masked_fill(~expanded_valid_mask, 0.0)
        correct = correct * expanded_weight.detach()

    if weight is not None:
        mean_correct = correct.sum() / torch.clamp(expanded_weight.sum(), min=1e-6)
    else:
        mean_correct = correct.sum() / torch.clamp(expanded_valid_mask.sum(), min=1e-6).float()
    
    return mean_correct
