import abc
from typing import Generic, Optional, Tuple, Union, Dict, Any, Callable, TypeVar, Type, List
from typing_extensions import Self
import torch
from torch import nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tensordict import TensorDict
import copy
import warnings

ContextType = TypeVar("ContextType")
ObsType = TypeVar("ObsType")
try:
    from accelerate import Accelerator
except:
    Accelerator = object
import dataclasses

class PolicyFeatureExtractor(abc.ABC, nn.Module, Generic[ContextType, ObsType]):
    """
    Feature extractor
    """
    action_condition_output_size : int

    @abc.abstractmethod
    def forward(self, context : ContextType, obs : ObsType) -> TensorDict:
        """
        Input:
            context: (B, *context_dim)
            obs: (B, obs_horizon, *obs_dim)
        Output:
            A TensorDict containing:
            - action_condition: (B, action_condition_output_size)
            - other key-value pairs as needed
        """
        raise NotImplementedError

    def compute_auxilary_losses(
        self, context : ContextType, obs : ObsType, output : TensorDict, *args, **kwargs
    ) -> TensorDict:
        """
        Input:
            output: TensorDict of scalars, containing the output of the forward method
        Output:
            A TensorDict containing key value pairs of auxiliary losses.
        """
        return TensorDict()

@dataclasses.dataclass
class PolicyActorConfig:
    pred_horizon : int = 16
    obs_horizon : int = 8
    act_horizon : int = 8
    learning_rate : float = 1e-4
    weight_decay : float = 1e-6
    use_ema : bool = True
    max_grad_norm : float = 10.0

class PolicyActor(abc.ABC, Generic[ContextType, ObsType]):
    act_horizon : int
    pred_horizon : int
    obs_horizon : int

    learning_rate : float
    weight_decay : float
    num_training_steps : int
    num_warmup_steps : int = 1000  # LR warmup, in scheduler steps (see training/train.py)

    optimizer : Optional[torch.optim.Optimizer] = None
    lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler] = None

    feature_extractor : PolicyFeatureExtractor[ContextType, ObsType]

    def __init__(
        self,
        nets : Dict[str, nn.Module],
        feature_extractor : PolicyFeatureExtractor[ContextType, ObsType],
        *,
        loss_weights : Dict[str, float] = {
            "action_loss": 1.0,
        },
    ) -> None:
        self.nets = nets
        self.feature_extractor = feature_extractor
        self.loss_weights = loss_weights
        self.ema_model : Optional[EMAModel] = None
        self.ema_nets : Optional[Dict[str, nn.Module]] = None
        self._ema_nets_dirty : bool = False
    
    def _init_optimizer(self) -> None:
        parameters = []
        for net in self.nets.values():
            parameters.extend(net.parameters())

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

    def _init_emanet(
        self,
        *,
        accelerator : Optional["Accelerator"] = None,
    ) -> None:
        if accelerator is not None:
            self.ema_nets = copy.deepcopy({
                key: accelerator.unwrap_model(net) for key, net in self.nets.items()
            })
        else:
            self.ema_nets = copy.deepcopy(self.nets)

        parameters = []
        for net in self.ema_nets.values():
            parameters.extend(net.parameters())
        
        self.ema_model = EMAModel(
            parameters=parameters,
            power=0.75
        )
        for net in self.ema_nets.values():
            net.eval()
        self._ema_nets_dirty = False

    def _sync_emanet(self) -> None:
        """
        Synchronizes weights inside `self.ema_model` into `self.ema_nets` if we want to predict actions using EMA model.
        This is useful when we want to use the EMA model for prediction after training.
        """
        if self.ema_model is not None and self._ema_nets_dirty:
            target_parameters = []
            for net in self.ema_nets.values():
                target_parameters.extend(net.parameters())
            self.ema_model.copy_to(target_parameters)
            self._ema_nets_dirty = False

    def _step_emanet(
        self, 
        *,
        accelerator : Optional["Accelerator"] = None
    ) -> None:
        """
        This updates the EMA model with the latest weights from `self.nets`.
        """
        if self.ema_model is None:
            return

        parameters = []
        if accelerator is not None:
            for net in self.nets.values():
                parameters.extend(accelerator.unwrap_model(net).parameters())
        else:
            for net in self.nets.values():
                parameters.extend(net.parameters())
        self.ema_model.step(parameters)
        self._ema_nets_dirty = True
    
    def train(self, mode: bool = True) -> Self:
        for net in self.nets.values():
            net.train(mode)
        return self

    def eval(self) -> Self:
        self.train(False)

    def predict_executable_action(
        self,
        context : ContextType,
        obs : ObsType,
        use_ema : bool = True,
        *,
        accelerator : Optional["Accelerator"] = None,
        **kwargs,
    ) -> TensorDict:
        """
        Predicts an executable action based on the context and observation.
        Args:
            context: (B, *context_dim)
            obs: (B, obs_horizon, *obs_dim)
            use_ema: Whether to use the EMA model for prediction if available.
        Returns:
            A TensorDict with batch size (B, ) containing:
            - "executable_action": (B, action_horizon, *action_dim)
            - "action_condition": (B, action_condition_output_size)
            - other key-value pairs as needed
        """
        if use_ema and self.ema_model is not None:
            self._sync_emanet()
            model = self.ema_nets
        else:
            model = self.nets if accelerator is None else {
                key: accelerator.unwrap_model(net) for key, net in self.nets.items()
            }
        with torch.no_grad():
            return self.predict_executable_action_with_nets(
                model,
                context,
                obs,
                **kwargs,
            )
    
    @abc.abstractmethod
    def predict_executable_action_with_nets(
        self,
        nets : Dict[str, nn.Module], 
        context : ContextType,
        obs : ObsType, 
        **kwargs,
    ) -> TensorDict:
        """
        Predicts an action using the provided networks.
        Args:
            nets: A dictionary of networks to use for prediction.
            context: (B, *context_dim)
            obs: (B, obs_horizon, *obs_dim)
        Returns:
            A TensorDict with batch size (B, ) containing:
            - "executable_action": (B, action_horizon, *action_dim)
            - "action_condition": (B, action_condition_output_size)
            - other key-value pairs as needed
        """
        raise NotImplementedError
    
    def prepare_accelerate(self, *args, accelerator : "Accelerator") -> Tuple[Any, ...]:
        """
        Prepare the actor for distributed training using the provided Accelerator.
        Args:
            accelerator: An instance of Accelerator from the accelerate library.
            *args: Additional arguments for preparation (dataset, dataloader, etc.).
        Returns:
            *prepared_args: Tuple of prepared arguments
        """
        nets_values = list(self.nets.values())
        len_nets = len(nets_values)
        ret = accelerator.prepare(
            *nets_values,
            self.optimizer,
            self.lr_scheduler,
            *args,
        )
        *new_nets_values, self.optimizer, self.lr_scheduler = ret[:len_nets + 2]
        self.nets = {
            key: net for key, net in zip(self.nets.keys(), new_nets_values)
        }
        if self.ema_model is not None:
            self._init_emanet(accelerator=accelerator)
        return ret[len_nets+2:]
    
    @property
    def device(self) -> torch.device:
        first_net = next(iter(self.nets.values()))
        return next(first_net.parameters()).device

    def to(self, device : Union[str, torch.device]) -> Self:
        self.nets = {
            key: net.to(device) for key, net in self.nets.items()
        }
        if self.ema_model is not None:
            self.ema_model.to(device)
            self.ema_nets = {
                key: net.to(device) for key, net in self.ema_nets.items()
            }
        self._init_optimizer()
        return self
    
    @abc.abstractmethod
    def compute_losses(
        self,
        context : ContextType,
        obs : ObsType,
        ref_action : torch.Tensor,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the losses for the actor based on the context, observation, and reference action.
        Args:
            context: (B, *context_dim)
            obs: (B, obs_horizon, *obs_dim)
            ref_action: (B, action_horizon, *action_dim)
            *args: Additional arguments for loss computation.
            **kwargs: Additional keyword arguments for loss computation.
            Note that the additional arguments and keyword arguments are passed to the feature extractor's auxiliary loss computation method.
        Returns:
            A dictionary of loss names and their corresponding loss values, including:
            - "total_loss": (scalar) total loss value after weighting and summing
            - "action_loss": (scalar) loss for the action prediction
            - other key-value pairs as needed
        """
        raise NotImplementedError

    def update(
        self,
        context : ContextType,
        obs : ObsType,
        ref_action : torch.Tensor,
        *args,
        accelerator : Optional["Accelerator"] = None,
        max_grad_norm : Optional[float] = None,
        sync_losses : bool = True,
        **kwargs,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Updates the actor's parameters based on the context, observation, and reference action.
        Args:
            context: (B, *context_dim)
            obs: (B, obs_horizon, *obs_dim)
            ref_action: (B, action_horizon, *action_dim)
            sync_losses: ``True`` checks the total loss for non-finite values
                before the backward pass and returns Python floats; ``False``
                returns detached device tensors and skips the check, so the
                step never waits on the GPU (the caller checks the values
                when it reads them).
        Returns:
            A dictionary of metrics after the update.
        """
        assert self.optimizer is not None, "Optimizer must be initialized before calling update."
        if accelerator is not None:
            with accelerator.autocast():
                losses = self.compute_losses(
                    context, obs, ref_action, *args, **kwargs
                )
        else:
            losses = self.compute_losses(
                context, obs, ref_action, *args, **kwargs
            )
        total_loss = losses['total_loss']
        if sync_losses and not torch.isfinite(total_loss):
            scalar_losses = {
                key: float(value.detach().float().cpu())
                for key, value in losses.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            raise FloatingPointError(f"Non-finite total loss detected: {scalar_losses}")

        if accelerator is not None:
            accelerator.backward(total_loss)
        else:
            total_loss.backward()

        grad_norm = None
        
        should_sync_gradients = accelerator is None or accelerator.sync_gradients
        if should_sync_gradients and max_grad_norm is not None and max_grad_norm > 0:
            parameters = [
                param
                for net in self.nets.values()
                for param in net.parameters()
                if param.requires_grad
            ]
            if accelerator is not None:
                grad_norm = accelerator.clip_grad_norm_(parameters, max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.optimizer.zero_grad()

        if should_sync_gradients:
            self._step_emanet(accelerator=accelerator)
        
        if not sync_losses:
            ret = {key: value.detach() for key, value in losses.items()}
            if grad_norm is not None:
                ret['grad_norm'] = grad_norm.detach() if isinstance(grad_norm, torch.Tensor) else grad_norm
            return ret
        ret = {
            key: value.item()
            for key, value in losses.items()
        }
        if grad_norm is not None:
            ret['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        return ret

    def state_dict(self, *, accelerator : Optional["Accelerator"] = None) -> Dict[str, Any]:
        unwrapped_nets = self.nets if accelerator is None else {
            key: accelerator.unwrap_model(net) for key, net in self.nets.items()
        }

        ret = {
            'nets': {
                key: net.state_dict() for key, net in unwrapped_nets.items()
            }
        }
        if self.optimizer is not None:
            ret['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            ret['lr_scheduler'] = self.lr_scheduler.state_dict()
        if self.ema_model is not None:
            ret['ema_model'] = self.ema_model.state_dict()
        return ret
    
    def load_state_dict(self, state_dict : Dict[str, Any]) -> None:
        for key, net in self.nets.items():
            if key in state_dict['nets']:
                load_result = net.load_state_dict({
                    k if not k.startswith("_orig_mod.") else k[len("_orig_mod."):]: v
                    for k, v in state_dict['nets'][key].items()
                }, strict=False)
                if load_result.missing_keys:
                    warnings.warn(
                        f"Missing keys when loading state_dict for net '{key}': {load_result.missing_keys}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                raise KeyError(f"Key '{key}' not found in state_dict['nets']. Available keys: {list(state_dict['nets'].keys())}")
        if self.ema_model is not None and 'ema_model' in state_dict:
            self.ema_model.load_state_dict(state_dict['ema_model'])
            self._ema_nets_dirty = True
        elif self.ema_model is not None:
            self.ema_nets = copy.deepcopy(self.nets)
            for net in self.ema_nets.values():
                net.eval()
            self._ema_nets_dirty = False

        optimizer_loaded = 'optimizer' not in state_dict or self.optimizer is None
        if 'optimizer' in state_dict and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state_dict['optimizer'])
                optimizer_loaded = True
            except ValueError as exc:
                warnings.warn(
                    f"Skipping optimizer state because it is incompatible with the current model: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if 'lr_scheduler' in state_dict and self.lr_scheduler is not None:
            if optimizer_loaded:
                self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
            else:
                warnings.warn(
                    "Skipping lr_scheduler state because optimizer state was not loaded.",
                    RuntimeWarning,
                    stacklevel=2,
                )
