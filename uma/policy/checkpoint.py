from pathlib import Path
from typing import Optional, Union, Tuple, List
import torch
import torch.nn as nn
import os
from .base import PolicyActor
import shutil
import warnings

class ActorCheckPointer:
    def __init__(
        self,
        save_path : Union[str, os.PathLike],
        prefix : str = "",
        format : str = ".pt"
    ):
        assert len(format) > 1 and format[0] == ".", "Format must start with a dot (.)"
        self.save_path = Path(save_path)
        self.prefix = prefix
        self.format = format

    @staticmethod
    def _normalize_state_key(key : str) -> str:
        return key if not key.startswith("_orig_mod.") else key[len("_orig_mod."):]

    def _warn_missing_net_keys(self, module : PolicyActor, state_dict, accelerator = None) -> None:
        if 'nets' not in state_dict:
            return

        for key, net in module.nets.items():
            if key not in state_dict['nets']:
                continue

            target_net = accelerator.unwrap_model(net) if accelerator is not None else net
            checkpoint_keys = {
                self._normalize_state_key(state_key)
                for state_key in state_dict['nets'][key].keys()
            }
            target_keys = {
                self._normalize_state_key(state_key)
                for state_key in target_net.state_dict().keys()
            }
            missing_keys = sorted(target_keys - checkpoint_keys)
            if missing_keys:
                warnings.warn(
                    f"Missing keys when loading state_dict for net '{key}': {missing_keys}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    
    def list_checkpoint_steps(self) -> List[int]:
        if not self.save_path.is_dir():
            return []
        
        filenames = self.save_path.glob(f"{self.prefix}*{self.format}")
        steps = []
        for filename in filenames:
            try:
                step = int(filename.stem[len(self.prefix):])
                steps.append(step)
            except ValueError:
                continue
        return sorted(steps)

    def try_load_latest_checkpoint(
        self, 
        module : PolicyActor,
        device : Optional[Union[str, torch.device]] = None,
        accelerator = None
    ) -> Optional[Tuple[int, PolicyActor]]:
        steps = self.list_checkpoint_steps()
        if len(steps) == 0:
            return None, module
        
        latest_step = steps[-1]
        new_module = self.try_load_checkpoint_at_step(
            module, 
            latest_step, 
            device=device,
            accelerator=accelerator
        )
        if new_module is None:
            return None, module
        else:
            return latest_step, new_module

    def try_load_checkpoint_at_step(
        self, 
        module : PolicyActor, 
        step : int,
        device : Optional[Union[str, torch.device]] = None,
        accelerator = None,
    ) -> Optional[PolicyActor]:
        checkpoint_path = self.save_path / f"{self.prefix}{step}{self.format}"
        print(f"Loading checkpoint from {checkpoint_path}")
        if not checkpoint_path.exists():
            return None
        
        if device is not None:
            module.to(device)
        
        state_dict = torch.load(checkpoint_path, map_location=module.device, weights_only=True)
        if accelerator is not None:
            model_state_path = self.save_path / f"{self.prefix}{step}_state"
            if not model_state_path.exists():
                return None
            self._warn_missing_net_keys(module, state_dict, accelerator=accelerator)
            if module.ema_model is not None:
                module.ema_model.load_state_dict(state_dict['ema_model'])
                module._ema_nets_dirty = True
            try:
                accelerator.load_state(model_state_path, strict=False)
            except FileNotFoundError as exc:
                missing_path = Path(exc.filename) if exc.filename is not None else None
                scheduler_state_path = model_state_path / "scheduler.bin"
                if missing_path == scheduler_state_path:
                    warnings.warn(
                        f"Loaded model weights but skipped accelerator scheduler state because '{scheduler_state_path}' is missing.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                else:
                    raise
            except ValueError as exc:
                warnings.warn(
                    f"Loaded model weights but skipped incompatible accelerator optimizer/scheduler state: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            module.load_state_dict(state_dict)
        return module

    def save_checkpoint_at_step(
        self, 
        module : PolicyActor, 
        step : int,
        accelerator = None
    ) -> None:
        if (accelerator is None or accelerator.is_main_process) and not self.save_path.is_dir():
            self.save_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving checkpoint {self.prefix}{step} to {self.save_path}")
        if accelerator is not None and accelerator.is_main_process:
            accelerator.save_state(self.save_path / f"{self.prefix}{step}_state")
        
        if accelerator is None or accelerator.is_main_process:
            torch.save(module.state_dict(accelerator=accelerator), self.save_path / f"{self.prefix}{step}{self.format}")
    
    def remove_checkpoint_at_step(
        self,
        step : Optional[int] = None,
    ) -> None:
        if step is None:
            for filename in self.save_path.glob(f"{self.prefix}*{self.format}"):
                filename.unlink(missing_ok=True)
            for filename in self.save_path.glob(f"{self.prefix}*_state"):
                if filename.is_dir():
                    shutil.rmtree(filename, ignore_errors=True)
                else:
                    filename.unlink(missing_ok=True)
        else:
            (self.save_path / f"{self.prefix}{step}{self.format}").unlink(missing_ok=True)
            state_path = self.save_path / f"{self.prefix}{step}_state"
            if state_path.is_dir():
                shutil.rmtree(state_path, ignore_errors=True)
            else:
                state_path.unlink(missing_ok=True)
    
    def save_wandb(
        self,
        step : Optional[int],
        base_path : Union[str, os.PathLike] = None,
    ) -> None:
        import wandb
        if base_path is not None:
            base_path = os.path.join(self.save_path, base_path)
        if step is None:
            wandb.save(
                self.save_path / "*",
                base_path=base_path
            )
        else:
            wandb.save(
                self.save_path / f"{self.prefix}{step}{self.format}",
                base_path=base_path
            )
            state_path = self.save_path / f"{self.prefix}{step}_state"
            if state_path.is_dir():
                wandb.save(
                    state_path / "*",
                    base_path=base_path
                )
        config_path = self.save_path / f"{self.prefix}config.json"
        if config_path.exists():
            wandb.save(
                config_path,
                base_path=base_path
            )
