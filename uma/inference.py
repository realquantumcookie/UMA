"""Load a UMA checkpoint and run a forward pass from plain tensors.

UMA predicts, from one RGB-D camera view (plus an optional wrist image) and the
robot's recent state, a chunk of end-effector actions and the future motion of
tracked points, conditioned on a *motion prompt* (the desired 3D trajectory of a
few points on the object to move) or on a language instruction.

Conventions (all tensors batched with a leading ``B``):

* **Frames.** 3D quantities are given in the robot-base frame. ``extrinsics`` is
  the ``(3, 4)`` transform from the robot-base frame to the camera frame
  (``x_cam = R x_base + t``); ``intrinsics`` is the ``(3, 3)`` pinhole matrix of
  the image passed in.
* **Proprioception / actions.** 10-D: end-effector position (3, metres), rotation
  as the first two rows of the rotation matrix (6), gripper (1, ``0`` open …
  ``1`` closed). :func:`proprioception_from_pose` builds it from a position and a
  ``(w, x, y, z)`` quaternion.
* **Time.** The model sees the last ``obs_horizon`` (8) control steps (15 Hz,
  oldest first) and predicts ``pred_horizon`` (16) future steps of actions; flows
  are also predicted at the far-future steps ``additional_times``.

Example::

    import torch
    from uma.inference import load_model, UMAObservation, predict

    model = load_model(device="cuda")  # the pretrained checkpoint, downloaded on first use
    out = predict(model, observation, motion_prompt)  # see examples/forward_pass.py
    out["actions"]  # (B, 16, 10) end-effector targets in the robot-base frame
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from uma.policy.masked_policy import (
    UMAActor,
    UMAFlowSpecContextItem,
    UMAImageContext,
    UMAMaskedTokens,
    UMATokenItem,
    transform_3d,
)

__all__ = [
    "DEFAULT_CONFIG",
    "PRETRAINED_CHECKPOINTS",
    "UMAObservation",
    "build_model",
    "build_inputs",
    "load_model",
    "predict",
    "proprioception_from_pose",
    "resolve_checkpoint",
]

#: The released model configuration (Hydra / OmegaConf YAML).
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "uma.yaml"

#: Released checkpoints: name -> (Hugging Face repo, file).
PRETRAINED_CHECKPOINTS = {
    "uma_pretrain": ("quantumcookie/UMA", "uma_pretrain.pt"),
}

#: Input image size of the vision encoder; images are resized to it.
IMAGE_SIZE = 224

#: Depth beyond this range (metres) is treated as invalid, as in training.
MAX_DEPTH_M = 2.0


@dataclasses.dataclass
class UMAObservation:
    """One observation for the policy (batched).

    Attributes:
        rgb: ``(B, H, W, 3)`` uint8 image of the view (exterior) camera, current step.
        depth: ``(B, H, W)`` depth of the same camera in metres (``0`` / NaN = invalid).
        intrinsics: ``(B, 3, 3)`` camera intrinsics for ``rgb`` / ``depth``.
        extrinsics: ``(B, 3, 4)`` robot-base-to-camera transform.
        proprioception: ``(B, obs_horizon, 10)`` end-effector state history, oldest first
            (see :func:`proprioception_from_pose`).
        tracked_points: ``(B, obs_horizon, M, 5)`` tracked points, oldest first: position in the
            robot-base frame (3), visible (``1`` in view, ``0`` occluded), valid (``1`` tracked,
            ``0`` lost / padding). They need not be the prompt's points: training picked them as
            a random subset of scene points, independently of the prompt. They can be followed by
            a point tracker, or be any sampled scene points that the model then tracks implicitly
            (at some cost in performance).
        wrist_rgb: Optional ``(B, H, W, 3)`` uint8 wrist-camera image, current step.
    """

    rgb: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    proprioception: torch.Tensor
    tracked_points: torch.Tensor
    wrist_rgb: Optional[torch.Tensor] = None

    def to(self, device: Union[str, torch.device]) -> "UMAObservation":
        return dataclasses.replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                for f in dataclasses.fields(self)
                if getattr(self, f.name) is not None
            },
        )


def build_model(config_path: Union[str, Path] = DEFAULT_CONFIG) -> UMAActor:
    """Instantiate the (untrained) UMA actor from its configuration."""
    import hydra
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": OmegaConf.load(config_path)})
    return hydra.utils.instantiate(cfg.model.actor)


def resolve_checkpoint(checkpoint: Union[str, Path] = "uma_pretrain") -> Path:
    """Local path of a checkpoint: a released checkpoint's name (see :data:`PRETRAINED_CHECKPOINTS`) is
    downloaded from Hugging Face on first use and cached; anything else is a local file."""
    if str(checkpoint) in PRETRAINED_CHECKPOINTS:
        from huggingface_hub import hf_hub_download

        repo_id, filename = PRETRAINED_CHECKPOINTS[str(checkpoint)]
        return Path(hf_hub_download(repo_id, filename))
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"no checkpoint file {str(checkpoint)!r}; released checkpoints: {sorted(PRETRAINED_CHECKPOINTS)}"
        )
    return path


def load_model(
    checkpoint: Union[str, Path] = "uma_pretrain",
    device: Union[str, torch.device] = "cuda",
    config_path: Union[str, Path] = DEFAULT_CONFIG,
) -> UMAActor:
    """Build the actor and load a checkpoint (network and, if present, EMA weights).

    Args:
        checkpoint: A released checkpoint's name (default ``"uma_pretrain"``, downloaded from Hugging Face on
            first use and cached by ``huggingface_hub``) or the path of a local ``.pt`` file.
        device: Device to load the model on.
        config_path: Model configuration; the released one by default.
    """
    actor = build_model(config_path).to(device)
    state = torch.load(resolve_checkpoint(checkpoint), map_location=device, weights_only=True)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def proprioception_from_pose(
    position: torch.Tensor, quaternion_wxyz: torch.Tensor, gripper: torch.Tensor
) -> torch.Tensor:
    """``(..., 10)`` proprioception from an EEF position ``(..., 3)``, quaternion ``(..., 4)``
    ``(w, x, y, z)`` and gripper ``(..., 1)`` (``0`` open … ``1`` closed)."""
    q = quaternion_wxyz.float()
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1)
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1)
    return torch.cat([position.float(), row0, row1, gripper.float()], dim=-1)


def _image(rgb: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` uint8 -> ``(B, 3, H, W)`` float in ``[0, 1]``."""
    return (rgb.permute(0, 3, 1, 2).float() / 255.0).contiguous()


def _depth(depth: torch.Tensor) -> torch.Tensor:
    depth = torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0).clone()
    depth[(depth < 0) | (depth > MAX_DEPTH_M)] = 0
    return depth


def build_inputs(
    model: UMAActor,
    observation: UMAObservation,
    motion_prompt: Optional[torch.Tensor] = None,
    *,
    language: Optional[Union[str, Sequence[str]]] = None,
    prompt_rgb: Optional[torch.Tensor] = None,
    prompt_depth: Optional[torch.Tensor] = None,
):
    """Build the model's ``(image context, masked tokens)`` inputs.

    The policy is conditioned on exactly one task description: a motion prompt or a language
    instruction.

    Args:
        model: The actor (its horizons are read from it).
        observation: The current observation.
        motion_prompt: ``(B, T, N, 4)`` desired trajectory of ``N`` points over ``T`` steps in
            the robot-base frame, plus a validity flag (``1`` valid), from where the points were
            when the prompt was given.
        language: Instead of a motion prompt, an instruction (one string for the whole batch, or
            one per batch element), e.g. ``"put the spatula into the utensil holder"``.
        prompt_rgb / prompt_depth: The view camera's image / depth when the prompt was given
            (the episode's first frame); default: the current frame.
    """
    obs = observation
    B = obs.rgb.shape[0]
    device = obs.rgb.device
    if (motion_prompt is None) == (language is None):
        raise ValueError("pass exactly one of motion_prompt and language")
    if language is not None:
        instructions = [language] * B if isinstance(language, str) else list(language)
        if len(instructions) != B:
            raise ValueError(f"{len(instructions)} instructions for a batch of {B}")
        # No motion prompt: a placeholder, masked out entirely.
        motion_prompt = torch.zeros((B, 2, 1, 4), device=device)
    L_ctx = model.obs_horizon
    L = model.obs_horizon + model.pred_horizon + len(model.additional_times)
    if obs.proprioception.shape[1] < L_ctx or obs.tracked_points.shape[1] < L_ctx:
        raise ValueError(f"proprioception and tracked_points need at least {L_ctx} steps of history")

    prompt_rgb = _image(obs.rgb if prompt_rgb is None else prompt_rgb)
    prompt_depth = _depth(obs.depth if prompt_depth is None else prompt_depth)
    intrinsics = obs.intrinsics.float()
    extrinsics = obs.extrinsics.float()
    prompt_positions = motion_prompt[..., :3].float()
    prompt_valid = motion_prompt[..., -1] != 0

    image_ctx = UMAImageContext.create(
        image=UMATokenItem.create(token_data=_image(obs.rgb), sequence_shape=(B,), device=device),
        depth=UMATokenItem.create(token_data=_depth(obs.depth), sequence_shape=(B,), device=device),
        intrinsics=UMATokenItem.create(token_data=intrinsics, sequence_shape=(B,), device=device),
        extrinsics=UMATokenItem.create(token_data=extrinsics, sequence_shape=(B,), device=device),
        gripper_image=None
        if obs.wrist_rgb is None
        else UMATokenItem.create(token_data=_image(obs.wrist_rgb), sequence_shape=(B,), device=device),
        language_instructions=None
        if language is None
        else UMATokenItem.create(
            token_data=np.array(instructions, dtype=object), sequence_shape=(B,), empty=False, device=device
        ),
        flow_spec_context=UMAFlowSpecContextItem.create(
            flow_spec=transform_3d(prompt_positions, extrinsics, is_action=False),
            initial_image=prompt_rgb,
            initial_depth=prompt_depth,
            initial_intrinsics=intrinsics,
            initial_extrinsics=extrinsics,
            valid_mask=prompt_valid,
        ),
    )
    image_ctx = UMAImageContext.resize_image(image_ctx, IMAGE_SIZE, IMAGE_SIZE)

    # Actions: the history is observed, the rest is predicted.
    action_data = torch.cat(
        [
            obs.proprioception[:, -L_ctx:].float(),
            torch.zeros((B, L - L_ctx, model.feature_extractor.action_dim), device=device),
        ],
        dim=1,
    )
    action_pred_mask = torch.ones((B, L), dtype=torch.bool, device=device)
    action_pred_mask[:, :L_ctx] = False
    action_valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

    # Flows: the tracked points' history is observed, their future predicted. Only the current
    # step of the history is used, and only points valid now (normalization is anchored there).
    tracked = obs.tracked_points[:, -L_ctx:].float()
    M = tracked.shape[2]
    flow_data = torch.zeros((B, L, M, model.feature_extractor.flow_dim), device=device)
    flow_data[:, :L_ctx] = tracked[..., :-1]  # position + visible
    flow_pred_mask = torch.cat(
        [
            torch.zeros((B, L_ctx, M), dtype=torch.bool, device=device),
            torch.ones((B, L - L_ctx, M), dtype=torch.bool, device=device),
        ],
        dim=1,
    )
    valid = tracked[..., -1] != 0
    flow_valid_mask = torch.cat(
        [valid, valid.any(dim=1)[:, None, :].expand(-1, L - L_ctx, -1)], dim=1
    )
    flow_valid_mask[:, :L_ctx - 1] = False
    flow_valid_mask = flow_valid_mask & flow_valid_mask[:, L_ctx - 1:L_ctx, :]

    masked_tokens = UMAMaskedTokens.create(
        action=UMATokenItem.create(
            token_data=action_data, pred_mask=action_pred_mask, valid_mask=action_valid_mask,
            sequence_shape=(B, L),
        ),
        flow=UMATokenItem.create(
            token_data=flow_data, pred_mask=flow_pred_mask, valid_mask=flow_valid_mask,
            sequence_shape=(B, L, M),
        ),
    )
    masked_tokens = UMAMaskedTokens.transform_3d(
        masked_tokens, extrinsics, transform_action=False, transform_flow=True
    )
    return image_ctx, masked_tokens


@torch.no_grad()
def predict(
    model: UMAActor,
    observation: UMAObservation,
    motion_prompt: Optional[torch.Tensor] = None,
    *,
    language: Optional[Union[str, Sequence[str]]] = None,
    prompt_rgb: Optional[torch.Tensor] = None,
    prompt_depth: Optional[torch.Tensor] = None,
    use_ema: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Sample an action chunk and the future motion of the tracked points.

    Args:
        model / observation / motion_prompt / language / prompt_rgb / prompt_depth: As
            :func:`build_inputs` (a motion prompt or a language instruction).
        use_ema: Use the exponential-moving-average weights (recommended; the released ones).
        seed: Seed for the flow-matching noise; the same seed gives the same sample.

    Returns:
        ``actions``: ``(B, act_horizon, 10)`` end-effector targets (robot-base frame), to be
        executed in order at 15 Hz;
        ``future_points``: ``(B, pred_horizon + len(additional_times), M, 3)`` predicted
        positions of the tracked points (robot-base frame) at the future steps followed by the
        far-future steps ``model.additional_times``.
    """
    context, tokens = build_inputs(
        model, observation, motion_prompt, language=language, prompt_rgb=prompt_rgb, prompt_depth=prompt_depth
    )
    device = observation.rgb.device
    if seed is None:
        out = model.predict_executable_action(context, tokens, use_ema=use_ema)
    else:
        # Sampling also draws from torch's global generator and from the model's own generator
        # (point-cloud subsampling): seed both, then restore them.
        devices = [device] if device.type == "cuda" else []
        model_rng_state = model.rng.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                model.rng.manual_seed(int(seed))
                rng = torch.Generator(device=device).manual_seed(int(seed))
                out = model.predict_executable_action(context, tokens, use_ema=use_ema, rng=rng)
        finally:
            model.rng.set_state(model_rng_state)
    future_cam = out["future_task_flows"][..., :3]
    future_points = transform_3d(future_cam, observation.extrinsics.float(), is_action=False, inv=True)
    return {"actions": out["executable_action"], "future_points": future_points}
