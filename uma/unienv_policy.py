"""UMA as a `UniEnv <https://github.com/UniEnvOrg/UniEnv>`_ policy.

:class:`UMAPolicy` implements ``unienv_interface.policy.Policy``: it keeps the observation history the model
needs, predicts an action chunk, executes it step by step and re-plans when the chunk is used up::

    policy = UMAPolicy(load_model(), batch_size=1, image_size=(224, 224), num_tracked_points=24)
    # load_model() is uma.inference.load_model: the released checkpoint, downloaded on first use
    policy.reset(context, observation)        # context: {"motion_prompt": ...} or {"language": ...}
    while running:
        action = policy.get_action()           # (B, 10) end-effector target
        observation, reward, terminated, truncated, info = env.step(action)
        policy.observe_transition(action, observation, dt)

Observation (``observation_space``, a ``DictSpace``; ``B`` leading dimension when batched):

* ``rgb`` ``(B, H, W, 3)`` uint8 and ``depth`` ``(B, H, W)`` metres: the exterior (view) camera;
* ``intrinsics`` ``(B, 3, 3)`` and ``extrinsics`` ``(B, 3, 4)`` (robot base -> camera) of that camera;
* ``eef_position`` ``(B, 3)``, ``eef_quaternion_wxyz`` ``(B, 4)`` and ``gripper`` ``(B, 1)`` (``0`` open … ``1``
  closed): the end effector in the robot-base frame;
* ``tracked_points`` ``(B, M, 5)``: tracked points (robot-base position, visible, valid); see
  :class:`uma.inference.UMAObservation`;
* ``wrist_rgb`` ``(B, Hw, Ww, 3)`` uint8 (only with ``use_wrist_camera=True``).

Context (``context_space``), fixed for an episode: ``{"motion_prompt": (B, T, N, 4)}`` (robot-base positions plus a
validity flag) or, with ``conditioning="language"``, ``{"language": (B,) strings}``. The episode's first observation
is the frame the prompt refers to.

Action (``action_space``): ``(B, 10)`` absolute end-effector target in the robot-base frame: position (3), rotation as
the first two rows of the rotation matrix (6, orthonormalized), gripper (1, in ``[0, 1]``). The model was trained at
a 15 Hz control rate.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch

from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.policy import Policy
from unienv_interface.space import BatchedSpace, BoxSpace, DictSpace, TextSpace

from uma.inference import UMAObservation, predict, proprioception_from_pose
from uma.policy.masked_policy import UMAActor

__all__ = ["UMAPolicy"]

#: Longest language instruction accepted in the context (characters).
MAX_INSTRUCTION_LENGTH = 256


def _valid_actions(actions: torch.Tensor) -> torch.Tensor:
    """Make predicted actions valid for ``action_space``: the two rotation rows orthonormalized (Gram-Schmidt, as
    the DROID environment does with a 6D rotation) and the gripper clipped to ``[0, 1]``."""
    row0 = torch.nn.functional.normalize(actions[..., 3:6], dim=-1)
    row1 = actions[..., 6:9] - (row0 * actions[..., 6:9]).sum(-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    return torch.cat([actions[..., :3], row0, row1, actions[..., 9:].clamp(0.0, 1.0)], dim=-1)


class UMAPolicy(Policy):
    """UniEnv policy around a UMA model (see the module docstring for the spaces).

    Args:
        model: A loaded model (:func:`uma.inference.load_model`).
        batch_size: Number of environments (``None``: unbatched, no leading batch dimension).
        image_size: ``(H, W)`` of the view camera's ``rgb`` / ``depth``.
        num_tracked_points: ``M``, the number of tracked points in each observation.
        conditioning: ``"motion_prompt"`` or ``"language"``: what the context holds.
        prompt_shape: ``(T, N)`` of the motion prompt.
        use_wrist_camera: Whether observations include ``wrist_rgb``.
        wrist_image_size: ``(H, W)`` of ``wrist_rgb``.
        execute_horizon: Actions executed from each predicted chunk before re-planning (``1`` to the model's
            ``act_horizon``, 8).
        use_ema: Use the exponential-moving-average weights.
        device: Torch device of the model and of the data the policy returns.
    """

    metadata: Dict[str, Any] = {"control_rate_hz": 15.0}

    def __init__(
        self,
        model: UMAActor,
        *,
        batch_size: Optional[int] = 1,
        image_size: Tuple[int, int] = (224, 224),
        num_tracked_points: int = 24,
        conditioning: Literal["motion_prompt", "language"] = "motion_prompt",
        prompt_shape: Tuple[int, int] = (16, 24),
        use_wrist_camera: bool = True,
        wrist_image_size: Tuple[int, int] = (224, 224),
        execute_horizon: Optional[int] = None,
        use_ema: bool = True,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if conditioning not in ("motion_prompt", "language"):
            raise ValueError(f"conditioning must be 'motion_prompt' or 'language', got {conditioning!r}")
        self.model = model
        self.backend = PyTorchComputeBackend
        self.device = torch.device(device) if device is not None else next(
            model.nets["feature_extractor"].parameters()
        ).device
        self.batch_size = batch_size
        self.conditioning = conditioning
        self.use_wrist_camera = use_wrist_camera
        self.use_ema = use_ema
        self.execute_horizon = int(execute_horizon or model.act_horizon)
        if not 1 <= self.execute_horizon <= model.act_horizon:
            raise ValueError(f"execute_horizon must be in [1, {model.act_horizon}], got {execute_horizon}")
        self.history_length = model.obs_horizon
        self.rng = torch.Generator(device="cpu")

        H, W = image_size
        lead = () if batch_size is None else (int(batch_size),)
        box = lambda low, high, shape, dtype=torch.float32: BoxSpace(  # noqa: E731
            self.backend, low, high, dtype=dtype, device=self.device, shape=lead + tuple(shape)
        )
        observation = {
            "rgb": box(0, 255, (H, W, 3), torch.uint8),
            "depth": box(0.0, float("inf"), (H, W)),
            "intrinsics": box(-float("inf"), float("inf"), (3, 3)),
            "extrinsics": box(-float("inf"), float("inf"), (3, 4)),
            "eef_position": box(-float("inf"), float("inf"), (3,)),
            "eef_quaternion_wxyz": box(-1.0, 1.0, (4,)),
            "gripper": box(0.0, 1.0, (1,)),
            "tracked_points": box(-float("inf"), float("inf"), (num_tracked_points, 5)),
        }
        if use_wrist_camera:
            observation["wrist_rgb"] = box(0, 255, (*wrist_image_size, 3), torch.uint8)
        self.observation_space = DictSpace(self.backend, observation, device=self.device)

        if conditioning == "motion_prompt":
            context = {"motion_prompt": box(-float("inf"), float("inf"), (*prompt_shape, 4))}
        else:
            text = TextSpace(self.backend, max_length=MAX_INSTRUCTION_LENGTH, device=self.device)
            context = {"language": text if batch_size is None else BatchedSpace(text, (int(batch_size),))}
        self.context_space = DictSpace(self.backend, context, device=self.device)

        low = torch.tensor([-float("inf")] * 3 + [-1.0] * 6 + [0.0])
        high = torch.tensor([float("inf")] * 3 + [1.0] * 6 + [1.0])
        self.action_space = BoxSpace(
            self.backend,
            low.expand(*lead, 10).to(self.device),
            high.expand(*lead, 10).to(self.device),
            dtype=torch.float32,
            device=self.device,
            shape=lead + (10,),
        )

        self._task: Any = None
        self._prompt_rgb: Optional[torch.Tensor] = None
        self._prompt_depth: Optional[torch.Tensor] = None
        self._history: Optional[Dict[str, torch.Tensor]] = None  # proprioception / tracked points, (B, L, ...)
        self._latest: Optional[Dict[str, torch.Tensor]] = None
        self._chunk: Optional[torch.Tensor] = None  # (B, act_horizon, 10)
        self._chunk_step: Optional[torch.Tensor] = None  # (B,) index of the next action in the chunk
        self.last_prediction: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ helpers

    def _batched(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Tensors on the policy's device with a leading batch dimension (added for an unbatched policy);
        language instructions as a list of strings."""
        out = {}
        for key, value in data.items():
            if key == "language":
                out[key] = [value] if isinstance(value, str) else [str(v) for v in np.asarray(value, dtype=object).reshape(-1)]
            else:
                value = torch.as_tensor(value, device=self.device)
                out[key] = value[None] if self.batch_size is None else value
        return out

    def _frame(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Per-step quantities of an observation: proprioception (B, 10) and tracked points (B, M, 5)."""
        return {
            "proprioception": proprioception_from_pose(obs["eef_position"], obs["eef_quaternion_wxyz"], obs["gripper"]),
            "tracked_points": obs["tracked_points"].float(),
        }

    def _full_mask(self, mask: Optional[torch.Tensor], B: int) -> torch.Tensor:
        if mask is None:
            return torch.ones(B, dtype=torch.bool, device=self.device)
        if self.batch_size is None:
            raise ValueError("mask is only supported by batched policies")
        return torch.as_tensor(mask, dtype=torch.bool, device=self.device).reshape(B)

    # ------------------------------------------------------------------ Policy interface

    def reset(self, context, observation, *, seed: Optional[int] = None, mask=None, **kwargs) -> None:
        if seed is not None:
            self.rng.manual_seed(int(seed))
        obs = self._batched(dict(observation))
        ctx = self._batched(dict(context))
        B = obs["rgb"].shape[0]
        rows = self._full_mask(mask, B)
        frame = self._frame(obs)
        history = {k: v[:, None].expand(-1, self.history_length, *v.shape[1:]).clone() for k, v in frame.items()}

        first = self._history is None
        if first and not rows.all():
            raise ValueError("the first reset must start every environment (mask=None)")
        if first or rows.all():
            self._history = history
            self._prompt_rgb = obs["rgb"].clone()
            self._prompt_depth = obs["depth"].float().clone()
            self._task = ctx["language"] if self.conditioning == "language" else ctx["motion_prompt"].float().clone()
            self._chunk = None
            self._chunk_step = torch.full((B,), self.execute_horizon, dtype=torch.long, device=self.device)
        else:
            for key in self._history:
                self._history[key][rows] = history[key][rows]
            self._prompt_rgb[rows] = obs["rgb"][rows]
            self._prompt_depth[rows] = obs["depth"].float()[rows]
            if self.conditioning == "language":
                self._task = [new if r else old for new, old, r in zip(ctx["language"], self._task, rows.tolist())]
            else:
                self._task[rows] = ctx["motion_prompt"].float()[rows]
            self._chunk_step[rows] = self.execute_horizon  # re-plan these environments
        self._latest = obs

    def observe_transition(self, action_taken, new_observation, dt: float) -> None:
        obs = self._batched(dict(new_observation))
        frame = self._frame(obs)
        for key, value in frame.items():
            self._history[key] = torch.cat([self._history[key][:, 1:], value[:, None]], dim=1)
        self._latest = obs
        self._chunk_step = self._chunk_step + 1

    @torch.no_grad()
    def get_action(self, **kwargs) -> torch.Tensor:
        if self._history is None:
            raise RuntimeError("call reset() before get_action()")
        replan = self._chunk_step >= self.execute_horizon
        if replan.any():
            self._plan(replan)
        B = self._chunk.shape[0]
        action = self._chunk[torch.arange(B, device=self.device), self._chunk_step]
        return action[0] if self.batch_size is None else action

    # ------------------------------------------------------------------ planning

    def _plan(self, rows: torch.Tensor) -> None:
        """Predict a new chunk for the environments in ``rows`` (the model runs on the whole batch)."""
        latest = self._latest
        observation = UMAObservation(
            rgb=latest["rgb"],
            depth=latest["depth"],
            intrinsics=latest["intrinsics"],
            extrinsics=latest["extrinsics"],
            proprioception=self._history["proprioception"],
            tracked_points=self._history["tracked_points"],
            wrist_rgb=latest.get("wrist_rgb") if self.use_wrist_camera else None,
        )
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.rng))
        task = {"language": self._task} if self.conditioning == "language" else {"motion_prompt": self._task}
        out = predict(
            self.model,
            observation,
            **task,
            prompt_rgb=self._prompt_rgb,
            prompt_depth=self._prompt_depth,
            use_ema=self.use_ema,
            seed=seed,
        )
        chunk = _valid_actions(out["actions"].float())
        if self._chunk is None:
            self._chunk = chunk.clone()
        else:
            self._chunk[rows] = chunk[rows]
        self._chunk_step = torch.where(rows, torch.zeros_like(self._chunk_step), self._chunk_step)
        self.last_prediction = out
