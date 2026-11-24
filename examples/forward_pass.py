"""Run UMA on one recorded observation and print / plot its prediction.

The sample (``sample_insertion.npz``) is step 150 of a simulated episode of the insertion task
("put the spatula into the utensil holder"): the exterior camera's RGB-D frame, the wrist
camera's image, the last 8 end-effector states, 24 tracked points (16 on the spatula, 8 on the
scene) and the motion prompt: the trajectory of those points over the episode, at 16 steps.

    python examples/forward_pass.py [--plot prediction.png] [--checkpoint uma_pretrain | /path/to/uma.pt]
    python examples/forward_pass.py --language "put the spatula into the utensil holder"

The pretrained checkpoint is downloaded from Hugging Face on first use.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from uma.inference import UMAObservation, load_model, predict, proprioception_from_pose

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="uma_pretrain",
                        help="released checkpoint name (downloaded on first use) or a local .pt file")
    parser.add_argument("--sample", default=str(HERE / "sample_insertion.npz"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--language", default=None,
                        help="condition on this instruction instead of the sample's motion prompt")
    parser.add_argument("--plot", default=None, help="save the prediction drawn on the camera image here")
    args = parser.parse_args()

    data = {k: torch.from_numpy(v) for k, v in np.load(args.sample).items() if v.dtype != np.dtype("<U39")}
    model = load_model(args.checkpoint, device=args.device)

    # Batch of one: add the leading batch dimension to every input.
    observation = UMAObservation(
        rgb=data["rgb"][None],  # (1, H, W, 3) uint8
        depth=data["depth"][None],  # (1, H, W) metres
        intrinsics=data["intrinsics"][None],  # (1, 3, 3)
        extrinsics=data["extrinsics"][None],  # (1, 3, 4) robot base -> camera
        proprioception=proprioception_from_pose(
            data["eef_position"], data["eef_quaternion_wxyz"], data["gripper"]
        )[None],  # (1, 8, 10)
        tracked_points=data["tracked_points"][None],  # (1, 8, 24, 5)
        wrist_rgb=data["wrist_rgb"][None],  # (1, H, W, 3) uint8
    ).to(args.device)
    motion_prompt = data["motion_prompt"][None].to(args.device)  # (1, 16, 24, 4)

    if args.language is None:
        out = predict(
            model,
            observation,
            motion_prompt,
            prompt_rgb=data["prompt_rgb"][None].to(args.device),
            prompt_depth=data["prompt_depth"][None].to(args.device),
            seed=args.seed,
        )
    else:
        out = predict(model, observation, language=args.language, seed=args.seed)
    actions = out["actions"][0].cpu()  # (8, 10): position, 6D rotation, gripper
    points = out["future_points"][0].cpu()  # (19, 24, 3)
    np.set_printoptions(precision=3, suppress=True)
    print(f"current end-effector position: {data['eef_position'][-1].numpy()}, gripper {data['gripper'][-1].item():.2f}")
    print("predicted end-effector positions (m) and gripper, next steps:")
    for i, a in enumerate(actions):
        print(f"  t+{i + 1}: {a[:3].numpy()}  gripper {a[9].item():.2f}")
    moved = (points[model.pred_horizon - 1, :16] - data["tracked_points"][-1, :16, :3]).norm(dim=-1)
    print(f"predicted spatula point displacement over the next {model.pred_horizon} steps: "
          f"mean {moved.mean().item() * 100:.1f} cm")

    if args.plot:
        plot(data, points[: model.pred_horizon], actions, args.plot)
        print(f"wrote {args.plot}")


def plot(data, points, actions, path) -> None:
    """Draw the prompt (white), the predicted point motion over the next steps (colour) and the
    predicted end-effector path (red) on the camera image."""
    import matplotlib.pyplot as plt

    K, E = data["intrinsics"].double(), data["extrinsics"].double()

    def project(xyz):
        cam = xyz.double() @ E[:, :3].T + E[:, 3]
        uv = cam @ K.T
        return (uv[..., :2] / uv[..., 2:]).numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(data["rgb"].numpy())
    prompt = data["motion_prompt"]
    for m in range(prompt.shape[1]):
        if prompt[:, m, 3].all():
            uv = project(prompt[:, m, :3])
            ax.plot(uv[:, 0], uv[:, 1], color="white", lw=0.8, alpha=0.7)
    now = data["tracked_points"][-1, :, :3]
    valid = data["tracked_points"][-1, :, 4] != 0
    for m in range(points.shape[1]):
        if not valid[m]:
            continue
        uv = project(torch.cat([now[m : m + 1], points[:, m]], 0))
        ax.plot(uv[:, 0], uv[:, 1], color=plt.cm.viridis(m / points.shape[1]), lw=1.2)
    uv = project(torch.cat([data["eef_position"][-1:], actions[:, :3]], 0))
    ax.plot(uv[:, 0], uv[:, 1], color="red", lw=2, marker="o", ms=2)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
