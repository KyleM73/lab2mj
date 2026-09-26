"""Closed-loop policy rollout of a converted bundle, with optional viewer.

Usage::

    uv run lab2mj-play --bundle DIR --policy policy.pt \\
        --command "0.8,0.0,0.0" --steps 400 [--viewer] [--strict] [--seed N]

Runs the TorchScript policy at the bundle's policy rate and prints velocity
tracking statistics (achieved base velocity vs the commanded ``vx,vy,wz``).
The command is pinned for the whole rollout in and out of ``--strict`` mode.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from lab2mj.env import MjEnv
from lab2mj.validate import base_vel_b


def _parse_command(spec: str) -> np.ndarray:
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--command expects 'vx,vy,wz', got {spec!r}")
    return np.asarray(parts, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Bundle directory from lab2mj.convert.")
    parser.add_argument("--policy", required=True, help="TorchScript policy.pt.")
    parser.add_argument("--command", default="0.8,0.0,0.0", help="Pinned base-frame velocity command 'vx,vy,wz'.")
    parser.add_argument("--steps", type=int, default=400, help="Number of policy steps.")
    parser.add_argument("--viewer", action="store_true", help="Open the passive MuJoCo viewer.")
    parser.add_argument("--strict", action="store_true", help="Deterministic mode: no obs noise, no DR events.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args(argv)

    command = _parse_command(args.command)
    env = MjEnv(args.bundle, policy_path=args.policy, strict=args.strict, seed=args.seed, command=command)
    obs = env.reset()
    print(f"[play] obs dim {env.obs_dim}, action dim {env.action_dim}, policy dt {env.policy_dt:.4f} s")

    viewer_ctx = None
    if args.viewer:
        try:
            import mujoco.viewer as mj_viewer
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("mujoco.viewer is unavailable; run without --viewer") from exc
        viewer_ctx = mj_viewer.launch_passive(env.model, env.data)

    rec: dict[str, list[np.ndarray]] = {"root_pos_w": [], "root_quat_w": [], "root_lin_vel_w": [], "root_ang_vel_w": []}
    valid: list[bool] = []
    resets = 0
    next_sync = time.perf_counter()
    try:
        for _ in range(args.steps):
            action_raw = env.policy_action(obs)
            obs, terminated, truncated, info = env.step(action_raw)
            state = env.record_state()
            for key in rec:
                rec[key].append(state[key])
            # Non-strict auto-reset happens inside step(): this step's recorded state
            # is the respawn pose, not rollout physics — mask it out of the statistics.
            valid.append(not info["did_reset"])
            if terminated or truncated:
                resets += 1
                if env.strict:
                    print(f"[play] termination in strict mode ({info['reasons']}); stopping")
                    break
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                # Pace to real time net of compute; a step that overran resets the
                # deadline instead of accumulating debt.
                next_sync += env.policy_dt
                delay = next_sync - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
                else:
                    next_sync = time.perf_counter()
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    if not rec["root_pos_w"]:
        print("[play] no steps completed")
        return 1
    traj = {key: np.stack(values) for key, values in rec.items()}
    valid_mask = np.asarray(valid, dtype=bool)
    vel_b = base_vel_b(traj["root_quat_w"], traj["root_lin_vel_w"], traj["root_ang_vel_w"])
    # Skip the first half second while the gait settles.
    skip = min(int(round(0.5 / env.policy_dt)), max(vel_b.shape[0] - 1, 0))
    kept = valid_mask[skip:]
    if not kept.any():
        print(f"[play] every post-settle step ended in a reset ({resets} resets); no trajectory statistics")
        return 1
    mean_vel = vel_b[skip:][kept].mean(axis=0)
    excluded = int((~kept).sum())
    last_valid = int(np.nonzero(valid_mask)[0][-1])
    print(f"[play] steps: {vel_b.shape[0]}, episode resets: {resets}, spawn frames excluded from stats: {excluded}")
    print(f"[play] commanded  [vx, vy, wz] = [{command[0]:+.3f}, {command[1]:+.3f}, {command[2]:+.3f}]")
    print(f"[play] achieved   [vx, vy, wz] = [{mean_vel[0]:+.3f}, {mean_vel[1]:+.3f}, {mean_vel[2]:+.3f}]")
    print(f"[play] tracking error |v - cmd| = {np.linalg.norm(mean_vel - command):.3f}")
    print(f"[play] final root z = {traj['root_pos_w'][last_valid, 2]:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
