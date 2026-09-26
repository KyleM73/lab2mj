"""One-step parity: per-step model error at matched states, free of compounding.

For every step ``t`` of a reference dump, reset the MuJoCo env to Isaac's exact
recorded state, apply the recorded action for one policy step, and measure the
fresh per-joint error against the recorded next state. Unlike the open-loop replay gate this
never accumulates, so it separates genuine per-step model mismatch from Lyapunov
growth — and its spike steps localize WHERE the model differs (per joint, per
gait phase). Median one-step error on a well-converted flat bundle is a few
milliradians; a joint spiking 10x the median marks a mechanism worth chasing.

Usage::

    uv run python scripts/one_step_parity.py --bundle data/mj_bundles/h1_flat \\
        --dump data/sim2sim_dumps/h1_fwd_settled_phys.npz [--steps N] [--out report.json]

Caveat: actuator state that cannot be reconstructed per step (LSTM actuator-net
hidden state, mid-episode delay-buffer contents) starts cold at every reset, so
robots with such actuators carry a floor unrelated to the plant/contact model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from lab2mj.bundle import check_dump_joint_order, read_manifest
from lab2mj.env import MjEnv, dump_actuator_lags
from lab2mj.validate import dump_fixed_commands


def one_step_parity(bundle: str, dump_path: str, steps: int | None) -> dict:
    dump = np.load(dump_path)
    manifest = read_manifest(bundle)
    check_dump_joint_order(manifest, dump)
    names = manifest["robot"]["isaac_joint_order"]
    env = MjEnv(bundle, strict=True, command=dump_fixed_commands(manifest, dump))
    n = int(dump["action_raw"].shape[0]) - 1
    if steps is not None:
        n = min(n, steps)

    actuator_lags = dump_actuator_lags(dump)
    init_episode_step = int(dump["init_episode_step"]) if "init_episode_step" in dump else 0
    # Hoist the trajectory arrays: NpzFile.__getitem__ re-decompresses the whole
    # member on every access, which is O(T^2) bytes inside the per-step loop.
    root_pos_w, root_quat_w = dump["root_pos_w"], dump["root_quat_w"]
    root_link_lin_vel_w, root_ang_vel_w = dump["root_link_lin_vel_w"], dump["root_ang_vel_w"]
    joint_pos, joint_vel, action_raw = dump["joint_pos"], dump["joint_vel"], dump["action_raw"]
    ll_action = dump["ll_action"] if "ll_action" in dump else None
    per_step = np.zeros(n)
    per_joint = np.zeros((n, len(names)))
    for t in range(n):
        env.reset_from_state(
            root_pos_w=root_pos_w[t],
            root_quat_wxyz=root_quat_w[t],
            root_link_lin_vel_w=root_link_lin_vel_w[t],
            root_ang_vel_w=root_ang_vel_w[t],
            joint_pos_isaac=joint_pos[t],
            joint_vel_isaac=joint_vel[t],
            last_action_raw=action_raw[t],
            # Isaac's episode clock at the restored state (post-step-t) — a zero
            # would blank the low-level policy's 'actions' obs slot at every
            # scored step, an artifact Isaac only has at true t=0.
            episode_step=init_episode_step + t + 1,
            # The low-level action buffer BEFORE step t+1's decimation loop.
            low_level_last_raw=None if ll_action is None else ll_action[t + 1],
            actuator_lags=actuator_lags,
        )
        env.step(action_raw[t + 1])
        err = np.abs(env.data.qpos[env.robot_map.qpos_adr] - joint_pos[t + 1])
        per_joint[t] = err
        per_step[t] = float(err.max())

    median = float(np.median(per_step))
    spikes = [
        {"step": int(t), "max_abs_dq": float(per_step[t]), "joint": names[int(per_joint[t].argmax())]}
        for t in np.argsort(per_step)[::-1][:5]
    ]
    joint_medians = np.median(per_joint, axis=0)
    worst_joints = [
        {"joint": names[j], "median_abs_dq": float(joint_medians[j])} for j in np.argsort(joint_medians)[::-1][:5]
    ]
    return {
        "bundle": str(bundle),
        "dump": str(dump_path),
        "steps": n,
        "median_max_abs_dq": median,
        "p90_max_abs_dq": float(np.percentile(per_step, 90)),
        "max_abs_dq": float(per_step.max()),
        "top_spikes": spikes,
        "worst_joints_by_median": worst_joints,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument("--steps", type=int, default=None, help="Steps to analyze (default: the whole dump).")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    report = one_step_parity(args.bundle, args.dump, args.steps)
    print(
        f"one-step |dq| over {report['steps']} steps: median {report['median_max_abs_dq']:.5f}, "
        f"p90 {report['p90_max_abs_dq']:.5f}, max {report['max_abs_dq']:.5f} rad"
    )
    for spike in report["top_spikes"]:
        print(f"  spike step {spike['step']:>3}: {spike['max_abs_dq']:.5f} rad on {spike['joint']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
