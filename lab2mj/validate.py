"""Sim2sim validation gates: converted MuJoCo bundle vs an Isaac reference dump.

Usage::

    uv run lab2mj-validate --bundle DIR --dump NPZ --out DIR \\
        [--gates obs,open_loop,closed_loop,behavior] [--policy PATH]

Gates (thresholds are the pre-registered strict short-horizon bar):

* **obs** (first-observation parity) — initialize MuJoCo from the dump's exact
  post-reset state and compare the first observation elementwise per term
  (tolerance 1e-3; noise is off on both sides since the dump is strict).
  ``base_lin_vel`` / ``base_ang_vel`` terms are reported separately: they can
  carry residual init-velocity frame-conversion error and do not gate.
* **open_loop** (open-loop parity) — replay the dump's recorded raw actions
  over the first 1.0 s; gate on max per-joint ``|dq|`` <= 0.05 rad and root
  ``|dz|`` <= 0.02 m over the first 0.5 s; full divergence curves are plotted
  either way. On nav tasks only the HIGH-level action is replayed — the frozen
  low-level policy runs closed-loop inside the replay, so its own compounding
  dominates the gate metric (measured on anymal_c_nav: ll_action error ~0.12
  in the gate window vs ~2.7 over the full horizon); the informative
  ``max_ll_action_err_*`` metrics separate that from plant/contact mismatch.
* **closed_loop** (closed-loop parity) — run the policy from the dump's init for up
  to 2 s; same thresholds over the first 0.5 s, divergence beyond is reported,
  not gated.
* **behavior** (informative, never gates) — full-dump-length rollout: stability
  (root height above the env origin stays above half its initial value),
  velocity-tracking error vs the pinned command compared to Isaac's, and
  per-term observation statistics (mean/std) on both sides.

Outputs ``report.json`` plus diagnostic plots under ``--out`` and prints a
summary table. Exit code is 0 iff every requested gate other than behavior
passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lab2mj.bundle import check_dump_joint_order, read_manifest
from lab2mj.env import MjEnv, dump_actuator_lags
from lab2mj.env_yaml import class_name
from lab2mj.quat import quat_angle_rad, quat_apply_inverse, yaw_quat

OBS_TOL = 1e-3
GATE_JOINT_TOL_RAD = 0.05
GATE_ROOT_DZ_TOL_M = 0.02
GATE_WINDOW_S = 0.5
OPEN_LOOP_HORIZON_S = 1.0
CLOSED_LOOP_HORIZON_S = 2.0
_VELOCITY_FUNCS = ("base_lin_vel", "base_ang_vel")

# Fixed categorical plot colors; magnitude heatmaps use the one-hue "Blues" colormap.
_C1, _C2, _C3 = "#2a78d6", "#eb6834", "#1baf7a"


def dump_fixed_commands(manifest: dict[str, Any], dump: Any) -> dict[str, np.ndarray]:
    """Per-term pinned commands from the dump's recorded keys.

    Velocity terms pin to the dump's CLI ``command``; pose terms pin to the resolved
    world-frame goal ``command_<term>_goal_w`` (the per-step ``command_<term>`` arrays
    are body-frame and cannot seed the world goal); any other term pins to the first
    row of its recorded ``command_<term>`` array. Terms with no recorded value stay
    unpinned (strict mode pins them to zeros).

    Requires a strict dump: the dumper applies the CLI command only in strict mode,
    so on a ``--no-strict`` dump every replay would chase a command (and noise-free
    observations) Isaac never had.
    """
    if "strict" in dump and not bool(dump["strict"]):
        raise ValueError(
            "the reference dump was recorded with --no-strict (sampled commands, obs noise, "
            "randomized events); the strict replay protocols cannot score against it — "
            "re-dump with --strict"
        )
    fixed: dict[str, np.ndarray] = {}
    for entry in manifest["commands"]:
        name, command_type = entry["name"], entry["type"]
        if command_type == "UniformVelocityCommand":
            fixed[name] = np.asarray(dump["command"], dtype=np.float64)
        elif command_type in ("UniformPose2dCommand", "TerrainBasedPose2dCommand"):
            if f"command_{name}_goal_w" in dump:
                fixed[name] = np.asarray(dump[f"command_{name}_goal_w"], dtype=np.float64)
            elif "pose_command" in dump:
                # Old dump without the resolved goal: the CLI vector is NOT world-frame
                # (Isaac offsets it by env origin and default root z), so the pinned
                # goal is systematically off. Re-dump for a trustworthy comparison.
                print(
                    f"[WARN] dump lacks 'command_{name}_goal_w'; pinning the raw CLI pose_command "
                    "as a world goal (frame mismatch vs Isaac — re-dump the reference)."
                )
                fixed[name] = np.asarray(dump["pose_command"], dtype=np.float64)
        elif f"command_{name}" in dump:
            recorded = np.asarray(dump[f"command_{name}"], dtype=np.float64)
            fixed[name] = recorded[0] if recorded.ndim == 2 else recorded
    return fixed


def reset_env_from_dump(env: MjEnv, dump: Any) -> np.ndarray:
    return env.reset_from_state(
        root_pos_w=dump["init_root_pos_w"],
        root_quat_wxyz=dump["init_root_quat_w"],
        root_link_lin_vel_w=dump["init_root_link_lin_vel_w"],
        root_ang_vel_w=dump["init_root_ang_vel_w"],
        joint_pos_isaac=dump["init_joint_pos"],
        joint_vel_isaac=dump["init_joint_vel"],
        last_action_raw=dump["init_last_action"] if "init_last_action" in dump else None,
        obs0=dump["obs0"] if "obs0" in dump else None,
        episode_step=int(dump["init_episode_step"]) if "init_episode_step" in dump else 0,
        low_level_last_raw=dump["init_low_level_action"] if "init_low_level_action" in dump else None,
        actuator_lags=dump_actuator_lags(dump),
    )


def base_vel_b(root_quat_w: np.ndarray, root_lin_vel_w: np.ndarray, root_ang_vel_w: np.ndarray) -> np.ndarray:
    """Per-step [vx_b, vy_b, wz_b] from world-frame root trajectories (yaw-invariant xy)."""
    out = np.zeros((root_quat_w.shape[0], 3), dtype=np.float64)
    for t in range(root_quat_w.shape[0]):
        quat = root_quat_w[t]
        out[t, :2] = quat_apply_inverse(quat, root_lin_vel_w[t])[:2]
        out[t, 2] = quat_apply_inverse(yaw_quat(quat), root_ang_vel_w[t])[2]
    return out


def rollout_open_loop(env: MjEnv, dump: Any, n_steps: int) -> dict[str, np.ndarray]:
    """Replay the dump's recorded raw actions from its init; public API for the
    protocol scripts (calibrate_contact) as well as the open_loop gate."""
    rec: dict[str, list[np.ndarray]] = {"joint_pos": [], "root_pos_w": [], "root_quat_w": []}
    reset_env_from_dump(env, dump)
    actions = dump["action_raw"]
    for t in range(n_steps):
        if env._pre_trained is not None:
            # Mirror the dump's ll_action sampling point: the buffer BEFORE this
            # policy step's decimation loop fires the low-level policy again.
            rec.setdefault("ll_action", []).append(env._pre_trained._ll_last_raw.astype(np.float64).copy())
        env.step(actions[t])
        state = env.record_state()
        rec["joint_pos"].append(state["joint_pos"])
        rec["root_pos_w"].append(state["root_pos_w"])
        rec["root_quat_w"].append(state["root_quat_w"])
    return {key: np.stack(values) for key, values in rec.items()}


def trajectory_metrics(
    rec: dict[str, np.ndarray], dump: Any, n_steps: int, gate_steps: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dq = np.abs(rec["joint_pos"][:n_steps] - dump["joint_pos"][:n_steps])  # (T, J)
    dz = np.abs(rec["root_pos_w"][:n_steps, 2] - dump["root_pos_w"][:n_steps, 2])
    quat_err = quat_angle_rad(rec["root_quat_w"][:n_steps], dump["root_quat_w"][:n_steps])
    metrics = {
        "max_joint_err_rad_gate_window": float(dq[:gate_steps].max()),
        "max_root_dz_m_gate_window": float(dz[:gate_steps].max()),
        "max_joint_err_rad_full": float(dq.max()),
        "max_root_dz_m_full": float(dz.max()),
        "max_quat_err_rad_full": float(quat_err.max()),
        "gate_window_s": gate_steps * float(dump["physics_dt"]) * float(dump["decimation"]),
        "horizon_s": n_steps * float(dump["physics_dt"]) * float(dump["decimation"]),
    }
    curves = {"dq": dq, "dz": dz, "quat_err": quat_err}
    return metrics, curves


def _plot_trajectory_gate(
    curves: dict[str, np.ndarray], dump: Any, joint_names: list[str], title: str, out_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dq, dz, quat_err = curves["dq"], curves["dz"], curves["quat_err"]
    policy_dt = float(dump["physics_dt"]) * float(dump["decimation"])
    time_s = (np.arange(dq.shape[0]) + 1) * policy_dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    fig.suptitle(title)

    ax = axes[0]
    mesh = ax.pcolormesh(time_s, np.arange(dq.shape[1]), dq.T, cmap="Blues", shading="nearest")
    ax.set_yticks(np.arange(len(joint_names)))
    ax.set_yticklabels(joint_names, fontsize=6)
    ax.set_xlabel("time [s]")
    ax.set_title("per-joint |dq| [rad]", fontsize=10)
    fig.colorbar(mesh, ax=ax, label="|dq| [rad]")

    ax = axes[1]
    ax.plot(time_s, dq.max(axis=1), color=_C1, linewidth=1.8, label="max over joints |dq|")
    ax.axhline(GATE_JOINT_TOL_RAD, color="0.4", linestyle="--", linewidth=1.0, label=f"gate {GATE_JOINT_TOL_RAD} rad")
    ax.axvline(GATE_WINDOW_S, color="0.6", linestyle=":", linewidth=1.0, label="gate window")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|dq| [rad]")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    ax.plot(time_s, dz, color=_C1, linewidth=1.8, label="root |dz|")
    ax.plot(time_s, quat_err, color=_C2, linewidth=1.8, label="root quat angle err [rad]")
    ax.axhline(GATE_ROOT_DZ_TOL_M, color="0.4", linestyle="--", linewidth=1.0, label=f"gate {GATE_ROOT_DZ_TOL_M} m")
    ax.axvline(GATE_WINDOW_S, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def gate_obs_parity(env: MjEnv, dump: Any, layout: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    obs_mj = reset_env_from_dump(env, dump)
    obs_isaac = np.asarray(dump["obs0"], dtype=np.float64)
    if obs_mj.shape != obs_isaac.shape:
        raise ValueError(f"obs dim mismatch: mujoco {obs_mj.shape} vs dump {obs_isaac.shape}")

    terms = []
    gated_errors = []
    for entry in layout:
        sl = slice(entry["start"], entry["stop"])
        err = float(np.abs(obs_mj[sl].astype(np.float64) - obs_isaac[sl]).max())
        informative = class_name(entry["func"]) in _VELOCITY_FUNCS
        terms.append({"name": entry["name"], "max_abs_err": err, "informative": informative})
        if not informative:
            gated_errors.append(err)
    passed = bool(all(err <= OBS_TOL for err in gated_errors))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    names = [t["name"] for t in terms]
    errors = np.maximum([t["max_abs_err"] for t in terms], 1e-12)
    colors = [_C3 if t["informative"] else _C1 for t in terms]
    ax.bar(names, errors, color=colors)
    ax.axhline(OBS_TOL, color="0.4", linestyle="--", linewidth=1.0, label=f"tolerance {OBS_TOL:g}")
    ax.set_yscale("log")
    ax.set_ylabel("max |obs_mj - obs_isaac|")
    ax.set_title("Obs parity at t=0 (green = informative velocity terms)", fontsize=10)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "obs_parity.png", dpi=150)
    plt.close(fig)

    return {"passed": passed, "tolerance": OBS_TOL, "terms": terms}


def gate_windows(dump: Any, horizon_s: float, gate_window_s: float = GATE_WINDOW_S) -> tuple[int, int]:
    """(n_steps, gate_steps) for a horizon, both capped at the dump length; public
    API for the protocol scripts (calibrate_contact) as well as the gates."""
    policy_dt = float(dump["physics_dt"]) * float(dump["decimation"])
    n_steps = min(int(round(horizon_s / policy_dt)), int(dump["action_raw"].shape[0]))
    gate_steps = min(int(round(gate_window_s / policy_dt)), n_steps)
    return n_steps, gate_steps


def _trajectory_gate_result(metrics: dict[str, Any]) -> dict[str, Any]:
    """Gate verdict on the shared pre-registered window thresholds (open/closed-loop gates)."""
    passed = bool(
        metrics["max_joint_err_rad_gate_window"] <= GATE_JOINT_TOL_RAD
        and metrics["max_root_dz_m_gate_window"] <= GATE_ROOT_DZ_TOL_M
    )
    return {
        "passed": passed,
        "joint_tol_rad": GATE_JOINT_TOL_RAD,
        "root_dz_tol_m": GATE_ROOT_DZ_TOL_M,
        "metrics": metrics,
    }


def gate_open_loop(env: MjEnv, dump: Any, joint_names: list[str], out_dir: Path) -> dict[str, Any]:
    n_steps, gate_steps = gate_windows(dump, OPEN_LOOP_HORIZON_S)
    rec = rollout_open_loop(env, dump, n_steps)
    metrics, curves = trajectory_metrics(rec, dump, n_steps, gate_steps)
    if "ll_action" in rec and "ll_action" in dump:
        # Informative low-level parity (nav tasks); see the module docstring's open_loop note.
        ll_err = np.abs(rec["ll_action"][:n_steps] - np.asarray(dump["ll_action"], dtype=np.float64)[:n_steps])
        metrics["max_ll_action_err_gate_window"] = float(ll_err[:gate_steps].max())
        metrics["max_ll_action_err_full"] = float(ll_err.max())
    _plot_trajectory_gate(curves, dump, joint_names, "Open-loop parity (action replay)", out_dir / "open_loop.png")
    return _trajectory_gate_result(metrics)


def closed_loop_rollout(env: MjEnv, dump: Any, gates: list[str]) -> dict[str, np.ndarray]:
    """One policy rollout serving every requested closed-loop gate.

    The closed_loop gate's 2-second window is the prefix of the behavior
    gate's full-horizon rollout: the strict env consumes no RNG (noise off,
    events off, commands and lags pinned), so the trajectories are identical
    and each gate evaluates its own window of a single recording. This also
    removes any cross-gate state coupling a second reset-and-rerun could
    pick up.
    """
    if "behavior" in gates:
        horizon_steps = int(dump["action_raw"].shape[0])
    else:
        horizon_steps = gate_windows(dump, CLOSED_LOOP_HORIZON_S)[0]
    reset_env_from_dump(env, dump)
    return env.run_policy(horizon_steps)


def gate_closed_loop(dump: Any, joint_names: list[str], out_dir: Path, rec: dict[str, np.ndarray]) -> dict[str, Any]:
    n_steps, gate_steps = gate_windows(dump, CLOSED_LOOP_HORIZON_S)
    metrics, curves = trajectory_metrics(rec, dump, n_steps, gate_steps)
    _plot_trajectory_gate(curves, dump, joint_names, "Closed-loop parity (policy rollout)", out_dir / "closed_loop.png")
    return _trajectory_gate_result(metrics)


def gate_behavior(
    dump: Any, layout: list[dict[str, Any]], manifest: dict[str, Any], out_dir: Path, rec: dict[str, np.ndarray]
) -> dict[str, Any]:
    policy_dt = float(dump["physics_dt"]) * float(dump["decimation"])
    n_steps = int(dump["action_raw"].shape[0])

    # Stability is height above the env origin, not absolute world z: measured
    # terrain bundles put the origin at the walked tile's (possibly non-zero) height.
    origin_z = float(dump["env_origin_w"][2]) if "env_origin_w" in dump else 0.0
    init_h = float(dump["init_root_pos_w"][2]) - origin_z
    min_h = float(rec["root_pos_w"][:, 2].min()) - origin_z
    stable = bool(min_h > 0.5 * init_h)

    # The dump's ``command`` key is the dumper's CLI velocity vector; it is only a
    # tracking reference when the task actually pins a velocity command term to it.
    # Pose-command (nav) dumps never applied it — scoring against it would compare
    # both rollouts to a phantom command neither sim was tracking.
    has_vel_command = any(entry["type"] == "UniformVelocityCommand" for entry in manifest["commands"])
    command = np.asarray(dump["command"], dtype=np.float64) if has_vel_command else None
    vel_mj_b = base_vel_b(rec["root_quat_w"], rec["root_lin_vel_w"], rec["root_ang_vel_w"])
    vel_isaac_b = base_vel_b(
        dump["root_quat_w"][:n_steps], dump["root_lin_vel_w"][:n_steps], dump["root_ang_vel_w"][:n_steps]
    )
    err_mj = err_isaac = None
    if command is not None:
        err_mj = float(np.mean(np.linalg.norm(vel_mj_b - command, axis=1)))
        err_isaac = float(np.mean(np.linalg.norm(vel_isaac_b - command, axis=1)))

    obs_stats = []
    obs_isaac = np.asarray(dump["obs"], dtype=np.float64)[:n_steps]
    obs_mj = np.asarray(rec["obs"], dtype=np.float64)
    for entry in layout:
        sl = slice(entry["start"], entry["stop"])
        std_isaac = float(obs_isaac[:, sl].std())
        obs_stats.append(
            {
                "name": entry["name"],
                "mean_mj": float(obs_mj[:, sl].mean()),
                "mean_isaac": float(obs_isaac[:, sl].mean()),
                "std_mj": float(obs_mj[:, sl].std()),
                "std_isaac": std_isaac,
                "std_ratio": float(obs_mj[:, sl].std() / std_isaac) if std_isaac > 1e-9 else None,
            }
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_s = (np.arange(n_steps) + 1) * policy_dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("Behavior (informative)")
    ax = axes[0]
    labels = ("vx_b [m/s]", "vy_b [m/s]", "wz_b [rad/s]")
    for k, (color, label) in enumerate(zip((_C1, _C2, _C3), labels)):
        ax.plot(time_s, vel_mj_b[:, k], color=color, linewidth=1.8, label=f"mujoco {label}")
        ax.plot(time_s, vel_isaac_b[:, k], color=color, linewidth=1.2, linestyle="--", label=f"isaac {label}")
        if command is not None:
            ax.axhline(command[k], color=color, linewidth=0.8, linestyle=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("base velocity")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax = axes[1]
    ax.plot(time_s, rec["root_pos_w"][:, 2], color=_C1, linewidth=1.8, label="mujoco root z")
    ax.plot(time_s, dump["root_pos_w"][:n_steps, 2], color=_C2, linewidth=1.8, linestyle="--", label="isaac root z")
    ax.axhline(origin_z + 0.5 * init_h, color="0.4", linestyle="--", linewidth=1.0, label="fall threshold")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("root z [m]")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_dir / "behavior.png", dpi=150)
    plt.close(fig)

    return {
        "passed": True,  # informative only
        "stable": stable,
        "min_root_height_m": min_h,
        "init_root_height_m": init_h,
        "vel_tracking_err_mj": err_mj,
        "vel_tracking_err_isaac": err_isaac,
        "vel_tracking_err_ratio": (
            err_mj / err_isaac if err_mj is not None and err_isaac is not None and err_isaac > 1e-9 else None
        ),
        "obs_stats": obs_stats,
    }


def run_validation(
    bundle_dir: str | Path,
    dump_path: str | Path,
    out_dir: str | Path,
    gates: list[str],
    policy_path: str | Path | None = None,
) -> tuple[dict[str, Any], bool]:
    """Run the requested gates; returns ``(report, all_gated_passed)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = np.load(dump_path)
    manifest = read_manifest(bundle_dir)

    dump_policy_dt = float(dump["physics_dt"]) * float(dump["decimation"])
    manifest_policy_dt = float(manifest["timing"]["physics_dt"]) * float(manifest["timing"]["decimation"])
    if abs(dump_policy_dt - manifest_policy_dt) > 1e-9:
        raise ValueError(f"policy period mismatch: bundle {manifest_policy_dt} s vs dump {dump_policy_dt} s")
    check_dump_joint_order(manifest, dump)
    # A measured-terrain bundle is only valid for dumps recorded on the same terrain
    # instance: a different seed re-draws env 0's tile assignment, and every
    # height-scan term then honestly diverges (looks like a confusing obs-parity failure).
    if (manifest.get("terrain") or {}).get("source") == "measured-from-dump" and "env_origin_w" in dump:
        bundle_origin = np.asarray(manifest["init"]["env_origin_w"], dtype=np.float64)
        dump_origin = np.asarray(dump["env_origin_w"], dtype=np.float64)
        if not np.allclose(bundle_origin, dump_origin, atol=1e-6):
            raise ValueError(
                f"dump env-0 origin {dump_origin.tolist()} != measured-terrain bundle origin "
                f"{bundle_origin.tolist()}: the dump walked a different terrain instance (different "
                "seed/num_envs?) — re-dump with the bundle's seed or re-convert against this dump"
            )

    needs_policy = any(g in gates for g in ("closed_loop", "behavior"))
    if needs_policy and policy_path is None:
        recorded = str(dump["policy_path"]) if "policy_path" in dump else None
        if recorded and Path(recorded).is_file():
            policy_path = recorded
        else:
            raise FileNotFoundError(
                "the closed_loop/behavior gates need the TorchScript policy; pass --policy (the dump's recorded "
                f"policy_path {recorded!r} is not available locally)"
            )

    fixed_commands = dump_fixed_commands(manifest, dump)
    env = MjEnv(bundle_dir, policy_path=policy_path, strict=True, command=fixed_commands)
    layout = manifest["obs"]["layouts"][manifest["obs"]["policy_group"]]
    joint_names = manifest["robot"]["isaac_joint_order"]

    report: dict[str, Any] = {
        "bundle": str(bundle_dir),
        "dump": str(dump_path),
        "gates_requested": gates,
        "gates": {},
    }
    if "obs" in gates:
        report["gates"]["obs"] = gate_obs_parity(env, dump, layout, out_dir)
    if "open_loop" in gates:
        report["gates"]["open_loop"] = gate_open_loop(env, dump, joint_names, out_dir)
    if "closed_loop" in gates or "behavior" in gates:
        rec = closed_loop_rollout(env, dump, gates)
        if "closed_loop" in gates:
            report["gates"]["closed_loop"] = gate_closed_loop(dump, joint_names, out_dir, rec)
        if "behavior" in gates:
            report["gates"]["behavior"] = gate_behavior(dump, layout, manifest, out_dir, rec)

    all_passed = all(report["gates"][g]["passed"] for g in report["gates"] if g != "behavior")
    report["all_gated_passed"] = all_passed

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    return report, all_passed


def _print_summary(report: dict[str, Any]) -> None:
    print("[validate] gate summary")
    print(f"  {'gate':<12} {'result':<8} key metrics")
    for gate_name, result in report["gates"].items():
        if gate_name == "obs":
            worst = max((t["max_abs_err"] for t in result["terms"] if not t["informative"]), default=0.0)
            detail = f"worst gated term err {worst:.2e} (tol {result['tolerance']:g})"
        elif gate_name in ("open_loop", "closed_loop"):
            m = result["metrics"]
            detail = (
                f"|dq| {m['max_joint_err_rad_gate_window']:.4f} rad (tol {result['joint_tol_rad']}), "
                f"|dz| {m['max_root_dz_m_gate_window']:.4f} m (tol {result['root_dz_tol_m']})"
            )
        else:
            ratio = result["vel_tracking_err_ratio"]
            detail = (
                f"stable={result['stable']} min_height={result['min_root_height_m']:.3f} m, "
                f"vel err mj/isaac={ratio if ratio is None else f'{ratio:.2f}'}"
            )
        status = "INFO" if gate_name == "behavior" else ("PASS" if result["passed"] else "FAIL")
        print(f"  {gate_name:<12} {status:<8} {detail}")
    print(f"  overall: {'PASS' if report['all_gated_passed'] else 'FAIL'} (behavior gate is informative)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Bundle directory from lab2mj.convert.")
    parser.add_argument("--dump", required=True, help="Isaac reference dump npz (dump_isaac_reference.py).")
    parser.add_argument("--out", required=True, help="Output directory for report.json and plots.")
    parser.add_argument(
        "--gates",
        default="obs,open_loop,closed_loop,behavior",
        help="Comma-separated subset of obs,open_loop,closed_loop,behavior.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="TorchScript policy for the closed_loop/behavior gates (default: dump's policy_path).",
    )
    args = parser.parse_args(argv)

    gates = [g.strip().lower() for g in args.gates.split(",") if g.strip()]
    unknown = [g for g in gates if g not in ("obs", "open_loop", "closed_loop", "behavior")]
    if unknown:
        parser.error(f"unknown gates {unknown} (choose from obs,open_loop,closed_loop,behavior)")

    _, all_passed = run_validation(args.bundle, args.dump, args.out, gates, policy_path=args.policy)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
