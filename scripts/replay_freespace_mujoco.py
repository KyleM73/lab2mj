"""MuJoCo-side free-space plant-dynamics replay against an Isaac reference dump.

Local-runnable counterpart to ``scripts/dump_isaac_freespace.py``: loads a
converted bundle's scene model, strips everything that is not PLANT dynamics
(see ``lab2mj.freespace`` for what that means; additionally
``model.opt.gravity`` is set from the dump), replays the dump's recorded
per-physics-step torques open-loop, and reports the state divergence per phase.

Each Isaac physics step is integrated as ``--substeps`` MuJoCo steps at
``physics_dt / substeps`` with the ctrl held constant across the substeps.
The default is 1 — matching PhysX's discretization, which integrates once per
physics step. Substepping changes the per-step second-order position term
(``(N+1)/(2N) * dt^2 * a`` for N substeps vs PhysX's ``dt^2 * a``), which
accumulates to a pure phase offset of ``(1 - (N+1)/(2N)) * dt * v(t)`` on
every coordinate — measured at exactly ``0.375 * dt`` against a CPU-PhysX
dump with N = 4 — and dominated the early-time divergence. Pass the bundle's
``physics_substeps`` to reproduce the runtime integration instead (contact
resolution needs it; free space does not).

The ``--zero-*`` sensitivity flags deliberately corrupt the plant to
demonstrate the test detects wrong values (such runs are expected to FAIL the
gate where the corrupted parameter is nonzero; see each flag's help).

Outputs under ``--out``: ``report.json`` (all stats + flags + pass/fail),
``replay.npz`` (the MuJoCo trajectories and signed differences), and one
``<phase>_divergence.png`` plot per phase.

Exit code 0 iff ``max |dq| <= 0.01`` rad over the first ``GATE_WINDOW_S``
(1 s) of every replayed phase; the full horizon is reported as informative,
mirroring the open_loop/closed_loop gate design. Tolerance rationale: the
measured healthy floors against CPU-PhysX references are 4.5e-4 (h1),
7.2e-4/2.4e-3 (spot passive/excited — the excited residual is
stiction-coupling breakaway events on the held knees), and 2.6e-3 (g1),
while real plant defects measured 0.5-2.2 rad in-window (cassie's invalid
authored inertia, h1's dropped USD armature) — the 0.01 bar sits ~4x above
the worst healthy floor and 50x+ below any measured defect. NOTE: dumps recorded on the GPU PhysX
pipeline carry a dt-independent angular-momentum drift the CPU pipeline does
not (measured ~35 % |L| over 2 s on Spot, opposite sign on G1, identical at
dt 0.005/0.0025/0.00125; MuJoCo and CPU PhysX both conserve L on the same
states) — against such dumps the late-horizon divergence reflects that GPU
solver artifact, not a conversion error. The dumper therefore defaults to the
CPU pipeline and records the ``device`` key; this replay prints a warning (and
sets ``gpu_reference_warning`` in the report) when fed a cuda-recorded dump.

Usage::

    uv run python scripts/replay_freespace_mujoco.py \\
        --bundle data/mj_bundles/spot_velocity \\
        --dump logs/freespace/spot_freespace.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab2mj.bundle import check_dump_joint_order
from lab2mj.freespace import (
    PhaseResult,
    build_stiction,
    disable_contacts,
    disable_joint_limits,
    divergence_stats,
    load_freespace_model,
    quat_angle_rad,
    replay_phase,
    zero_implicit_pd_damping,
)

# Measured populations over the 1 s gate window (see the module docstring):
# healthy CPU-reference floors 4.5e-4 .. 2.6e-3 across h1/spot/g1; real plant
# defects 0.5-2.2 rad.
DQ_TOLERANCE_RAD = 0.01
# Gate window: like the open_loop/closed_loop gates, the strict bar applies to a
# pre-registered early window and the full horizon is reported as informative —
# the emulated-stiction capture-timing floor (see lab2mj.stiction) grows with the
# horizon even against a CPU-PhysX dump, while a wrong mass/armature/friction/
# gravity value blows through the bar well inside the window.
GATE_WINDOW_S = 1.0
MARKS_S = (0.5, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay an Isaac free-space dump in MuJoCo and report the divergence.")
    p.add_argument("--bundle", type=str, required=True, help="Converted MuJoCo bundle directory.")
    p.add_argument("--dump", type=str, required=True, help="npz produced by dump_isaac_freespace.py.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for report.json / replay.npz / plots (default: '<dump stem>_replay' next to the dump).",
    )
    p.add_argument(
        "--substeps",
        type=int,
        default=1,
        help="MuJoCo steps per Isaac physics step (model timestep = physics_dt / substeps). Default 1 matches "
        "PhysX's one-integration-per-step discretization; N > 1 shifts every coordinate by (1-(N+1)/(2N))*dt*v.",
    )
    p.add_argument(
        "--zero-armature",
        action="store_true",
        help="Sensitivity run: zero dof_armature on the robot joints (expected to fail where armature is nonzero).",
    )
    p.add_argument(
        "--zero-friction",
        action="store_true",
        help="Sensitivity run: zero dof_frictionloss on the robot joints AND disable the static-friction "
        "(stiction) emulation (expected to fail where either matters).",
    )
    p.add_argument(
        "--zero-damping-check",
        action="store_true",
        help="Negative control: KEEP the implicit_pd actuator kd in dof_damping instead of zeroing it "
        "(expected to fail for bundles with implicit_pd groups; a no-op otherwise).",
    )
    return p.parse_args()


def phase_diffs(dump: Any, phase: str, result: PhaseResult) -> dict[str, np.ndarray]:
    """Signed MuJoCo-minus-Isaac differences per physics step."""
    dq = result.joint_pos - dump[f"{phase}_joint_pos"]
    dqd = result.joint_vel - dump[f"{phase}_joint_vel"]
    root_pos_err = np.linalg.norm(result.root_pos_w - dump[f"{phase}_root_pos_w"], axis=-1)
    root_quat_err = quat_angle_rad(result.root_quat_w, dump[f"{phase}_root_quat_w"])
    return {"dq": dq, "dqd": dqd, "root_pos_err_m": root_pos_err, "root_quat_err_rad": root_quat_err}


def save_divergence_plot(out_path: Path, phase: str, diffs: dict[str, np.ndarray], dt: float) -> None:
    t = (np.arange(diffs["dq"].shape[0]) + 1) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].semilogy(t, np.clip(np.abs(diffs["dq"]).max(axis=1), 1e-12, None), color="tab:blue")
    axes[0].axhline(DQ_TOLERANCE_RAD, color="tab:red", linestyle="--", label=f"tolerance {DQ_TOLERANCE_RAD:g} rad")
    axes[0].set_ylabel("max |dq| (rad)")
    axes[0].legend()
    axes[1].semilogy(t, np.clip(np.abs(diffs["dqd"]).max(axis=1), 1e-12, None), color="tab:orange")
    axes[1].set_ylabel("max |dqd| (rad/s)")
    axes[2].semilogy(t, np.clip(diffs["root_pos_err_m"], 1e-12, None), color="tab:green", label="root pos err (m)")
    axes[2].semilogy(
        t, np.clip(diffs["root_quat_err_rad"], 1e-12, None), color="tab:purple", label="root quat err (rad)"
    )
    axes[2].set_ylabel("root error")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        for mark in MARKS_S:
            if mark <= t[-1]:
                ax.axvline(mark, color="gray", alpha=0.3, linewidth=0.8)
    axes[0].set_title(f"Free-space divergence, phase '{phase}' (MuJoCo vs Isaac)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_phase_stats(phase: str, stats: dict[str, Any]) -> None:
    print(f"\n  Phase '{phase}' ({stats['num_steps']} steps, {stats['duration_s']:.2f} s):")
    print(
        f"    gate |dq|  = {stats['gate_window_max_abs_dq_rad']:.3e} rad over the first "
        f"{stats['gate_window_s']:g} s (tolerance {DQ_TOLERANCE_RAD:g})"
    )
    print(f"    max  |dq|  = {stats['max_abs_dq_rad']:.3e} rad   (full horizon, informative)")
    print(f"    mean |dq|  = {stats['mean_abs_dq_rad']:.3e} rad")
    print(f"    max  |dqd| = {stats['max_abs_dqd_rad_s']:.3e} rad/s")
    print(f"    mean |dqd| = {stats['mean_abs_dqd_rad_s']:.3e} rad/s")
    print(f"    root pos err: max {stats['max_root_pos_err_m']:.3e} m, final {stats['final_root_pos_err_m']:.3e} m")
    print(
        f"    root quat err: max {stats['max_root_quat_err_rad']:.3e} rad, "
        f"final {stats['final_root_quat_err_rad']:.3e} rad"
    )
    if stats["at"]:
        header = (
            f"    {'mark':>6}  {'max|dq| rad':>12}  {'max|dqd| rad/s':>14}  {'pos err m':>10}  {'quat err rad':>12}"
        )
        print(header)
        for mark, row in stats["at"].items():
            print(
                f"    {mark:>6}  {row['max_abs_dq_rad']:>12.3e}  {row['max_abs_dqd_rad_s']:>14.3e}  "
                f"{row['root_pos_err_m']:>10.3e}  {row['root_quat_err_rad']:>12.3e}"
            )


def main() -> int:
    args = parse_args()
    dump_path = Path(args.dump).resolve()
    out_dir = Path(args.out) if args.out is not None else dump_path.parent / f"{dump_path.stem}_replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, robot_map, manifest = load_freespace_model(args.bundle)
    dump = np.load(dump_path, allow_pickle=False)

    physics_dt = float(dump["physics_dt"])
    if float(manifest["timing"]["physics_dt"]) != physics_dt:
        print(
            f"[INFO] dump physics_dt {physics_dt} != bundle physics_dt {manifest['timing']['physics_dt']} "
            "(dt-sweep dump); the replay integrates at the dump's dt."
        )
    substeps = int(args.substeps)
    if substeps < 1:
        raise ValueError(f"--substeps must be >= 1, got {substeps}")
    # The bundle model's timestep is authored for the runtime's contact substepping;
    # the free-space replay re-times it to the requested discretization.
    model.opt.timestep = physics_dt / substeps
    check_dump_joint_order(manifest, dump)
    if "joint_limits_disabled" not in dump or not bool(dump["joint_limits_disabled"]):
        raise ValueError("dump was recorded WITH joint limits; the replay disables them and would not match")

    model.opt.gravity[:] = np.asarray(dump["gravity_w"], dtype=np.float64)
    disable_contacts(model)
    disable_joint_limits(model)
    if args.zero_damping_check:
        print("[WARN] --zero-damping-check: implicit_pd kd stays in dof_damping (negative control).")
        zeroed_joints: list[str] = []
    else:
        zeroed_joints = zero_implicit_pd_damping(model, manifest)
        print(f"[INFO] Zeroed implicit_pd dof_damping on {len(zeroed_joints)} joint(s).")
    if args.zero_armature:
        model.dof_armature[robot_map.dof_adr] = 0.0
        print("[WARN] --zero-armature: dof_armature zeroed on the robot joints (sensitivity run).")
    if args.zero_friction:
        model.dof_frictionloss[robot_map.dof_adr] = 0.0
        print("[WARN] --zero-friction: dof_frictionloss zeroed on the robot joints (sensitivity run).")
    # PhysX static-friction capture of stationary joints IS plant behavior in the dump;
    # the per-step friction-bound switch reproduces it (no-op when no joint needs it).
    # The capture window follows the DUMP's physics dt: dt-sweep dumps re-time the
    # integration above, and a manifest-dt window would scale the capture velocity
    # by the dt ratio.
    stiction = None if args.zero_friction else build_stiction(model, robot_map, manifest, physics_dt=physics_dt)
    if stiction is not None and stiction.active:
        print(f"[INFO] Stiction emulation active on {stiction.dof_adr.size} joint(s) (static > dynamic friction).")

    # Model-vs-dump plant cross-check: what PhysX received should equal what the model carries.
    armature_mj = model.dof_armature[robot_map.dof_adr]
    friction_mj = model.dof_frictionloss[robot_map.dof_adr]
    if not args.zero_armature and not np.allclose(armature_mj, dump["armature_physx"], rtol=1e-5, atol=1e-8):
        print(
            f"[WARN] dof_armature {armature_mj} != PhysX armature {np.asarray(dump['armature_physx'])} "
            "(the divergence below will show whether it matters)"
        )
    # PhysX >= 5: a moving joint's Coulomb friction is the DYNAMIC friction effort
    # (friction-props column 1); the static effort (column 0 / friction_physx) is a
    # stationary-joint breakaway threshold with no dof_frictionloss counterpart.
    if "friction_props_physx" in dump:
        friction_ref = np.asarray(dump["friction_props_physx"])[:, 1]
    else:
        friction_ref = np.asarray(dump["friction_physx"])
    if not args.zero_friction and not np.allclose(friction_mj, friction_ref, rtol=1e-5, atol=1e-8):
        print(
            f"[WARN] dof_frictionloss {friction_mj} != PhysX dynamic friction {friction_ref} "
            "(the divergence below will show whether it matters)"
        )

    # GPU-pipeline dumps are not a valid strict reference: GPU PhysX does not conserve
    # angular momentum in free fall (measured ~35 % |L| over 2 s on Spot, dt-independent;
    # MuJoCo and CPU PhysX both conserve it), so the late horizon diverges for reasons
    # that are not conversion errors. The dumper defaults to the CPU pipeline.
    dump_device = str(dump["device"]) if "device" in dump else None
    gpu_reference = dump_device is not None and dump_device.startswith("cuda")
    if gpu_reference:
        print(
            f"[WARN] dump was recorded on the GPU PhysX pipeline ({dump_device}), which does not conserve "
            "angular momentum — a FAIL below reflects the reference's solver drift, not the plant "
            "conversion. Re-dump with --device cpu for a gate-able reference."
        )

    phases = [str(p) for p in dump["phases"]]
    report: dict[str, Any] = {
        "bundle": str(Path(args.bundle).resolve()),
        "dump": str(dump_path),
        "robot": str(dump["robot"]),
        "dump_device": dump_device,
        "gpu_reference_warning": gpu_reference,
        "physics_dt": physics_dt,
        "physics_substeps": substeps,
        "tolerance_max_abs_dq_rad": DQ_TOLERANCE_RAD,
        "flags": {
            "zero_armature": bool(args.zero_armature),
            "zero_friction": bool(args.zero_friction),
            "zero_damping_check": bool(args.zero_damping_check),
        },
        "implicit_pd_damping_zeroed_joints": zeroed_joints,
        "phases": {},
    }
    replay_arrays: dict[str, np.ndarray] = {}
    all_passed = True
    for phase in phases:
        init = {
            "root_pos_w": dump[f"{phase}_init_root_pos_w"],
            "root_quat_w": dump[f"{phase}_init_root_quat_w"],
            "root_link_lin_vel_w": dump[f"{phase}_init_root_link_lin_vel_w"],
            "root_ang_vel_w": dump[f"{phase}_init_root_ang_vel_w"],
            "joint_pos": dump[f"{phase}_init_joint_pos"],
            "joint_vel": dump[f"{phase}_init_joint_vel"],
        }
        result = replay_phase(
            model,
            robot_map,
            manifest,
            init=init,
            tau_isaac=dump[f"{phase}_tau"],
            substeps=substeps,
            stiction=stiction,
        )
        diffs = phase_diffs(dump, phase, result)
        stats = divergence_stats(
            dq=diffs["dq"],
            dqd=diffs["dqd"],
            root_pos_err_m=diffs["root_pos_err_m"],
            root_quat_err_rad=diffs["root_quat_err_rad"],
            dt=physics_dt,
            marks_s=MARKS_S,
        )
        gate_steps = min(int(round(GATE_WINDOW_S / physics_dt)), diffs["dq"].shape[0])
        stats["gate_window_s"] = gate_steps * physics_dt
        stats["gate_window_max_abs_dq_rad"] = float(np.abs(diffs["dq"][:gate_steps]).max())
        passed = stats["gate_window_max_abs_dq_rad"] <= DQ_TOLERANCE_RAD
        stats["passed"] = passed
        all_passed &= passed
        report["phases"][phase] = stats
        print_phase_stats(phase, stats)
        save_divergence_plot(out_dir / f"{phase}_divergence.png", phase, diffs, physics_dt)
        replay_arrays[f"{phase}_mj_joint_pos"] = result.joint_pos
        replay_arrays[f"{phase}_mj_joint_vel"] = result.joint_vel
        replay_arrays[f"{phase}_mj_root_pos_w"] = result.root_pos_w
        replay_arrays[f"{phase}_mj_root_quat_w"] = result.root_quat_w
        replay_arrays[f"{phase}_mj_root_lin_vel_w"] = result.root_lin_vel_w
        replay_arrays[f"{phase}_mj_root_link_lin_vel_w"] = result.root_link_lin_vel_w
        replay_arrays[f"{phase}_mj_root_ang_vel_w"] = result.root_ang_vel_w
        replay_arrays[f"{phase}_dq"] = diffs["dq"]
        replay_arrays[f"{phase}_dqd"] = diffs["dqd"]

    report["passed"] = bool(all_passed)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    np.savez_compressed(out_dir / "replay.npz", **replay_arrays)

    print("\n" + "=" * 72)
    verdict = "PASS" if all_passed else "FAIL"
    worst = max(report["phases"][p]["gate_window_max_abs_dq_rad"] for p in phases)
    print(
        f"FREE-SPACE PLANT PARITY: {verdict}  (worst max |dq| = {worst:.3e} rad over the first "
        f"{GATE_WINDOW_S:g} s, tolerance {DQ_TOLERANCE_RAD:g}; full horizon is informative)"
    )
    print(f"Report: {out_dir / 'report.json'}")
    print("=" * 72)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
