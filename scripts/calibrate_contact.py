"""Offline contact-parameter calibration against an Isaac reference dump.

Sweeps the contact model of a converted bundle — ``impratio``, the robot
collision geoms' ``solimp``, and a ``solref`` timeconst scale — and scores each
configuration with the open-loop parity-gate error (the pre-registered strict
gate), entirely on CPU from an existing bundle + dump pair. No Isaac Sim.

The parameters are patched on the loaded model between rollouts (no
re-conversion), so a full grid over a measured-terrain bundle runs in minutes.
The result is a report, not a mutation: apply a winner by re-converting with
``--contact_profile`` (shipped profiles) or the explicit ``--contact_solimp``
/ ``--contact_impratio`` / ``--contact_solref`` flags.

Usage::

    uv run python scripts/calibrate_contact.py --bundle data/mj_bundles/anymal_c_rough \\
        --dump data/sim2sim_dumps/anymal_c_rough_fwd_settled_phys.npz [--out report.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lab2mj.bundle import check_dump_joint_order, read_manifest
from lab2mj.env import MjEnv
from lab2mj.validate import (
    GATE_WINDOW_S,
    OPEN_LOOP_HORIZON_S,
    dump_fixed_commands,
    gate_windows,
    rollout_open_loop,
    trajectory_metrics,
)

# (name, solimp 5-tuple or None to keep the authored values). The first two entries
# are the builder's two shipped profiles; the rest probe the space between them.
SOLIMP_CANDIDATES: list[tuple[str, tuple[float, ...] | None]] = [
    ("authored", None),
    ("default", (0.9, 0.95, 0.001, 0.5, 2.0)),
    ("engagement", (0.001, 0.9999, 0.001, 0.5, 2.0)),
    ("mid", (0.5, 0.99, 0.001, 0.5, 2.0)),
    ("firm-wide", (0.8, 0.999, 0.002, 0.5, 2.0)),
]
IMPRATIO_CANDIDATES = (1.0, 2.0, 3.0, 5.0)
SOLREF_SCALE_CANDIDATES = (1.0, 2.0, 4.0)
# Stage 2 crosses the stage-1 top configs with the solver-level axes.
CONE_CANDIDATES = ("authored", "pyramidal", "elliptic")
NOSLIP_CANDIDATES = ("authored", 0, 3)
STAGE2_TOP = 5


def _robot_collision_geoms(model) -> np.ndarray:
    """Robot collision geom ids: the geoms add_to_spec stamped with priority 1."""
    ids = np.nonzero(model.geom_priority == 1)[0]
    if ids.size == 0:
        raise ValueError("no priority-1 robot collision geoms found; was the bundle built by convert?")
    return ids


def _score(env: MjEnv, dump: Any, n_steps: int, gate_steps: int) -> dict[str, float]:
    rec = rollout_open_loop(env, dump, n_steps)
    metrics, _ = trajectory_metrics(rec, dump, n_steps, gate_steps)
    return {
        "gate_dq": metrics["max_joint_err_rad_gate_window"],
        "full_dq": metrics["max_joint_err_rad_full"],
        "gate_dz": metrics["max_root_dz_m_gate_window"],
    }


def calibrate(bundle: str, dump_path: str, horizon_s: float, gate_window_s: float) -> dict[str, Any]:
    npz = np.load(dump_path)
    # Materialize the npz once: NpzFile re-decompresses the whole member on every
    # __getitem__, and the sweep replays the same dump for every configuration.
    dump = {key: npz[key] for key in npz.files}
    manifest = read_manifest(bundle)
    check_dump_joint_order(manifest, dump)
    env = MjEnv(bundle, strict=True, command=dump_fixed_commands(manifest, dump))
    policy_dt = float(dump["physics_dt"]) * float(dump["decimation"])
    n_steps, gate_steps = gate_windows(dump, horizon_s, gate_window_s)

    geom_ids = _robot_collision_geoms(env.model)
    authored_impratio = float(env.model.opt.impratio)
    authored_solimp = env.model.geom_solimp[geom_ids].copy()
    authored_solref = env.model.geom_solref[geom_ids].copy()

    authored_cone = int(env.model.opt.cone)
    authored_noslip = int(env.model.opt.noslip_iterations)
    cone_values = {"pyramidal": int(mujoco.mjtCone.mjCONE_PYRAMIDAL), "elliptic": int(mujoco.mjtCone.mjCONE_ELLIPTIC)}

    def apply_and_score(solimp_name, solimp, impratio, solref_scale, cone, noslip):
        env.model.opt.impratio = impratio
        env.model.geom_solimp[geom_ids] = authored_solimp if solimp is None else np.asarray(solimp)
        env.model.geom_solref[geom_ids] = authored_solref
        env.model.geom_solref[geom_ids, 0] *= solref_scale
        env.model.opt.cone = authored_cone if cone == "authored" else cone_values[cone]
        env.model.opt.noslip_iterations = authored_noslip if noslip == "authored" else int(noslip)
        score = _score(env, dump, n_steps, gate_steps)
        return {
            "solimp": solimp_name,
            "impratio": impratio,
            "solref_scale": solref_scale,
            "cone": cone,
            "noslip": noslip,
            **score,
        }

    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    solimp_of = dict(SOLIMP_CANDIDATES)
    for (solimp_name, solimp), impratio, solref_scale in itertools.product(
        SOLIMP_CANDIDATES, IMPRATIO_CANDIDATES, SOLREF_SCALE_CANDIDATES
    ):
        row = apply_and_score(solimp_name, solimp, impratio, solref_scale, "authored", "authored")
        rows.append(row)
        is_authored = solimp is None and impratio == authored_impratio and solref_scale == 1.0
        if is_authored:
            baseline = {k: row[k] for k in ("gate_dq", "full_dq", "gate_dz")}
        print(
            f"solimp={solimp_name:<11} impratio={impratio:<3g} solref_x={solref_scale:<3g} "
            f"gate|dq| {row['gate_dq']:.4f}  full|dq| {row['full_dq']:.4f}  gate|dz| {row['gate_dz']:.4f}"
            + ("  <- authored" if is_authored else ""),
            flush=True,
        )
    # Stage 2: cross the best stage-1 configs with the friction cone and noslip axes.
    stage1_top = sorted(rows, key=lambda r: r["gate_dq"])[:STAGE2_TOP]
    for top in stage1_top:
        for cone, noslip in itertools.product(CONE_CANDIDATES, NOSLIP_CANDIDATES):
            if cone == "authored" and noslip == "authored":
                continue
            row = apply_and_score(
                top["solimp"], solimp_of[top["solimp"]], top["impratio"], top["solref_scale"], cone, noslip
            )
            rows.append(row)
            print(
                f"solimp={row['solimp']:<11} impratio={row['impratio']:<3g} solref_x={row['solref_scale']:<3g} "
                f"cone={cone:<9} noslip={noslip!s:<8} gate|dq| {row['gate_dq']:.4f}",
                flush=True,
            )
    # Restore the authored contact model on the shared env.
    env.model.opt.impratio = authored_impratio
    env.model.geom_solimp[geom_ids] = authored_solimp
    env.model.geom_solref[geom_ids] = authored_solref
    env.model.opt.cone = authored_cone
    env.model.opt.noslip_iterations = authored_noslip

    rows.sort(key=lambda r: r["gate_dq"])
    return {
        "bundle": str(bundle),
        "dump": str(dump_path),
        "horizon_s": n_steps * policy_dt,
        "gate_window_s": gate_steps * policy_dt,
        "authored_impratio": authored_impratio,
        "baseline": baseline,
        "results": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dump", required=True)
    parser.add_argument(
        "--horizon_s",
        type=float,
        default=OPEN_LOOP_HORIZON_S,
        help=f"Open-loop replay horizon (open_loop gate default {OPEN_LOOP_HORIZON_S} s).",
    )
    parser.add_argument(
        "--gate_window_s",
        type=float,
        default=GATE_WINDOW_S,
        help=f"Scored error window (open_loop gate default {GATE_WINDOW_S} s).",
    )
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)

    report = calibrate(args.bundle, args.dump, args.horizon_s, args.gate_window_s)
    best, base = report["results"][0], report["baseline"]
    print(
        f"\nbest: solimp={best['solimp']} impratio={best['impratio']:g} solref_x={best['solref_scale']:g} "
        f"gate|dq| {best['gate_dq']:.4f}"
        + (
            f"  (authored {base['gate_dq']:.4f}, {100 * (1 - best['gate_dq'] / base['gate_dq']):.0f}% lower)"
            if base
            else ""
        )
    )
    nondefault = (
        best["solimp"] != "authored"
        or best["impratio"] != report["authored_impratio"]
        or best["solref_scale"] != 1.0
        or best.get("cone", "authored") != "authored"
        or best.get("noslip", "authored") != "authored"
    )
    if nondefault:
        solimp = dict(SOLIMP_CANDIDATES).get(best["solimp"])
        flags = [] if solimp is None else [f"--contact_solimp {','.join(f'{v:g}' for v in solimp)}"]
        flags.append(f"--contact_impratio {best['impratio']:g}")
        if best["solref_scale"] != 1.0:
            flags.append(f"--contact_solref <authored_timeconst*{best['solref_scale']:g}>,<dampratio>")
        print(f"apply by re-converting with: {' '.join(flags)}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
