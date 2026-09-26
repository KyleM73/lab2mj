"""Replay a contact-drop dump against a converted bundle's contact model.

Builds robot + default ground plane from the bundle's ``robot.xml`` (which
carries the converted contact options: cone, impratio, solref/solimp defaults),
zeroes the implicit-PD drive damping (drive, not plant — the dump's passive
drops ran with all gains zeroed; the stiff drop rebuilds its drive from the
recorded gains), stamps the exact inertias, and integrates each drop phase
at the bundle's substep scheme. With plant parity already established
by the free-space protocol, the reported divergence isolates the CONTACT model:
first-impact response, rebound, and early settling under canonical impacts —
task-independent ground truth for contact-parameter decisions.

Usage::

    uv run python scripts/replay_contact_mujoco.py --bundle data/mj_bundles/spot_velocity \\
        --dump logs/contact/spot_contact.npz [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from lab2mj import bundle as bundle_io
from lab2mj.env import build_robot_map, stamp_exact_inertia
from lab2mj.events import RobotMap
from lab2mj.freespace import build_stiction, replay_phase, zero_implicit_pd_damping
from lab2mj.terrain import add_to_spec, build_terrain


def load_drop_model(bundle_dir: str | Path) -> tuple[mujoco.MjModel, RobotMap, dict]:
    """Robot + default ground plane with the bundle's contact model and friction stamping."""
    manifest = bundle_io.read_manifest(bundle_dir)
    spec = mujoco.MjSpec.from_file(str(Path(bundle_dir) / bundle_io.ROBOT_XML_NAME))
    # Plane at z=0 with the canonical dump material (friction 1.0); add_to_spec applies
    # the same priority/pair-friction stamping (and scheme) the task scene got.
    add_to_spec(
        spec,
        build_terrain(None),
        robot_self_pair=bool(manifest["robot"].get("friction_self_pair", False)),
    )
    model = spec.compile()
    stamp_exact_inertia(model, mujoco.MjData(model), manifest)
    robot_map = build_robot_map(model, manifest)
    return model, robot_map, manifest


def _replay_stiff(model, robot_map, manifest, dump, init, n_steps, substeps, stiction):
    """Stiff-drop replay: PhysX-style implicit PD holding the default pose.

    The dump's stiff phase writes the authored gains as PhysX DRIVE gains, so the
    replay mirrors the runtime's implicit_pd construction for every joint: the kd
    adds onto the model's plant ``dof_damping`` (integrating implicitly against the
    end-of-substep velocity, like the PhysX drive), and ctrl carries
    ``clip(pd, +-maxforce) + kd*qd`` recomputed each substep (via ``replay_phase``'s
    per-substep ``ctrl_fn``).
    """
    from lab2mj.freespace import isaac_to_mj_perm

    perm = isaac_to_mj_perm(manifest)
    kp = np.asarray(dump["stiff_kp"], dtype=np.float64)
    kd = np.asarray(dump["stiff_kd"], dtype=np.float64)
    maxf = np.asarray(dump["dof_max_forces_physx"], dtype=np.float64)
    qpos_adr, dof_adr = robot_map.qpos_adr, robot_map.dof_adr
    q_des = np.asarray(init["joint_pos"], dtype=np.float64)

    def implicit_pd_ctrl(data: mujoco.MjData) -> None:
        qd = data.qvel[dof_adr]
        tau = kp * (q_des - data.qpos[qpos_adr]) - kd * qd
        np.clip(tau, -maxf, maxf, out=tau)
        data.ctrl[:] = (tau + kd * qd)[perm]

    damping_before = model.dof_damping[dof_adr].copy()
    model.dof_damping[dof_adr] = damping_before + kd
    try:
        return replay_phase(
            model,
            robot_map,
            manifest,
            init=init,
            tau_isaac=np.zeros((n_steps, len(q_des))),
            substeps=substeps,
            stiction=stiction,
            ctrl_fn=implicit_pd_ctrl,
        )
    finally:
        model.dof_damping[dof_adr] = damping_before


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Converted bundle of the same robot.")
    parser.add_argument("--dump", required=True, help="Contact-drop dump npz (dump_isaac_contact.py).")
    parser.add_argument("--out", default=None, help="Optional output dir for report.json.")
    args = parser.parse_args(argv)

    dump = np.load(args.dump)
    model, robot_map, manifest = load_drop_model(args.bundle)
    physics_dt = float(dump["physics_dt"])
    if abs(physics_dt - float(manifest["timing"]["physics_dt"])) > 1e-9:
        print(f"[INFO] dump physics_dt {physics_dt} != bundle {manifest['timing']['physics_dt']}; re-timing.")
    substeps = int(manifest["timing"].get("physics_substeps", 1))
    model.opt.timestep = physics_dt / substeps

    bundle_io.check_dump_joint_order(manifest, dump)
    # The replay pairs the bundle's robot material against the protocol's canonical
    # friction-1.0 plane; a dump recorded on a different ground material would score
    # contact parity at the wrong friction.
    for key in ("ground_static_friction", "ground_dynamic_friction"):
        if key in dump and abs(float(dump[key]) - 1.0) > 1e-9:
            raise ValueError(f"dump {key}={float(dump[key]):g}: the drop replay builds the friction-1.0 plane")

    zero_implicit_pd_damping(model, manifest)
    stiction = build_stiction(model, robot_map, manifest, physics_dt=physics_dt)

    report: dict = {"bundle": str(args.bundle), "dump": str(args.dump), "phases": {}}
    phases = ["drop_lo", "drop_hi"] + (["drop_stiff"] if "drop_stiff_joint_pos" in dump else [])
    for phase in phases:
        # Fresh breakaway state per phase (each phase starts a new drop; the shipped
        # phases init at zero joint velocity so this is currently inert, but the
        # stiction latch must not carry across independent replays).
        stiction.reset()
        n = int(dump[f"{phase}_joint_pos"].shape[0])
        init = {
            key: dump[f"{phase}_init_{key}"]
            for key in ("root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_ang_vel_w")
        }
        init["joint_pos"] = dump[f"{phase}_init_joint_pos"]
        init["joint_vel"] = dump[f"{phase}_init_joint_vel"]
        if phase == "drop_stiff":
            rec = _replay_stiff(model, robot_map, manifest, dump, init, n, substeps, stiction)
        else:
            rec = replay_phase(
                model,
                robot_map,
                manifest,
                init=init,
                tau_isaac=np.zeros((n, len(manifest["robot"]["isaac_joint_order"]))),
                substeps=substeps,
                stiction=stiction,
            )
        dq = np.abs(rec.joint_pos - dump[f"{phase}_joint_pos"])
        dz = np.abs(rec.root_pos_w[:, 2] - dump[f"{phase}_root_pos_w"][:, 2])
        # First impact: the step of peak descent speed — contact decelerates the root
        # immediately after (release heights guarantee a clean free fall before it).
        isaac_vz = dump[f"{phase}_root_lin_vel_w"][:, 2]
        impact = int(np.argmin(isaac_vz))
        marks = {}
        for label, idx in (
            ("impact-1", impact - 1),
            ("impact+5", impact + 5),
            ("impact+20", impact + 20),
            ("end", n - 1),
        ):
            if idx < n:
                marks[label] = {"max_abs_dq": float(dq[idx].max()), "root_dz": float(dz[idx])}
        phase_report = {
            "steps": n,
            "impact_step": impact,
            "max_abs_dq": float(dq.max()),
            "max_root_dz": float(dz.max()),
            "marks": marks,
        }
        report["phases"][phase] = phase_report
        mark_txt = "  ".join(f"{k}: dq {v['max_abs_dq']:.4f} dz {v['root_dz']:.4f}" for k, v in marks.items())
        print(f"{phase}: impact@{impact}  {mark_txt}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
