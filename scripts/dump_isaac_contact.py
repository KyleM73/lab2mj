"""Contact-parity reference dump: drops onto a ground plane.

Spawns one robot articulation with ALL drive gains zeroed above a default
ground plane (friction 1.0, restitution 0) and records the physics-rate state
through a short drop — first impact, rebound, and early settling. With
the plant already verified by the free-space protocol, any divergence in the
replayed drop is the CONTACT model alone, measured through canonical impacts
instead of task trajectories. Three phases:

* ``drop_lo`` — root released ``DROP_LO_M`` above the default init height
  (touchdown-speed impacts, ~0.6 m/s),
* ``drop_hi`` — released ``DROP_HI_M`` above (hard impacts, ~1.4 m/s),
* ``drop_stiff`` — released ``DROP_HI_M`` above with the AUTHORED actuator
  gains holding the default pose: passive falls fold the legs into their
  joint limits before touchdown (limit-model divergence contaminates the
  contact signal), while the stiff drop impacts from a matched configuration
  and isolates the contact response under a realistic stance load.

Replay with ``scripts/replay_contact_mujoco.py``. npz schema mirrors the
free-space dump (per-phase ``<phase>_*`` state arrays at the physics rate plus
the same plant records), plus the stiff-drop gains (``stiff_kp`` / ``stiff_kd``),
the PhysX drive force limits, and the ground friction. The horizon is
deliberately short (chaotic limb flailing after the first impacts carries no
contact information).

Usage::

    uv run python scripts/dump_isaac_contact.py --robot spot \\
        --out logs/contact/spot_contact.npz
"""

# Isaac Sim must launch before any isaaclab import (see CLAUDE.md: Import Ordering).
import argparse

from isaac_dump_common import ROBOT_CHOICES, launch_app

parser = argparse.ArgumentParser(description="Dump passive contact-drop references for sim2sim contact parity.")
parser.add_argument(
    "--robot",
    type=str,
    required=True,
    choices=ROBOT_CHOICES,
    help="Robot articulation to dump (same registry as dump_isaac_freespace).",
)
parser.add_argument("--physics_dt", type=float, default=0.005, help="Physics step [s].")
parser.add_argument("--num_steps", type=int, default=120, help="Physics steps per phase (0.6 s at 5 ms).")
parser.add_argument("--seed", type=int, default=42, help="Torch seed (inert: the protocol is deterministic).")
parser.add_argument("--out", type=str, required=True, help="Output npz path.")
args_cli, simulation_app = launch_app(parser)

import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaac_dump_common import (
    append_state,
    load_robot_cfg,
    plant_record,
    reset_to_default_state,
    shutdown,
    state_recorder,
    to_numpy,
    written_init_state,
    zero_gain_implicit_actuators,
)
from isaaclab.assets import Articulation

DROP_LO_M = 0.02
DROP_HI_M = 0.10


def _implicit_cfg(base_cfg):
    """Replace every actuator group with a zero-gain ImplicitActuator.

    The implicit pipeline just forwards targets to the PhysX drive, so a single
    articulation serves both phases: passive drops keep the drive at zero gains,
    the stiff drop writes the hold gains. Explicit models (actuator nets, delayed
    PD) must not exist here at all — they compute torques from position targets
    regardless of their gains and would stack onto the drive. The swap carries
    over each group's armature/friction fields, so the drops run on the exact
    plant the bundle replay reconstructs (the free-space dump's carryover
    invariant); only the drive differs from training.
    """
    cfg = base_cfg.replace(prim_path="/World/Robot")
    return cfg.replace(actuators=zero_gain_implicit_actuators(cfg.actuators, unlimited=False))


# Canonical hold gains for joints whose actuator cfg authors no PD gains (actuator-net
# robots: the net replaces the PD law, so stiffness/damping are None). Any positive
# hold works — the recorded stiff_kp/stiff_kd keys make the replay exactly symmetric —
# but without one the "stiff" phase silently degrades to a passive collapse.
_STIFF_FALLBACK_KP = 60.0
_STIFF_FALLBACK_KD = 2.0


def _stiff_gains(base_cfg, robot) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint (stiffness, damping) from the authored cfg, with a canonical fallback."""
    kp = torch.zeros(1, robot.num_joints, device=robot.device)
    kd = torch.zeros(1, robot.num_joints, device=robot.device)
    for group in base_cfg.actuators.values():
        ids, names = robot.find_joints(group.joint_names_expr)
        for value, buf in ((group.stiffness, kp), (group.damping, kd)):
            if value is None:
                continue
            if isinstance(value, dict):
                import re

                for pattern, v in value.items():
                    for i, n in zip(ids, names):
                        if re.fullmatch(pattern, n):
                            buf[0, i] = float(v)
            else:
                buf[0, list(ids)] = float(value)
    unheld = kp[0] <= 0.0
    if bool(unheld.any()):
        kp[0, unheld] = _STIFF_FALLBACK_KP
        kd[0, unheld] = _STIFF_FALLBACK_KD
    return kp, kd


def _run_drop(sim, robot, drop_m: float, num_steps: int, hold: bool = False) -> dict[str, np.ndarray]:
    dt = float(sim.get_physics_dt())
    init_z = float(robot.cfg.init_state.pos[2]) + drop_m
    joint_pos = reset_to_default_state(robot, root_pos=(0.0, 0.0, init_z))

    rec = state_recorder()
    zero_tau = torch.zeros(1, robot.num_joints, device=robot.device)
    for _ in range(num_steps):
        if hold:
            robot.set_joint_position_target(robot.data.default_joint_pos)
        else:
            robot.set_joint_effort_target(zero_tau)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        append_state(rec, robot.data, np.float64)
    arrays = {key: np.stack(values) for key, values in rec.items()}
    arrays.update(written_init_state((0.0, 0.0, init_z), joint_pos, np.zeros(robot.num_joints)))
    arrays["drop_m"] = np.array(drop_m)
    return arrays


def main() -> None:
    torch.manual_seed(args_cli.seed)
    robot_cfg = _implicit_cfg(load_robot_cfg(args_cli.robot))

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=args_cli.device or "cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
    )
    ground.func("/World/Ground", ground)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(robot_cfg)
    sim.reset()

    arrays: dict[str, np.ndarray] = {
        "robot": np.array(args_cli.robot),
        "seed": np.array(args_cli.seed),
        # GPU pipeline on purpose: the contact model under test is the one training ran on.
        "device": np.array(str(sim_cfg.device)),
        "num_steps": np.array(args_cli.num_steps),
        "physics_dt": np.array(float(sim.get_physics_dt())),
        "gravity_w": np.array(sim_cfg.gravity, dtype=np.float64),
        **plant_record(robot, np.float64),
        "ground_static_friction": np.array(1.0),
        "ground_dynamic_friction": np.array(1.0),
        # PhysX drive force limits: the stiff-drop drive is clamped by these.
        "dof_max_forces_physx": to_numpy(robot.root_physx_view.get_dof_max_forces()[0], np.float64),
    }
    for phase, drop in (("drop_lo", DROP_LO_M), ("drop_hi", DROP_HI_M)):
        rec = _run_drop(sim, robot, drop, args_cli.num_steps)
        arrays.update({f"{phase}_{key}": value for key, value in rec.items()})
        print(f"[contact] {phase}: {args_cli.num_steps} steps from +{drop} m")

    # Stiff drop: restore the authored gains so the drive holds the default pose.
    base_cfg = load_robot_cfg(args_cli.robot)
    kp, kd = _stiff_gains(base_cfg, robot)
    robot.write_joint_stiffness_to_sim(kp)
    robot.write_joint_damping_to_sim(kd)
    arrays["stiff_kp"] = to_numpy(kp[0], np.float64)
    arrays["stiff_kd"] = to_numpy(kd[0], np.float64)
    rec = _run_drop(sim, robot, DROP_HI_M, args_cli.num_steps, hold=True)
    arrays.update({f"drop_stiff_{key}": value for key, value in rec.items()})
    print(f"[contact] drop_stiff: {args_cli.num_steps} steps from +{DROP_HI_M} m (authored gains)")

    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args_cli.out, **arrays)
    print(f"[contact] dumped {args_cli.robot} contact drops to {args_cli.out}")


if __name__ == "__main__":
    main()
    shutdown(simulation_app)
