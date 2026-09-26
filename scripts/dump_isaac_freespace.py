"""Isaac-side free-space plant-dynamics reference dumper for sim2sim validation.

Contrived no-contact experiment that isolates the PLANT dynamics — mass,
inertia, armature, joint friction, gravity, Coriolis — from everything RL:
one robot articulation is spawned at the world origin with NO terrain (it
free-falls), every actuator group is replaced by a zero-gain implicit
actuator so ``set_joint_effort_target`` is a pure torque input, and the sim
is stepped manually at the physics rate. The recorded npz is replayed by
``scripts/replay_freespace_mujoco.py`` against a converted MuJoCo
bundle of the same robot.

Invariants that make the comparison meaningful:

* The actuator replacement preserves exactly what training writes to PhysX
  for the plant: each original group's ``armature`` / ``friction`` (and
  ``dynamic_friction`` / ``viscous_friction``) fields are carried over
  verbatim (None keeps the USD-authored value — the same resolution rule
  ``ActuatorBase._parse_joint_parameter`` applies in training), while
  ``stiffness = damping = 0`` and ``effort_limit_sim = velocity_limit_sim =
  1e9`` remove every drive/clamp contribution. The armature and friction
  values PhysX actually received are read back from the physx view and
  stored in the npz.
* Self-collisions are disabled (the no-contact invariant; the rest of the
  authored ``articulation_props``, including the solver iteration counts,
  is kept).
* Joint position limits are widened to ±2π on the PhysX side — its maximum
  for revolute joints; unlimited joints keep their wider sentinel, and a
  refused write raises via a read-back check (the phase excursions exceed
  the authored ranges, and joint limits are constraint forces, not plant
  dynamics; the MuJoCo replay disables its limits to match). The authored
  limits are recorded under ``joint_pos_limits_orig``.

Protocol per phase (after resetting to the cfg's default joint state at
root pose (0, 0, 0), identity quaternion, zero root velocity):

* ``passive``: initial joint velocities ``qd0[j] = (-1)^j * (0.5 + 0.17 *
  (j % 7))`` rad/s (Isaac joint order), zero effort target every step. This
  phase owns the stiction-capture semantics: joints decay into PhysX's
  static-friction capture, single events the runtime emulation reproduces to
  within its documented capture-timing floor.
* ``excited``: zero initial joint velocity; per-physics-step effort
  ``tau[j](t_k) = A[j] * sin(2*pi*f[j]*t_k + phi[j])`` with ``t_k = k * dt``
  (k = physics-step index; the torque is held over step k), ``f[j] = 0.5 +
  0.35*(j % 8)`` Hz, ``phi[j] = 0.7*j`` rad, and

  ``A[j] = min(0.25 * min(effort_limit_authored[j], 60),
  excite_accel * M[j])`` N*m,

  EXCEPT joints whose amplitude cannot clear their breakaway effort
  (``A[j] < 2 * static_friction_physx[j]``): those are HELD instead
  (``A[j] = 0``, recorded in ``excited_held``). Joint stiction is a
  discontinuous per-event emulation costing O(one physics step) of friction
  impulse per capture or breakaway; a sub-breakaway sinusoid from rest
  produces ONLY such events (measured 0.075 rad/s per knee event on spot —
  0.25 rad of pure timing noise at 1 s), while an undriven resting joint is
  held bit-exactly by both engines as long as Coriolis coupling stays below
  breakaway (the dumper warns if a held joint moves or a driven joint
  stops). Held joints still load the chain, and their plant is probed by
  the passive phase.

  ``effort_limit_authored`` is resolved from the ORIGINAL actuator groups
  (``effort_limit``, falling back to ``effort_limit_sim``, else +inf).
  ``M[j]`` is the joint's reflected inertia, measured in-sim by a one-step
  torque probe (1 N*m on joint j alone from rest; ``M[j] = tau * dt /
  qd[j]``), so ``A[j]`` commands a peak joint acceleration of at most
  ``--excite_accel`` rad/s^2. The inertia cap is required for the test to be
  well-posed: an undamped free-floating chain under sustained sinusoidal
  torques sized from effort limits alone accumulates energy without bound
  (measured in MuJoCo: >1e5 rad/s joint speeds within 2 s on Spot, and
  low-inertia hand joints explode immediately on G1). The probe results and
  the final amplitudes are recorded, and the replay consumes the recorded
  torques verbatim.

Stepping (per physics step): ``robot.set_joint_effort_target(tau)`` ->
``robot.write_data_to_sim()`` -> ``sim.step(render=False)`` ->
``robot.update(dt)``; the state recorded at index ``t`` is the state after
physics step ``t``. The initial state is stored separately per phase under
``<phase>_init_*``.

npz schema (J = joints, B = bodies, T = num_steps; quaternions wxyz; joint
arrays in Isaac/PhysX breadth-first order; ``root_lin_vel_w`` is the root
CoM velocity, ``root_link_lin_vel_w`` the root link-origin velocity — same
conventions as ``dump_isaac_reference.py``). Phase-specific keys are ALWAYS
prefixed with the phase name; the ``phases`` key lists the recorded phases.

===============================  ========  =======================================
key                              shape     description
===============================  ========  =======================================
robot                            ()        registry name (a ROBOT_CHOICES entry)
seed                             ()        seed argument (protocol is deterministic)
device                           ()        PhysX pipeline the dump ran on; defaults to
                                           "cpu" — the GPU pipeline does not conserve
                                           angular momentum and poisons the reference
num_steps                        ()        physics steps per phase
physics_dt                       ()        simulation dt [s]
gravity_w                        (3,)      gravity vector, world frame
phases                           (P,)      recorded phase names
joint_names                      (J,)      Isaac-order joint names
body_names                       (B,)      body names
default_joint_pos                (J,)      articulation default joint positions
default_joint_vel                (J,)      articulation default joint velocities
body_masses                      (B,)      PhysX-resolved body masses
body_coms_physx                  (B, 7)    PhysX-resolved CoM pose in the link frame
                                           (x, y, z, qx, qy, qz, qw — principal axes)
body_inertias_physx              (B, 9)    PhysX-resolved inertia about the CoM,
                                           expressed in the link frame (3x3 rows)
body_link_pos_w                  (B, 3)    link positions at the default reset state
body_link_quat_w                 (B, 4)    link quaternions (wxyz) at the default reset state
armature_physx                   (J,)      armature as read back from PhysX
friction_physx                   (J,)      static joint friction read back from PhysX
friction_props_physx             (J, 3)    static/dynamic/viscous friction read back
                                           (Isaac Sim >= 5; zeros-padded below)
effort_limit_physx               (J,)      solver effort limit read back (1e9 expected)
velocity_limit_physx             (J,)      solver velocity limit read back (1e9 expected)
effort_limit_authored            (J,)      per-joint authored effort limit (inf allowed)
joint_reflected_inertia          (J,)      one-step-probe reflected inertia M[j] [kg m^2]
inertia_probe_tau                ()        probe torque used to measure M [N*m]
excite_accel                     ()        peak-acceleration cap for A[j] [rad/s^2]
joint_pos_limits_orig            (J, 2)    authored joint limits before widening
joint_limits_disabled            ()        True (limits at least +-2pi wide)
self_collisions_disabled         ()        True
<phase>_init_root_pos_w          (3,)      written initial root link position
<phase>_init_root_quat_w         (4,)      written initial root quaternion (wxyz)
<phase>_init_root_lin_vel_w      (3,)      written initial root CoM linear velocity
<phase>_init_root_link_lin_vel_w (3,)      written initial root link-origin velocity
<phase>_init_root_ang_vel_w      (3,)      written initial root angular velocity
<phase>_init_joint_pos           (J,)      written initial joint positions
<phase>_init_joint_vel           (J,)      written initial joint velocities
<phase>_tau                      (T, J)    commanded effort target per physics step
<phase>_joint_pos                (T, J)    joint positions after step t
<phase>_joint_vel                (T, J)    joint velocities after step t
<phase>_applied_torque           (T, J)    applied torque readback after step t
<phase>_root_pos_w               (T, 3)    root link position after step t
<phase>_root_quat_w              (T, 4)    root link quaternion (wxyz) after step t
<phase>_root_lin_vel_w           (T, 3)    root CoM linear velocity after step t
<phase>_root_link_lin_vel_w      (T, 3)    root link-origin linear velocity after step t
<phase>_root_ang_vel_w           (T, 3)    root angular velocity after step t
passive_qd0                      (J,)      the deterministic passive qd0 pattern
excited_held                     (J,)      bool: joint held (A[j] = 0, sub-breakaway)
excited_amp                      (J,)      A[j] torque amplitudes [N*m]
excited_freq_hz                  (J,)      f[j] excitation frequencies [Hz]
excited_phase_rad                (J,)      phi[j] excitation phases [rad]
===============================  ========  =======================================

Example commands::

    uv run python scripts/dump_isaac_freespace.py --robot spot --phase both \\
        --out logs/freespace/spot_freespace.npz

    uv run python scripts/dump_isaac_freespace.py --robot g1 --phase both \\
        --out logs/freespace/g1_freespace.npz
"""

# Isaac Sim must launch before any isaaclab import (see CLAUDE.md: Import Ordering).
import argparse
import sys

from isaac_dump_common import ROBOT_CHOICES, launch_app

PHASE_CHOICES = ("passive", "excited", "both")

# CPU PhysX unless --device is passed explicitly (AppLauncher's own --device default
# is cuda:0, so the override must happen before parsing): the GPU pipeline does not
# conserve angular momentum in free fall (measured ~35 % |L| drift over 2 s on Spot,
# dt-independent), which poisons a plant-parity reference — MuJoCo conserves L, so
# the late horizon diverges for reasons that are not conversion errors. The replay
# warns on cuda-recorded dumps.
if not any(arg == "--device" or arg.startswith("--device=") for arg in sys.argv[1:]):
    sys.argv += ["--device", "cpu"]

parser = argparse.ArgumentParser(description="Dump a free-space plant-dynamics reference for sim2sim validation.")
parser.add_argument("--robot", type=str, required=True, choices=ROBOT_CHOICES, help="Robot articulation to dump.")
parser.add_argument(
    "--phase",
    type=str,
    default="both",
    choices=PHASE_CHOICES,
    help="Which excitation phase(s) to record (npz keys are always phase-prefixed).",
)
parser.add_argument("--num_steps", type=int, default=400, help="Physics steps per phase (400 = 2 s at dt 0.005).")
parser.add_argument(
    "--physics_dt",
    type=float,
    default=0.005,
    help="Physics timestep [s]; non-default values are for integrator-truncation sweeps.",
)
parser.add_argument("--seed", type=int, default=42, help="Recorded seed (the protocol itself is deterministic).")
parser.add_argument(
    "--excite_accel",
    type=float,
    default=5.0,
    help="Peak joint acceleration [rad/s^2] the excited-phase torque amplitudes command: "
    "A[j] = min(0.25 * min(effort_limit_authored[j], 60), excite_accel * reflected_inertia[j]).",
)
parser.add_argument("--out", type=str, required=True, help="Output npz path.")
args_cli, simulation_app = launch_app(parser)

import os
from typing import Any

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
from isaaclab.utils.string import resolve_matching_names_values

# PhysX rejects revolute limit angles outside [-2pi, 2pi] (setLimitParams error,
# measured on anymal_c: a wider write was silently refused and the authored limits
# stayed active). 2pi is ample: phase excursions measure <= ~4 rad.
WIDE_JOINT_LIMIT_RAD = 2.0 * 3.141592653589793
AMP_EFFORT_CAP = 60.0
STICTION_HOLD_FACTOR = 2.0
INERTIA_PROBE_TAU = 1.0


def _freespace_cfg(base_cfg: Any) -> Any:
    """Free-space articulation cfg: raw-torque actuators, origin spawn, no self-collisions.

    Each actuator group becomes a zero-gain ``ImplicitActuatorCfg`` with
    effectively unlimited solver effort/velocity clamps, so
    ``set_joint_effort_target`` is a pure torque input; the plant fields are
    carried over verbatim (see ``zero_gain_implicit_actuators``).
    """
    cfg = base_cfg.copy()
    cfg.prim_path = "/World/Robot"
    cfg.init_state.pos = (0.0, 0.0, 0.0)
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.init_state.lin_vel = (0.0, 0.0, 0.0)
    cfg.init_state.ang_vel = (0.0, 0.0, 0.0)
    # No-contact invariant: nothing else is spawned, so self-collision is the only
    # possible contact source; the solver settings in articulation_props stay authored.
    if cfg.spawn.articulation_props is None:
        cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)
    else:
        cfg.spawn.articulation_props.enabled_self_collisions = False
    cfg.actuators = zero_gain_implicit_actuators(cfg.actuators, unlimited=True)
    return cfg


def _resolve_authored_effort_limits(robot: Articulation, actuators_cfg: dict[str, Any]) -> np.ndarray:
    """Per-joint authored effort limit from the ORIGINAL actuator groups (Isaac order).

    Resolution per group: ``effort_limit`` if set, else ``effort_limit_sim``,
    else +inf. Regex-dict values resolve with the same
    ``resolve_matching_names_values`` semantics as
    ``ActuatorBase._parse_joint_parameter``.
    """
    out = np.full(robot.num_joints, np.inf, dtype=np.float64)
    for group_cfg in actuators_cfg.values():
        ids, names = robot.find_joints(group_cfg.joint_names_expr)
        limit = group_cfg.effort_limit if group_cfg.effort_limit is not None else group_cfg.effort_limit_sim
        if limit is None:
            continue
        if isinstance(limit, dict):
            local_ids, _, values = resolve_matching_names_values(limit, names)
            for local_idx, value in zip(local_ids, values):
                out[ids[local_idx]] = float(value)
        else:
            out[np.asarray(ids)] = float(limit)
    return out


def _physx_armature_friction(robot: Articulation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Armature + joint friction as PhysX actually received them.

    Mirrors ``Articulation._log_articulation_info``: armature via
    ``get_dof_armatures``; friction via ``get_dof_friction_coefficients``
    (Isaac Sim < 5) or ``get_dof_friction_properties`` (>= 5, a (J, 3) array
    of static-friction effort / dynamic-friction effort / viscous-friction
    coefficient — the reworked PhysX joint-friction model). Falls back to the
    synced ``robot.data`` buffers if a physx-view accessor is unavailable.

    Returns ``(armature, static_friction, friction_props)`` where
    ``friction_props`` is always (J, 3) — for Isaac Sim < 5 the deprecated
    load-proportional coefficient lands in column 0 with zeros elsewhere.
    """
    pv = robot.root_physx_view
    num_joints = robot.num_joints
    try:
        armature = to_numpy(pv.get_dof_armatures())[0]
    except AttributeError:
        armature = to_numpy(robot.data.joint_armature)[0]
    props: np.ndarray | None = None
    try:
        from isaacsim.core.version import get_version  # ty: ignore[unresolved-import]

        isaac_sim_major: int | None = int(get_version()[2])
    except Exception:
        isaac_sim_major = None
    if isaac_sim_major is not None and isaac_sim_major < 5 and hasattr(pv, "get_dof_friction_coefficients"):
        coeff = to_numpy(pv.get_dof_friction_coefficients())[0]
        props = np.zeros((num_joints, 3), dtype=np.float64)
        props[:, 0] = coeff
    elif hasattr(pv, "get_dof_friction_properties"):
        props = to_numpy(pv.get_dof_friction_properties())[0].astype(np.float64)
    elif hasattr(pv, "get_dof_friction_coefficients"):
        coeff = to_numpy(pv.get_dof_friction_coefficients())[0]
        props = np.zeros((num_joints, 3), dtype=np.float64)
        props[:, 0] = coeff
    if props is None:
        props = np.zeros((num_joints, 3), dtype=np.float64)
        props[:, 0] = to_numpy(robot.data.joint_friction_coeff)[0]
    return armature, props[:, 0].copy(), props


def _measure_reflected_inertia(sim: sim_utils.SimulationContext, robot: Articulation, probe_tau: float) -> np.ndarray:
    """Per-joint reflected inertia from a one-step torque probe, ``M[j] = tau * dt / qd[j]``.

    Each probe resets to the default state at rest, applies ``probe_tau`` on
    joint j alone for one physics step, and reads that joint's velocity. In
    uniform free fall gravity induces no joint acceleration, so the response
    is the plant's alone (joint friction biases M[j] slightly upward-safe:
    the estimate is used only to CAP the excitation amplitudes).
    """
    dt = float(sim.get_physics_dt())
    num_joints = robot.num_joints
    inertia = np.zeros(num_joints, dtype=np.float64)
    zeros = np.zeros(num_joints)
    for j in range(num_joints):
        reset_to_default_state(robot, qd0=zeros)
        tau = torch.zeros(1, num_joints, device=robot.device)
        tau[0, j] = probe_tau
        robot.set_joint_effort_target(tau)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        qd_j = float(robot.data.joint_vel[0, j])
        if abs(qd_j) < 1e-9:
            raise RuntimeError(
                f"inertia probe: joint {j} ('{robot.data.joint_names[j]}') did not move under "
                f"{probe_tau} N*m; cannot size the excitation amplitude"
            )
        inertia[j] = probe_tau * dt / abs(qd_j)
    return inertia


def _run_phase(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    *,
    qd0: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reset to the default joint state at the origin and step ``tau.shape[0]`` physics steps.

    Returns the per-phase npz arrays WITHOUT the phase prefix. The ``init_*``
    entries are the exact values written to sim (root link pose at the origin,
    zero root velocity, default joint positions, ``qd0`` joint velocities).
    """
    device = robot.device
    num_steps = tau.shape[0]
    dt = float(sim.get_physics_dt())
    joint_pos = reset_to_default_state(robot, qd0=qd0)

    rec = state_recorder("applied_torque")
    tau_t = torch.as_tensor(tau, dtype=torch.float32, device=device)
    for k in range(num_steps):
        robot.set_joint_effort_target(tau_t[k].reshape(1, -1))
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        append_state(rec, robot.data)
        rec["applied_torque"].append(to_numpy(robot.data.applied_torque[0]))

    arrays = {key: np.stack(values) for key, values in rec.items()}
    arrays["tau"] = tau.astype(np.float64)
    arrays.update(written_init_state(np.zeros(3), joint_pos, qd0))
    return arrays


def main() -> None:
    torch.manual_seed(args_cli.seed)
    base_cfg = load_robot_cfg(args_cli.robot)
    robot_cfg = _freespace_cfg(base_cfg)

    # CPU by default (see the --device override above the parser).
    device = str(args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(robot_cfg)
    sim.reset()

    num_joints = robot.num_joints
    joint_pos_limits_orig = to_numpy(robot.root_physx_view.get_dof_limits())[0]
    wide_limits = torch.zeros(1, num_joints, 2, device=robot.device)
    wide_limits[..., 0] = -WIDE_JOINT_LIMIT_RAD
    wide_limits[..., 1] = WIDE_JOINT_LIMIT_RAD
    robot.write_joint_position_limit_to_sim(wide_limits, warn_limit_violation=False)
    # Read back and check the INVARIANT (limits at least +-2pi wide), not the write:
    # PhysX refuses some limit writes with only a console error (which would leave
    # narrow authored limits active while the npz claims otherwise), and unlimited
    # joints read back as the +-3.4e38 sentinel — wider than any write, fine as-is
    # (measured on anymal_c: unlimited joints, write ignored).
    applied_limits = to_numpy(robot.root_physx_view.get_dof_limits())[0]
    wide_enough = (applied_limits[:, 0] <= -WIDE_JOINT_LIMIT_RAD + 1e-3) & (
        applied_limits[:, 1] >= WIDE_JOINT_LIMIT_RAD - 1e-3
    )
    if not wide_enough.all():
        raise RuntimeError(
            f"joint-limit widening did not take (PhysX kept {applied_limits.tolist()}); "
            "the no-limit invariant of the protocol would be silently violated"
        )

    armature_physx, friction_physx, friction_props_physx = _physx_armature_friction(robot)
    effort_limit_physx = to_numpy(robot.root_physx_view.get_dof_max_forces())[0]
    velocity_limit_physx = to_numpy(robot.root_physx_view.get_dof_max_velocities())[0]
    effort_authored = _resolve_authored_effort_limits(robot, base_cfg.actuators)
    reflected_inertia = _measure_reflected_inertia(sim, robot, INERTIA_PROBE_TAU)

    # Default-state link poses, so welded-body composition can be checked offline
    # against the plant record.
    reset_to_default_state(robot)
    body_link_pos_w = to_numpy(robot.data.body_link_pos_w)[0]
    body_link_quat_w = to_numpy(robot.data.body_link_quat_w)[0]

    j = np.arange(num_joints, dtype=np.float64)
    qd0 = ((-1.0) ** j) * (0.5 + 0.17 * (np.arange(num_joints) % 7))
    amp = np.minimum(0.25 * np.minimum(effort_authored, AMP_EFFORT_CAP), args_cli.excite_accel * reflected_inertia)
    # Hold (do not drive) joints whose amplitude cannot clear breakaway: a
    # sub-breakaway sinusoid from rest produces only stiction capture/breakaway
    # events — O(one physics step) of friction impulse each, pure timing noise —
    # while an undriven resting joint is held bit-exactly by both engines
    # (see the module docstring; the passive phase probes those joints).
    held = amp < STICTION_HOLD_FACTOR * friction_physx
    amp = np.where(held, 0.0, amp)
    if held.any():
        held_names = [robot.data.joint_names[k] for k in np.nonzero(held)[0]]
        print(f"[INFO] Excited phase holds sub-breakaway joints {held_names} (probed by the passive phase).")
    freq_hz = 0.5 + 0.35 * (np.arange(num_joints) % 8)
    phase_rad = 0.7 * j
    t_k = np.arange(args_cli.num_steps, dtype=np.float64)[:, None] * args_cli.physics_dt
    tau_excited = amp[None, :] * np.sin(2.0 * np.pi * freq_hz[None, :] * t_k + phase_rad[None, :])
    tau_passive = np.zeros((args_cli.num_steps, num_joints), dtype=np.float64)

    print(
        f"[INFO] Robot '{args_cli.robot}': {num_joints} joints, dt={args_cli.physics_dt}, "
        f"num_steps={args_cli.num_steps}"
    )
    print(f"[INFO] PhysX armature: {np.array2string(armature_physx, precision=4)}")
    print(f"[INFO] PhysX friction (static/dynamic/viscous):\n{np.array2string(friction_props_physx, precision=4)}")
    print(f"[INFO] Reflected inertia (probe): {np.array2string(reflected_inertia, precision=5)}")
    print(f"[INFO] Excitation amplitudes A: {np.array2string(amp, precision=4)}")

    phases = ["passive", "excited"] if args_cli.phase == "both" else [args_cli.phase]
    arrays: dict[str, np.ndarray] = {
        "robot": np.array(args_cli.robot),
        "seed": np.array(args_cli.seed),
        "device": np.array(device),
        "num_steps": np.array(args_cli.num_steps),
        "physics_dt": np.array(args_cli.physics_dt),
        "gravity_w": np.array(sim_cfg.gravity, dtype=np.float64),
        "phases": np.array(phases),
        **plant_record(robot),
        "body_link_pos_w": body_link_pos_w,
        "body_link_quat_w": body_link_quat_w,
        "armature_physx": armature_physx,
        "friction_physx": friction_physx,
        "friction_props_physx": friction_props_physx,
        "effort_limit_physx": effort_limit_physx,
        "velocity_limit_physx": velocity_limit_physx,
        "effort_limit_authored": effort_authored,
        "joint_reflected_inertia": reflected_inertia,
        "inertia_probe_tau": np.array(INERTIA_PROBE_TAU),
        "excite_accel": np.array(args_cli.excite_accel),
        "joint_pos_limits_orig": joint_pos_limits_orig,
        "joint_limits_disabled": np.array(True),
        "self_collisions_disabled": np.array(True),
        "passive_qd0": qd0,
        "excited_held": held,
        "excited_amp": amp,
        "excited_freq_hz": freq_hz,
        "excited_phase_rad": phase_rad,
    }
    for phase in phases:
        print(f"[INFO] Running phase '{phase}' ({args_cli.num_steps} physics steps)...")
        if phase == "passive":
            phase_arrays = _run_phase(sim, robot, qd0=qd0, tau=tau_passive)
        else:
            phase_arrays = _run_phase(sim, robot, qd0=np.zeros(num_joints), tau=tau_excited)
        for key, value in phase_arrays.items():
            arrays[f"{phase}_{key}"] = value
        if phase == "excited":
            qd_phase = phase_arrays["joint_vel"]
            at_rest = np.any(qd_phase == 0.0, axis=0)
            stopped = [n for k, n in enumerate(robot.data.joint_names) if not held[k] and at_rest[k]]
            moved = [n for k, n in enumerate(robot.data.joint_names) if held[k] and bool(np.any(qd_phase[:, k] != 0.0))]
            if stopped:
                print(
                    f"[WARN] excited phase: driven joints {stopped} stopped mid-probe (PhysX capture "
                    "events); stiction-timing noise contaminates the plant signal for them"
                )
            if moved:
                print(
                    f"[WARN] excited phase: held joints {moved} broke away (Coriolis coupling exceeded "
                    "their breakaway effort); their divergence is stiction-timing noise"
                )
        final_pos = phase_arrays["root_pos_w"][-1]
        print(f"[INFO] Phase '{phase}' final root_pos_w = {np.array2string(final_pos, precision=4)}")

    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args_cli.out, **arrays)
    print(f"[INFO] Dumped phases {phases} for robot '{args_cli.robot}' to {args_cli.out}")


if __name__ == "__main__":
    main()
    shutdown(simulation_app)
