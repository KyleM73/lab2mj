"""Free-space plant-dynamics replay helpers for MuJoCo bundles.

Library side of the no-contact parity test between IsaacLab (PhysX) and a
converted bundle: ``lab2mj/isaac/dump_freespace.py`` records the Isaac
reference and ``scripts/replay_freespace_mujoco.py`` drives these
helpers. The replay isolates PLANT dynamics (mass / inertia / armature /
joint friction / gravity / Coriolis):

* all contacts are disabled (``geom_contype = geom_conaffinity = 0``) and the
  joint-limit constraint is switched off (the Isaac dump widens its limits to
  match),
* ``dof_damping`` is zeroed on joints owned by ``implicit_pd`` actuator
  groups — the converter authors those entries to realize the actuator kd,
  which is drive, not plant (:func:`zero_implicit_pd_damping` asserts the
  zeroed values equal the manifest kd before touching them),
* ``dof_armature`` and ``dof_frictionloss`` stay: they ARE plant here.

The recorded Isaac torques are applied as raw ``ctrl`` (the bundles' actuators
are unit-gain torque ``<general>`` actuators in MJCF joint order) and each
Isaac physics step is integrated as ``physics_substeps`` MuJoCo steps with the
ctrl held constant — the same substep scheme :class:`lab2mj.env.MjEnv`
uses for explicit actuator groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco
import numpy as np

from lab2mj.actuators import ActuatorSet
from lab2mj.bundle import XML_RTOL
from lab2mj.env import assert_actuator_order, build_robot_map, free_joint_qvel, stamp_exact_inertia
from lab2mj.events import RobotMap, read_root_com_vel_w
from lab2mj.ir import ActuatorGroupIR
from lab2mj.quat import quat_angle_rad
from lab2mj.stiction import JointStiction

__all__ = [
    "PhaseResult",
    "assert_actuator_order",
    "build_stiction",
    "disable_contacts",
    "disable_joint_limits",
    "divergence_stats",
    "implicit_pd_damping_selection",
    "isaac_to_mj_perm",
    "quat_angle_rad",
    "replay_phase",
    "zero_implicit_pd_damping",
]


def isaac_to_mj_perm(manifest: dict[str, Any]) -> np.ndarray:
    """The manifest's Isaac->MJCF joint permutation: ``values_mj = values_isaac[perm]``."""
    return np.asarray(manifest["robot"]["isaac_to_mj"], dtype=np.intp)


def disable_contacts(model: mujoco.MjModel) -> None:
    """Remove every possible contact pair by clearing all geom contype/conaffinity masks."""
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0


def disable_joint_limits(model: mujoco.MjModel) -> None:
    """Switch off the joint-limit constraint globally (frictionloss stays active)."""
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT


def implicit_pd_damping_selection(manifest: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Joints whose model ``dof_damping`` realizes an ``implicit_pd`` group's kd.

    Returns the joint names, their kd values, and their viscous-friction
    coefficients (all aligned): the model's ``dof_damping`` on these joints is
    ``kd + viscous`` — kd is drive, viscous is plant. Joints of explicit
    actuator groups are excluded: their kd is computed as a torque at runtime
    and never enters the compiled model.
    """
    isaac_joint_order = list(manifest["robot"]["isaac_joint_order"])
    actuator_set = ActuatorSet.from_ir([ActuatorGroupIR.from_dict(g) for g in manifest["actuators"]], isaac_joint_order)
    names: list[str] = []
    kds: list[float] = []
    viscous: list[float] = []
    for group in actuator_set.groups:
        if group.model != "implicit_pd":
            continue
        for joint_idx, joint_name in zip(group.joint_ids, group.joint_names):
            names.append(joint_name)
            kds.append(float(actuator_set.kd[int(joint_idx)]))
            viscous.append(float(actuator_set.viscous_friction[int(joint_idx)]))
    return names, np.asarray(kds, dtype=np.float64), np.asarray(viscous, dtype=np.float64)


def zero_implicit_pd_damping(model: mujoco.MjModel, manifest: dict[str, Any]) -> list[str]:
    """Remove the ``implicit_pd`` kd from ``dof_damping``; returns the affected joint names.

    Viscous joint friction stays in ``dof_damping`` — it is plant, not drive.
    Raises ``ValueError`` if a selected joint's current ``dof_damping`` does
    not equal the manifest kd + viscous — that means the model was not
    produced by the converter (or is stale) and the zeroing would remove
    plant damping.
    """
    names, kds, viscous = implicit_pd_damping_selection(manifest)
    for joint_name, kd, visc in zip(names, kds, viscous):
        dof = int(model.jnt_dofadr[model.joint(joint_name).id])
        if not np.isclose(model.dof_damping[dof], kd + visc, rtol=XML_RTOL, atol=1e-12):
            raise ValueError(
                f"joint '{joint_name}': dof_damping {model.dof_damping[dof]} != manifest implicit_pd kd {kd} "
                f"+ viscous friction {visc}; the bundle model does not carry the converter's actuator-kd damping"
            )
        model.dof_damping[dof] = visc
    return names


def build_stiction(
    model: mujoco.MjModel, robot_map: RobotMap, manifest: dict[str, Any], *, physics_dt: float | None = None
) -> JointStiction:
    """The bundle's PhysX static-friction emulation (no-op object when no joint needs it).

    Same construction :class:`lab2mj.env.MjEnv` uses: the capture window
    is computed against the ISAAC physics dt — the manifest's by default, or
    ``physics_dt`` when the replay re-times integration to a dt-sweep dump's rate
    (a window built from the wrong dt scales the capture velocity by the dt ratio).
    """
    isaac_joint_order = list(manifest["robot"]["isaac_joint_order"])
    actuator_set = ActuatorSet.from_ir([ActuatorGroupIR.from_dict(g) for g in manifest["actuators"]], isaac_joint_order)
    return JointStiction(
        model,
        robot_map.dof_adr,
        actuator_set.static_friction,
        actuator_set.dynamic_friction,
        float(manifest["timing"]["physics_dt"]) if physics_dt is None else float(physics_dt),
    )


@dataclass
class PhaseResult:
    """Per-physics-step MuJoCo trajectory in the Isaac dump's conventions.

    Joint arrays are in Isaac joint order; quaternions wxyz; ``root_lin_vel_w``
    is the root-link CoM velocity, ``root_link_lin_vel_w`` the link-origin
    velocity (both world frame).
    """

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_pos_w: np.ndarray
    root_quat_w: np.ndarray
    root_lin_vel_w: np.ndarray
    root_link_lin_vel_w: np.ndarray
    root_ang_vel_w: np.ndarray


def replay_phase(
    model: mujoco.MjModel,
    robot_map: RobotMap,
    manifest: dict[str, Any],
    *,
    init: dict[str, np.ndarray],
    tau_isaac: np.ndarray,
    substeps: int,
    stiction: JointStiction | None = None,
    ctrl_fn: Callable[[mujoco.MjData], None] | None = None,
) -> PhaseResult:
    """Replay one dump phase: raw torques on a fresh ``MjData``; returns the trajectory.

    ``init`` must contain ``root_pos_w``, ``root_quat_w``,
    ``root_link_lin_vel_w``, ``root_ang_vel_w``, ``joint_pos``, ``joint_vel``
    (Isaac order — the dump's ``<phase>_init_*`` keys). ``tau_isaac`` is
    ``(T, J)`` in Isaac joint order; row ``k`` is written to ``ctrl`` once and
    held across the ``substeps`` MuJoCo steps that integrate Isaac physics
    step ``k``. The state at index ``k`` is the state after step ``k``.
    ``stiction`` (see :func:`build_stiction`) switches the friction bound of
    static-friction joints before every MuJoCo step, mirroring PhysX's
    stationary-joint capture — it IS plant behavior in the Isaac dump.
    ``ctrl_fn`` replaces the held-torque control: it is called before every
    MuJoCo substep with the live ``MjData`` and must write ``data.ctrl``
    itself (``tau_isaac`` then only sets the trajectory length).
    """
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    assert_actuator_order(model, manifest)
    perm = isaac_to_mj_perm(manifest)
    tau_isaac = np.asarray(tau_isaac, dtype=np.float64)
    num_steps, num_joints = tau_isaac.shape
    if num_joints != perm.shape[0]:
        raise ValueError(f"tau_isaac has {num_joints} joints, manifest has {perm.shape[0]}")

    data = mujoco.MjData(model)
    qa, da = robot_map.root_qpos_adr, robot_map.root_dof_adr
    root_quat_wxyz = np.asarray(init["root_quat_w"], dtype=np.float64)
    data.qpos[qa : qa + 3] = np.asarray(init["root_pos_w"], dtype=np.float64)
    data.qpos[qa + 3 : qa + 7] = root_quat_wxyz
    data.qvel[da : da + 6] = free_joint_qvel(root_quat_wxyz, init["root_link_lin_vel_w"], init["root_ang_vel_w"])
    data.qpos[robot_map.qpos_adr] = np.asarray(init["joint_pos"], dtype=np.float64)
    data.qvel[robot_map.dof_adr] = np.asarray(init["joint_vel"], dtype=np.float64)
    mujoco.mj_forward(model, data)

    rec = PhaseResult(
        joint_pos=np.empty((num_steps, num_joints)),
        joint_vel=np.empty((num_steps, num_joints)),
        root_pos_w=np.empty((num_steps, 3)),
        root_quat_w=np.empty((num_steps, 4)),
        root_lin_vel_w=np.empty((num_steps, 3)),
        root_link_lin_vel_w=np.empty((num_steps, 3)),
        root_ang_vel_w=np.empty((num_steps, 3)),
    )
    tau_mj = tau_isaac[:, perm]
    for k in range(num_steps):
        if ctrl_fn is None:
            data.ctrl[:] = tau_mj[k]
        for _ in range(substeps):
            if ctrl_fn is not None:
                ctrl_fn(data)
            if stiction is not None:
                stiction.apply(model, data)
            mujoco.mj_step(model, data)
        rec.joint_pos[k] = data.qpos[robot_map.qpos_adr]
        rec.joint_vel[k] = data.qvel[robot_map.dof_adr]
        rec.root_pos_w[k] = data.qpos[qa : qa + 3]
        rec.root_quat_w[k] = data.qpos[qa + 3 : qa + 7]
        rec.root_link_lin_vel_w[k] = data.qvel[da : da + 3]
        com_vel_w = read_root_com_vel_w(model, data, robot_map)
        rec.root_lin_vel_w[k] = com_vel_w[0:3]
        rec.root_ang_vel_w[k] = com_vel_w[3:6]
    return rec


def divergence_stats(
    *,
    dq: np.ndarray,
    dqd: np.ndarray,
    root_pos_err_m: np.ndarray,
    root_quat_err_rad: np.ndarray,
    dt: float,
    marks_s: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Divergence summary over a phase: full-horizon extremes plus snapshots at ``marks_s``.

    ``dq``/``dqd`` are ``(T, J)`` signed differences; the root errors are
    ``(T,)``. Snapshot index for mark ``s`` is the state after physics step
    ``round(s / dt) - 1`` (skipped when beyond the horizon).
    """
    abs_dq, abs_dqd = np.abs(dq), np.abs(dqd)
    num_steps = abs_dq.shape[0]
    stats: dict[str, Any] = {
        "num_steps": int(num_steps),
        "duration_s": float(num_steps * dt),
        "max_abs_dq_rad": float(abs_dq.max()),
        "mean_abs_dq_rad": float(abs_dq.mean()),
        "max_abs_dqd_rad_s": float(abs_dqd.max()),
        "mean_abs_dqd_rad_s": float(abs_dqd.mean()),
        "max_root_pos_err_m": float(np.max(root_pos_err_m)),
        "final_root_pos_err_m": float(root_pos_err_m[-1]),
        "max_root_quat_err_rad": float(np.max(root_quat_err_rad)),
        "final_root_quat_err_rad": float(root_quat_err_rad[-1]),
        "at": {},
    }
    for mark in marks_s:
        idx = int(round(mark / dt)) - 1
        if idx < 0 or idx >= num_steps:
            continue
        stats["at"][f"{mark:g}s"] = {
            "step": idx,
            "max_abs_dq_rad": float(abs_dq[idx].max()),
            "max_abs_dqd_rad_s": float(abs_dqd[idx].max()),
            "root_pos_err_m": float(root_pos_err_m[idx]),
            "root_quat_err_rad": float(root_quat_err_rad[idx]),
        }
    return stats


def load_freespace_model(bundle_dir: Any) -> tuple[mujoco.MjModel, RobotMap, dict[str, Any]]:
    """Load a bundle's scene model + robot map + manifest (no free-space edits applied)."""
    from lab2mj import bundle

    manifest = bundle.read_manifest(bundle_dir)
    model = mujoco.MjModel.from_xml_path(str(bundle.scene_xml_path(bundle_dir)))
    stamp_exact_inertia(model, mujoco.MjData(model), manifest)
    robot_map = build_robot_map(model, manifest)
    return model, robot_map, manifest
