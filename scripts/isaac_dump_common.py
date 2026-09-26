"""Shared infrastructure for the Isaac-side dumper scripts (GPU box only).

``launch_app`` must run before any other isaaclab import; the remaining
helpers import torch lazily so this module is importable pre-launch.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

# Robot name -> (module, attribute), imported lazily (Isaac Sim must be up first).
# Spans the bundle fleet so freespace/contact references exist for every robot.
_ROBOT_CFGS: dict[str, tuple[str, str]] = {
    "spot": ("contact_lab.assets.spot", "SPOT_CFG"),
    "g1": ("isaaclab_assets", "G1_MINIMAL_CFG"),
    # The velocity tasks run the MINIMAL variant (h1_minimal.usd).
    "h1": ("isaaclab_assets", "H1_MINIMAL_CFG"),
    "a1": ("isaaclab_assets", "UNITREE_A1_CFG"),
    "go1": ("isaaclab_assets", "UNITREE_GO1_CFG"),
    "go2": ("isaaclab_assets", "UNITREE_GO2_CFG"),
    "anymal_b": ("isaaclab_assets", "ANYMAL_B_CFG"),
    "anymal_c": ("isaaclab_assets", "ANYMAL_C_CFG"),
    "anymal_d": ("isaaclab_assets", "ANYMAL_D_CFG"),
}
ROBOT_CHOICES = tuple(_ROBOT_CFGS)

# Effectively-unlimited PhysX solver clamp for pure-torque protocols.
RAW_TORQUE_LIMIT = 1.0e9

# Articulation state every dumper records per step, under the reference-dump
# key/frame conventions (root_lin_vel_w = root CoM velocity, root_link_lin_vel_w
# = link-origin velocity, both world frame; quaternions wxyz).
STATE_KEYS = (
    "joint_pos",
    "joint_vel",
    "root_pos_w",
    "root_quat_w",
    "root_lin_vel_w",
    "root_link_lin_vel_w",
    "root_ang_vel_w",
)


def launch_app(parser: Any) -> tuple[Any, Any]:
    """Parse CLI args and boot headless Isaac Sim; returns ``(args, simulation_app)``.

    Typo'd ``--`` flags error out instead of being silently ignored
    (single-dash tokens still pass through to Kit); Kit-style
    ``--/path`` args pass through to the app. ``--help`` is answered before the
    isaaclab import (fast, no sim boot) because ``AppLauncher.add_app_launcher_args``
    breaks argparse's help action; the printed help omits the AppLauncher flags.
    """
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        parser.print_help()
        raise SystemExit(0)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, passthrough = parser.parse_known_args()
    unknown = [a for a in passthrough if a.startswith("--") and not a.startswith("--/")]
    if unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    args_cli.headless = True
    sys.argv = [sys.argv[0]] + passthrough
    app_launcher = AppLauncher(args_cli)
    return args_cli, app_launcher.app


def shutdown(simulation_app: Any) -> None:
    """Close Isaac Sim and end the dumper process; never hangs.

    ``SimulationApp.close`` can deadlock in PhysX cleanup (known Isaac Sim
    issue; see ``contact_lab.app.minimalist``), and even a clean close can
    leave non-daemon Kit threads keeping a finished process alive — a hung
    dumper on the GPU box blocks later runs. A watchdog hard-exits if close
    deadlocks, and the process hard-exits afterwards regardless (the dump is
    already on disk; exit code 0 either way). Call as the script's last
    statement, after ``main()``.
    """
    import os
    import threading

    def _force_exit() -> None:
        print("[shutdown] Isaac Sim close() deadlocked; forcing exit (the dump is already written)", flush=True)
        os._exit(0)

    watchdog = threading.Timer(60.0, _force_exit)
    watchdog.daemon = True
    watchdog.start()
    simulation_app.close()
    try:
        from contact_lab.app.minimalist import cleanup
    except ImportError:  # stock IsaacLab install without contact_lab
        cleanup = None
    if cleanup is not None:
        cleanup()
    os._exit(0)


def load_robot_cfg(name: str) -> Any:
    """Articulation cfg registry shared by the robot-based dumpers."""
    import importlib

    try:
        module_name, attr = _ROBOT_CFGS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r} (choices: {ROBOT_CHOICES})") from None
    return getattr(importlib.import_module(module_name), attr)


def zero_gain_implicit_actuators(actuators: dict[str, Any], *, unlimited: bool) -> dict[str, Any]:
    """Zero-gain ``ImplicitActuatorCfg`` swap that preserves the plant fields.

    Each group's ``armature`` / ``friction`` / ``dynamic_friction`` /
    ``viscous_friction`` are carried over verbatim (None / scalar / regex-dict
    alike): ``Articulation._process_actuators_cfg`` then writes the exact same
    plant values to sim as in training (None resolves to the USD-authored
    value). ``stiffness = damping = 0`` removes the drive, so
    ``set_joint_effort_target`` is a pure torque input. With ``unlimited`` the
    solver effort/velocity clamps are lifted to ``RAW_TORQUE_LIMIT`` (the
    free-space protocol); otherwise the authored effort limit is kept (the
    contact protocol's stiff drop is clamped by it).
    """
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        name: ImplicitActuatorCfg(
            joint_names_expr=list(group.joint_names_expr),
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=RAW_TORQUE_LIMIT if unlimited else getattr(group, "effort_limit", None),
            velocity_limit_sim=RAW_TORQUE_LIMIT if unlimited else None,
            armature=group.armature,
            friction=group.friction,
            dynamic_friction=group.dynamic_friction,
            viscous_friction=group.viscous_friction,
        )
        for name, group in actuators.items()
    }


def reset_to_default_state(
    robot: Any, *, root_pos: tuple[float, float, float] = (0.0, 0.0, 0.0), qd0: np.ndarray | None = None
) -> Any:
    """Write root pose ``root_pos`` / identity quaternion, zero root velocity, the
    default joint positions, and ``qd0`` (default zero) joint velocities, then reset.

    Returns the written (1, J) joint-position tensor.
    """
    import torch

    device = robot.device
    root_pose = torch.zeros(1, 7, device=device)
    root_pose[0, :3] = torch.as_tensor(root_pos, dtype=torch.float32, device=device)
    root_pose[0, 3] = 1.0
    robot.write_root_link_pose_to_sim(root_pose)
    robot.write_root_com_velocity_to_sim(torch.zeros(1, 6, device=device))
    joint_pos = robot.data.default_joint_pos.clone()
    if qd0 is None:
        joint_vel = torch.zeros_like(joint_pos)
    else:
        joint_vel = torch.as_tensor(qd0, dtype=torch.float32, device=device).reshape(1, -1)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    return joint_pos


def written_init_state(root_pos: Any, joint_pos: Any, joint_vel: Any) -> dict[str, np.ndarray]:
    """The ``init_*`` npz block the protocol dumpers store: the exact reset state
    written by :func:`reset_to_default_state` (identity root quaternion, zero root
    velocity)."""
    return {
        "init_root_pos_w": np.asarray(root_pos, dtype=np.float64),
        "init_root_quat_w": np.array([1.0, 0.0, 0.0, 0.0]),
        "init_root_lin_vel_w": np.zeros(3),
        "init_root_link_lin_vel_w": np.zeros(3),
        "init_root_ang_vel_w": np.zeros(3),
        "init_joint_pos": to_numpy(joint_pos[0], np.float64),
        "init_joint_vel": np.asarray(joint_vel, dtype=np.float64),
    }


def to_numpy(value: Any, dtype: Any = None) -> np.ndarray:
    """Detached numpy copy of a tensor, optionally cast to ``dtype``."""
    import torch

    out = torch.as_tensor(value).detach().cpu().numpy()
    return out.astype(dtype) if dtype is not None else out.copy()


def state_recorder(*extra_keys: str) -> dict[str, list[np.ndarray]]:
    """Empty per-step recording dict over ``STATE_KEYS`` plus ``extra_keys``."""
    return {key: [] for key in STATE_KEYS + extra_keys}


def append_state(rec: dict[str, list[np.ndarray]], data: Any, dtype: Any = None) -> None:
    """Append env-0's current articulation state for every ``STATE_KEYS`` entry."""
    for key in STATE_KEYS:
        rec[key].append(to_numpy(getattr(data, key)[0], dtype))


def plant_record(robot: Any, dtype: Any = None) -> dict[str, np.ndarray]:
    """The PhysX-resolved plant record every dump schema shares.

    ``get_coms`` quaternions are xyzw. PhysX validates and silently recomputes
    invalid authored inertias, so these — not the USD-authored values — are the
    plant the reference ran with; ``convert --dump`` authors them into the
    bundle when they disagree with the USD.
    """
    view = robot.root_physx_view
    return {
        "joint_names": np.array(list(robot.data.joint_names)),
        "body_names": np.array(list(robot.data.body_names)),
        "default_joint_pos": to_numpy(robot.data.default_joint_pos[0], dtype),
        "default_joint_vel": to_numpy(robot.data.default_joint_vel[0], dtype),
        "body_masses": to_numpy(view.get_masses()[0], dtype),
        "body_coms_physx": to_numpy(view.get_coms()[0], dtype),
        "body_inertias_physx": to_numpy(view.get_inertias()[0], dtype),
    }
