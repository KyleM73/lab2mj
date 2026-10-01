"""Task-agnostic Isaac-side reference dumper for sim2sim validation.

Loads any manager-based task with ``num_envs=1``, runs a TorchScript policy
deterministically for ``--num_steps`` policy steps, and records the exact
observations, actions, and rigid-body state to an npz file. The dump is the
ground-truth reference the MuJoCo-side validation gates compare against.

With ``--strict`` (default) the environment is made fully deterministic:

* observation corruption is disabled on every observation group, including groups
  embedded in action terms (e.g. ``PreTrainedPolicyAction.low_level_observations``),
* ALL event terms (startup / reset / interval) are replaced by a single
  deterministic ``reset_scene_to_default`` reset term, so the robot starts
  from the scene's configured default state (default root pose/velocity and
  default joint state) with no randomization. Merely removing every event
  term would NOT achieve this: spawning only applies the root transform, so
  without a reset term the joints would start at the USD-authored pose, which
  need not match the articulation cfg's ``init_state``. contact_lab's deterministic
  plant terms (``set_legacy_joint_friction``, ``check_joint_friction``) are kept: they
  define the simulated joint friction rather than randomize it,
* the curriculum is disabled,
* every command term's ranges are pinned to the single commanded value with
  ``resampling_time_range=(1e9, 1e9)`` and ``rel_standing_envs=0``. If a
  velocity command term has ``heading_command=True``, it is set to False and
  the ``ang_vel_z`` range is pinned to the commanded ``wz`` instead (heading
  tracking would otherwise turn ``wz`` into a closed-loop heading controller).
  Command terms that are neither velocity nor pose commands only get their
  resampling frozen (a warning is printed); their actual, possibly
  seed-dependent value is recorded in the per-term ``command_<term>`` npz keys.

Example commands::

    # Stock IsaacLab G1 velocity task with an rsl_rl-exported jit policy.
    python -m lab2mj.isaac.dump_reference \\
        --task Isaac-Velocity-Flat-G1-v0 \\
        --policy logs/rsl_rl/g1_flat/<run>/exported/policy.pt \\
        --command 0.8,0.0,0.0 --num_steps 400 --seed 42 \\
        --out logs/sim2sim/g1_flat_reference.npz

    # contact_lab Spot velocity task with a contact_lab-exported jit policy.
    python -m lab2mj.isaac.dump_reference \\
        --task spot-velocity-v0 \\
        --policy logs/spot_velocity/<run>/exported/policy.pt \\
        --command 1.0,0.0,0.0 \\
        --out logs/sim2sim/spot_velocity_reference.npz

npz schema (J = num joints, B = num bodies, T = num_steps, C = contact-sensor
bodies, O = policy obs dim, A = action dim). All joint-indexed arrays use the
Isaac (PhysX breadth-first) joint order; quaternions are wxyz; ``root_lin_vel_w``
and ``root_ang_vel_w`` are the root CoM velocities in the world frame,
``root_link_lin_vel_w`` is the root link-origin linear velocity in the world
frame (angular velocity is point-independent, so no link duplicate is stored).

===========================  ===========  ====================================================
key                          shape        description
===========================  ===========  ====================================================
task                         ()           task id string
policy_path                  ()           path of the TorchScript policy that was run
seed                         ()           env seed
strict                       ()           whether strict overrides were applied
settle_steps                 ()           requested --settle_steps (init_episode_step is the
                                          episode clock actually reached)
command                      (3,)         requested [vx, vy, wz] CLI velocity command (pinned
                                          onto velocity terms only in strict mode)
command_<term>               (T, D)       per-step command tensor of each command-manager term
                                          (the ground truth actually fed to the policy)
command_<term>_goal_w        (4,)         resolved post-reset world-frame goal
                                          [x_w, y_w, z_w, heading_w] of each pose command
                                          term (the frame the MuJoCo side pins)
pose_command                 (4,)         raw CLI 'x,y,z,heading' (only with --pose_command;
                                          NOT the world goal — see command_<term>_goal_w)
joint_names                  (J,)         Isaac-order joint names
body_names                   (B,)         body names
default_joint_pos            (J,)         articulation default joint positions
default_joint_vel            (J,)         articulation default joint velocities
body_masses                  (B,)         PhysX-resolved body masses (post startup events)
body_coms_physx              (B, 7)       PhysX-resolved CoM pose in the link frame
                                          (x, y, z, qx, qy, qz, qw — principal axes)
body_inertias_physx          (B, 9)       PhysX-resolved inertia about the CoM, link frame
                                          (3x3 rows); authoritative over USD-authored values
physics_dt                   ()           simulation dt [s]
decimation                   ()           physics steps per policy step
episode_length_s             ()           configured episode length [s]
env_origin_w                 (3,)         env-0 origin, world frame
gravity_w                    (3,)         gravity vector, world frame
init_root_pos_w              (3,)         post-reset root link position, world
init_root_quat_w             (4,)         post-reset root link quaternion (wxyz), world
init_root_lin_vel_w          (3,)         post-reset root CoM linear velocity, world
init_root_link_lin_vel_w     (3,)         post-reset root link-origin linear velocity, world
init_root_ang_vel_w          (3,)         post-reset root angular velocity, world
init_joint_pos               (J,)         post-reset joint positions
init_joint_vel               (J,)         post-reset joint velocities
init_last_action             (A,)         action-manager buffer at recording start (the settle
                                          phase's last raw action; seeds the last_action obs
                                          term so obs parity matches on settled dumps)
init_episode_step            ()           episode_length_buf at recording start (settle_steps
                                          unless a reset fired during settling)
init_low_level_action        (A_ll,)      PreTrainedPolicyAction low-level action buffer at
                                          recording start (nav tasks only)
init_actuator_lag_<group>    ()           DelayedPD delay lag (physics steps) sampled at reset
                                          for env 0, one key per delayed actuator group
ll_action                    (T, A_ll)    low-level action buffer BEFORE each policy step's
                                          decimation loop (nav tasks only; feeds the
                                          informative low-level parity metric in validate)
obs0                         (O,)         first policy observation (== obs[0])
obs                          (T, O)       exact tensor fed to the policy at each step
action_raw                   (T, A)       raw policy output fed to env.step (no clipping)
processed_action             (T, A)       per-term processed actions after scale/offset
                                          (omitted if any action term does not expose them)
root_pos_w                   (T, 3)       root link position after step t, world
root_quat_w                  (T, 4)       root link quaternion (wxyz) after step t, world
root_lin_vel_w               (T, 3)       root CoM linear velocity after step t, world
root_link_lin_vel_w          (T, 3)       root link-origin linear velocity after step t, world
root_ang_vel_w               (T, 3)       root angular velocity after step t, world
joint_pos                    (T, J)       joint positions after step t (Isaac order)
joint_vel                    (T, J)       joint velocities after step t (Isaac order)
applied_torque               (T, J)       applied joint torques at the last physics sub-step
terminated                   (T,)         termination flag returned by step t
truncated                    (T,)         time-out flag returned by step t
contact_forces_w             (T, C, 3)    net contact forces, world (only if sensor exists)
contact_flag                 (T, C)       force norm > 1.0 N (only if sensor exists)
contact_body_names           (C,)         contact-sensor body names (only if sensor exists)
record_physics_steps         ()           whether --record_physics_steps was requested
contact_forces_w_phys        (T*S, C, 3)  net contact forces at PHYSICS rate, world (only with
                                          --record_physics_steps and a contact sensor);
                                          row ``t*S + k`` is physics sub-step ``k`` of policy
                                          step ``t``, so row ``(t+1)*S - 1`` equals
                                          ``contact_forces_w[t]``
contact_phys_per_policy      ()           S: physics rows per policy step. Normally
                                          ``decimation``; 1 on fallback to policy-rate
                                          recording (warned)
terrain_height_grid          (H, W)       measured terrain heights (generator terrains only):
                                          node (i, j) is the surface under
                                          ``terrain_grid_origin_xy + (i, j) *
                                          terrain_grid_resolution`` (i along x, j along y),
                                          from a vertical warp raycast of the actual terrain
                                          mesh over the full extent (all tiles + border) at
                                          the generator's ``horizontal_scale``. float32.
terrain_grid_origin_xy       (2,)         world xy of grid node (0, 0)
terrain_grid_resolution      ()           grid node spacing [m] (= horizontal_scale)
terrain_tile_origins         (R, C, 3)    sub-terrain (tile) origins, world frame
terrain_env_tile             (2,)         (row, col) of the tile env 0 spawned on
terrain_mesh_vertices        (V, 3)       terrain mesh vertices, world frame, float32 (the
                                          exact mesh IsaacLab's RayCaster casts against)
terrain_mesh_faces           (F, 3)       terrain mesh triangles (int32 vertex indices)
===========================  ===========  ====================================================

The ``terrain_*`` keys exist only for generator-type terrains; plane terrains
record nothing new. The mesh is the authoritative record (vertical faces such
as stair risers survive it exactly); the height grid is its raycast at the
generator lattice, ready for a MuJoCo hfield.

State at index ``t`` is the state after applying ``action_raw[t]`` for
``decimation`` physics steps; ``obs[t]`` is the observation the policy consumed
to produce ``action_raw[t]``. A ``terminated``/``truncated`` True at step ``t``
means the env auto-reset before the state recorded at ``t+1``.

Granularity with ``--record_physics_steps``: ONLY the contact forces are
recorded at the physics rate — IsaacLab steps physics ``decimation`` times
inside ``env.step()`` with no user hook in between, and ``ArticulationData``
keeps no per-physics-step history, so mid-step joint/root states and applied
torques are not observable from outside ``env.step()``. The eagerly-updated
``ContactSensor`` history is the one exception; see
``_enable_physics_rate_contact_history`` for the mechanism. Enlarging the
sensor history can affect history-based termination terms (e.g.
``illegal_contact`` maxes over the window); that never changes the physics and
only matters if a termination fires, which already invalidates the reference
(a warning is printed on any reset).
"""

# Isaac Sim must launch before any isaaclab import (see CLAUDE.md: Import Ordering).
import argparse

from lab2mj.isaac.dump_common import launch_app

parser = argparse.ArgumentParser(description="Dump a deterministic Isaac reference trajectory for sim2sim validation.")
parser.add_argument("--task", type=str, required=True, help="Manager-based task id (e.g. Isaac-Velocity-Flat-G1-v0).")
parser.add_argument("--policy", type=str, required=True, help="Path to a TorchScript policy.pt (exported jit module).")
parser.add_argument("--num_steps", type=int, default=400, help="Number of policy steps to record.")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=0,
    help=(
        "Policy steps to run (unrecorded) after reset before recording starts. The recorded init state is taken "
        "after settling, so the reference trajectory begins from a dynamically consistent contact state instead of "
        "the spawn pose (PhysX resolves spawn-pose ground penetration positionally, which other engines cannot "
        "reproduce)."
    ),
)
parser.add_argument(
    "--command",
    type=str,
    default="1.0,0.0,0.0",
    help="Pinned base-frame velocity command 'vx,vy,wz'. If the task's velocity command term has "
    "heading_command=True, it is set to False and the ang_vel_z range is pinned to wz instead of "
    "tracking a sampled heading.",
)
parser.add_argument(
    "--pose_command",
    type=str,
    default=None,
    help="For pose-command tasks: pinned 'x,y,z,heading'. z is applied only if the term's ranges define pos_z.",
)
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--out", type=str, required=True, help="Output npz path.")
parser.add_argument(
    "--strict",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable obs corruption on all groups, replace all events with a deterministic "
    "reset-to-default term, disable curriculum, and pin commands.",
)
parser.add_argument(
    "--record_physics_steps",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Additionally record per-physics-step net contact forces (contact_forces_w_phys) by forcing the "
    "contact-sensor history to cover one full decimation window. Joint/root states remain per-policy-step: "
    "IsaacLab exposes no mid-step articulation state.",
)
args_cli, simulation_app = launch_app(parser)

import os
from typing import Any, cast

import contact_lab.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import reset_scene_to_default
from isaaclab.managers import CommandTermCfg, EventTermCfg, ObservationGroupCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.utils import parse_env_cfg

from lab2mj.isaac.dump_common import append_state, plant_record, shutdown


def _parse_floats(spec: str, n: int, name: str) -> list[float]:
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != n:
        raise ValueError(f"--{name} expects {n} comma-separated floats, got {spec!r}")
    return parts


def _pin_command_terms(commands_cfg: object, vel_cmd: list[float], pose_cmd: list[float] | None) -> None:
    """Pin every command term to a single deterministic value.

    Velocity terms (ranges with ``lin_vel_x``) are pinned to ``vel_cmd``; pose terms
    (ranges with ``pos_x``) are pinned to ``pose_cmd``. All terms get an effectively
    infinite resampling interval and no standing envs.
    """
    for term_name, term_cfg in vars(commands_cfg).items():
        if not isinstance(term_cfg, CommandTermCfg):
            continue
        term_cfg.resampling_time_range = (1.0e9, 1.0e9)
        if hasattr(term_cfg, "rel_standing_envs"):
            setattr(term_cfg, "rel_standing_envs", 0.0)
        ranges = getattr(term_cfg, "ranges", None)
        if ranges is None:
            continue
        if hasattr(ranges, "lin_vel_x"):
            vx, vy, wz = vel_cmd
            setattr(ranges, "lin_vel_x", (vx, vx))
            if hasattr(ranges, "lin_vel_y"):
                setattr(ranges, "lin_vel_y", (vy, vy))
            if getattr(term_cfg, "heading_command", False):
                setattr(term_cfg, "heading_command", False)
                print(f"[INFO] Command '{term_name}': heading_command disabled; pinning ang_vel_z = {wz:+.3f}.")
            if hasattr(ranges, "ang_vel_z"):
                setattr(ranges, "ang_vel_z", (wz, wz))
        elif hasattr(ranges, "pos_x"):
            if pose_cmd is None:
                raise ValueError(f"Command term '{term_name}' is a pose command; pass --pose_command 'x,y,z,heading'.")
            x, y, z, heading = pose_cmd
            setattr(ranges, "pos_x", (x, x))
            if hasattr(ranges, "pos_y"):
                setattr(ranges, "pos_y", (y, y))
            if hasattr(ranges, "pos_z"):
                setattr(ranges, "pos_z", (z, z))
            if hasattr(ranges, "heading"):
                setattr(ranges, "heading", (heading, heading))
            if hasattr(term_cfg, "simple_heading"):
                setattr(term_cfg, "simple_heading", False)
        else:
            print(
                f"[WARN] Command term '{term_name}' ({type(term_cfg).__name__}) is neither a velocity nor a pose "
                "command; only its resampling is frozen, so its value is a seed-dependent sample. The npz key "
                f"'command_{term_name}' records the command actually applied."
            )


# Deterministic startup terms that define the plant (contact_lab), kept in strict mode.
_PLANT_EVENT_FUNCS = ("set_legacy_joint_friction", "check_joint_friction")


def _apply_strict_overrides(env_cfg: Any, vel_cmd: list[float], pose_cmd: list[float] | None) -> None:
    """Make the env deterministic: no corruption, reset-to-default only, no curriculum, pinned commands."""
    for group_cfg in vars(env_cfg.observations).values():
        if isinstance(group_cfg, ObservationGroupCfg):
            group_cfg.enable_corruption = False
    # Observation groups embedded in action terms (e.g. PreTrainedPolicyActionCfg's
    # low_level_observations) feed frozen policies and must be noise-free too.
    for term_cfg in vars(env_cfg.actions).values():
        if term_cfg is None or not hasattr(term_cfg, "__dict__"):
            continue
        for attr in vars(term_cfg).values():
            if isinstance(attr, ObservationGroupCfg):
                attr.enable_corruption = False
    # Replace every event term with a single deterministic reset-to-default term. Spawning
    # alone does not write the articulation cfg's default joint state to sim (only reset
    # events or scene.reset_to do), so without this term the robot would start from the
    # USD-authored joint pose instead of the documented default init state.
    if env_cfg.events is not None:
        for term_name, term_cfg in list(vars(env_cfg.events).items()):
            if isinstance(term_cfg, EventTermCfg) and getattr(term_cfg.func, "__name__", "") not in _PLANT_EVENT_FUNCS:
                setattr(env_cfg.events, term_name, None)
    else:
        env_cfg.events = type("StrictEventsCfg", (), {})()
    setattr(env_cfg.events, "reset_scene_to_default", EventTermCfg(func=reset_scene_to_default, mode="reset"))
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum = None
    if env_cfg.commands is not None:
        _pin_command_terms(env_cfg.commands, vel_cmd, pose_cmd)


def _enable_physics_rate_contact_history(scene_cfg: object, decimation: int) -> None:
    """Make every contact sensor retain one full decimation window of per-physics-step forces.

    ``ContactSensorData.net_forces_w_history`` rolls once per physics sub-step as long as
    ``history_length > 0`` (SensorBase.update recomputes eagerly, bypassing lazy evaluation)
    and the sensor is due (``update_period`` no larger than the physics dt). Forcing
    ``history_length >= decimation`` and ``update_period = 0.0`` guarantees that after each
    ``env.step()`` the newest ``decimation`` history entries are exactly the sub-steps of
    that policy step.
    """
    for name, cfg in vars(scene_cfg).items():
        if isinstance(cfg, ContactSensorCfg):
            new_len = max(cfg.history_length, decimation)
            if cfg.history_length != new_len or cfg.update_period != 0.0:
                print(
                    f"[INFO] Contact sensor '{name}': history_length {cfg.history_length} -> {new_len}, "
                    f"update_period {cfg.update_period} -> 0.0 (physics-rate force recording)."
                )
            cfg.history_length = new_len
            cfg.update_period = 0.0


def _capture_measured_terrain(menv: Any) -> dict[str, np.ndarray]:
    """Measured-terrain record for generator terrains; {} for plane/absent terrain.

    Records the exact terrain triangle mesh (the USD prim IsaacLab's RayCaster
    casts against, in world frame) plus a regular height grid: one vertical warp
    raycast per node at the generator's ``horizontal_scale`` over the full
    generated extent (all tiles + border). Grid nodes on the outer boundary sit
    exactly on the mesh edge, where a ray can miss; those are re-cast nudged 1 um
    toward the terrain center. Any remaining miss is an error.
    """
    terrain = getattr(menv.scene, "terrain", None)
    if terrain is None:
        return {}
    cfg = terrain.cfg
    if getattr(cfg, "terrain_type", "plane") != "generator":
        return {}

    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
    from pxr import UsdGeom

    prim_path = terrain.terrain_prim_paths[0]
    mesh_prim = sim_utils.get_first_matching_child_prim(prim_path, lambda prim: prim.GetTypeName() == "Mesh")
    if mesh_prim is None:
        raise RuntimeError(f"no Mesh prim found under terrain prim {prim_path!r}")
    mesh_prim = UsdGeom.Mesh(mesh_prim)
    points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float64)
    transform = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
    points = points @ transform[:3, :3].T + transform[:3, 3]
    counts = np.asarray(mesh_prim.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise RuntimeError(f"terrain mesh has non-triangle faces (counts {np.unique(counts)}); cannot record it")
    faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    # float32 vertices: exactly what convert_to_warp_mesh hands the RayCaster's warp mesh.
    points32 = points.astype(np.float32)

    gen_cfg = cfg.terrain_generator
    resolution = float(gen_cfg.horizontal_scale)
    half_x = 0.5 * int(gen_cfg.num_rows) * float(gen_cfg.size[0]) + float(gen_cfg.border_width)
    half_y = 0.5 * int(gen_cfg.num_cols) * float(gen_cfg.size[1]) + float(gen_cfg.border_width)
    num_x = int(round(2.0 * half_x / resolution)) + 1
    num_y = int(round(2.0 * half_y / resolution)) + 1
    grid_x, grid_y = np.meshgrid(
        -half_x + resolution * np.arange(num_x), -half_y + resolution * np.arange(num_y), indexing="ij"
    )
    top_z = float(points32[:, 2].max()) + 1.0
    max_dist = top_z - float(points32[:, 2].min()) + 1.0
    starts = np.stack([grid_x, grid_y, np.full_like(grid_x, top_z)], axis=-1).reshape(-1, 3)

    wp_mesh = convert_to_warp_mesh(points32, faces, device=menv.device)

    def cast(starts_np: np.ndarray) -> np.ndarray:
        starts_t = torch.tensor(starts_np, dtype=torch.float32, device=menv.device)
        dirs_t = torch.zeros_like(starts_t)
        dirs_t[:, 2] = -1.0
        hits = raycast_mesh(starts_t, dirs_t, wp_mesh, max_dist=max_dist)[0]
        return hits[:, 2].cpu().numpy().astype(np.float64)

    heights = cast(starts)
    miss = ~np.isfinite(heights)
    if miss.any():
        nudged = starts[miss].copy()
        nudged[:, 0] -= np.sign(nudged[:, 0]) * 1.0e-6
        nudged[:, 1] -= np.sign(nudged[:, 1]) * 1.0e-6
        heights[miss] = cast(nudged)
        still = ~np.isfinite(heights)
        if still.any():
            raise RuntimeError(f"{int(still.sum())}/{heights.size} height-grid nodes missed the terrain mesh")

    record: dict[str, np.ndarray] = {
        "terrain_height_grid": heights.reshape(num_x, num_y).astype(np.float32),
        "terrain_grid_origin_xy": np.array([-half_x, -half_y], dtype=np.float64),
        "terrain_grid_resolution": np.array(resolution, dtype=np.float64),
        "terrain_mesh_vertices": points32,
        "terrain_mesh_faces": faces,
    }
    if terrain.terrain_origins is not None:
        record["terrain_tile_origins"] = terrain.terrain_origins.cpu().numpy().astype(np.float32)
    if getattr(terrain, "terrain_levels", None) is not None and getattr(terrain, "terrain_types", None) is not None:
        record["terrain_env_tile"] = np.array(
            [int(terrain.terrain_levels[0].item()), int(terrain.terrain_types[0].item())], dtype=np.int64
        )
    print(
        f"[INFO] Measured terrain recorded: mesh {points32.shape[0]} vertices / {faces.shape[0]} triangles, "
        f"height grid {num_x} x {num_y} @ {resolution} m."
    )
    return record


def _policy_obs(obs_dict: Any) -> torch.Tensor:
    """Extract the 'policy' group as a single (num_envs, O) tensor."""
    group = obs_dict["policy"]
    if isinstance(group, dict):
        group = torch.cat([v.reshape(v.shape[0], -1) for v in group.values()], dim=-1)
    return cast(torch.Tensor, group)


def main() -> None:
    vel_cmd = _parse_floats(args_cli.command, 3, "command")
    pose_cmd = _parse_floats(args_cli.pose_command, 4, "pose_command") if args_cli.pose_command else None

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"Task '{args_cli.task}' is not a manager-based RL task: {type(env_cfg).__name__}")
    env_cfg = cast(Any, env_cfg)
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg, "export_io_descriptors"):
        env_cfg.export_io_descriptors = False
    if args_cli.strict:
        _apply_strict_overrides(env_cfg, vel_cmd, pose_cmd)
    if args_cli.record_physics_steps:
        _enable_physics_rate_contact_history(env_cfg.scene, int(env_cfg.decimation))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    menv = cast(Any, env.unwrapped)
    robot = menv.scene["robot"]

    measured_terrain = _capture_measured_terrain(menv)

    policy = torch.jit.load(args_cli.policy, map_location=menv.device)
    policy.eval()

    obs_dict, _ = env.reset()

    # Settle: run the policy unrecorded so the recorded trajectory starts from a
    # dynamically consistent contact state (no spawn-pose ground penetration).
    for _ in range(args_cli.settle_steps):
        with torch.no_grad():
            settle_action = policy(_policy_obs(obs_dict))
        settle_out = cast(tuple[Any, Any, torch.Tensor, torch.Tensor, Any], env.step(settle_action))
        obs_dict, _, terminated_t, truncated_t, _ = settle_out
        if bool(terminated_t[0]) or bool(truncated_t[0]):
            raise RuntimeError("episode ended during --settle_steps; reduce settle length or check the policy")

    # Optional contact sensor: first scene sensor that exposes net contact forces.
    contact_sensor = None
    contact_body_names: list[str] = []
    for sensor in menv.scene.sensors.values():
        if getattr(sensor.data, "net_forces_w", None) is not None:
            contact_sensor = sensor
            contact_body_names = list(sensor.body_names)
            break

    # Physics-rate contact recording: S history rows per policy step (newest-first in the
    # sensor buffer). S == decimation when the sensor history covers the decimation window,
    # else fall back to policy rate (S == 1).
    decimation = int(menv.cfg.decimation)
    contact_phys_per_policy = 0
    if args_cli.record_physics_steps:
        if contact_sensor is None:
            print("[WARN] --record_physics_steps requested but no contact sensor found; skipping.")
        else:
            history = contact_sensor.data.net_forces_w_history
            if history is not None and history.shape[1] >= decimation:
                contact_phys_per_policy = decimation
            else:
                hist_len = 0 if history is None else int(history.shape[1])
                contact_phys_per_policy = 1
                print(
                    f"[WARN] Contact-sensor history ({hist_len}) does not cover the decimation window "
                    f"({decimation}); 'contact_forces_w_phys' falls back to PER-POLICY-STEP forces "
                    "(contact_phys_per_policy = 1)."
                )

    d = robot.data
    init_state = {
        "init_root_pos_w": d.root_pos_w[0].cpu().numpy().copy(),
        "init_root_quat_w": d.root_quat_w[0].cpu().numpy().copy(),
        "init_root_lin_vel_w": d.root_lin_vel_w[0].cpu().numpy().copy(),
        "init_root_link_lin_vel_w": d.root_link_lin_vel_w[0].cpu().numpy().copy(),
        "init_root_ang_vel_w": d.root_ang_vel_w[0].cpu().numpy().copy(),
        "init_joint_pos": d.joint_pos[0].cpu().numpy().copy(),
        "init_joint_vel": d.joint_vel[0].cpu().numpy().copy(),
        # Raw policy action from the last settle step (zeros without settling); seeds the
        # last_action obs term so obs parity holds at the first recorded step.
        "init_last_action": menv.action_manager.action[0].cpu().numpy().copy(),
        # Episode clock at recording start (== settle_steps unless a reset fired); the
        # MuJoCo side mirrors it so the timeout clock and the PreTrainedPolicyAction
        # actions-slot zeroing (episode_length_buf == 0) line up.
        "init_episode_step": np.array(int(menv.episode_length_buf[0].item())),
    }
    # DelayedPD groups sample their delay lag per env at reset (randint(min_delay,
    # max_delay + 1)); the replay must serve targets at the same lag or every PD
    # torque leads/trails Isaac's by the sampled number of physics steps.
    for act_name, act in robot.actuators.items():
        delay_buf = getattr(act, "positions_delay_buffer", None)
        if delay_buf is not None:
            init_state[f"init_actuator_lag_{act_name}"] = np.array(int(delay_buf.time_lags[0].item()))
    # PreTrainedPolicyAction's low-level action buffer at recording start: after a
    # settle phase Isaac serves the real settled low-level action in the remapped
    # 'actions' obs slot (episode_length_buf > 0), which the MuJoCo side must seed.
    ll_action_term = None
    for term_name in menv.action_manager.active_terms:
        term = menv.action_manager.get_term(term_name)
        if getattr(term, "low_level_actions", None) is not None:
            ll_action_term = term
            init_state["init_low_level_action"] = term.low_level_actions[0].detach().cpu().numpy().copy()
            break
    obs0_t = _policy_obs(obs_dict)
    obs0 = obs0_t[0].cpu().numpy().copy()

    step_keys = (
        "obs",
        "action_raw",
        "processed_action",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_link_lin_vel_w",
        "root_ang_vel_w",
        "joint_pos",
        "joint_vel",
        "applied_torque",
        "terminated",
        "truncated",
        "contact_forces_w",
    )
    rec: dict[str, list[np.ndarray]] = {k: [] for k in step_keys}
    rec["contact_forces_w_phys"] = []
    processed_available = True

    # Ground-truth per-term commands: the 'command' npz key only stores the CLI request,
    # which need not match terms that _pin_command_terms could not pin (or --no-strict runs).
    command_manager: Any = getattr(menv, "command_manager", None)
    cmd_term_names: list[str] = list(command_manager.active_terms) if command_manager is not None else []
    for name in cmd_term_names:
        rec[f"command_{name}"] = []

    # Resolved world-frame pose goals as of the post-reset/settle state. The raw
    # --pose_command is NOT the world goal: the term offsets it by the env origin and
    # default root z (UniformPose2dCommand) or replaces it with a flat-patch sample
    # (TerrainBasedPose2dCommand). The MuJoCo side pins this world goal directly.
    pose_goals_w: dict[str, np.ndarray] = {}
    for name in cmd_term_names:
        term = command_manager.get_term(name)
        pos_w = getattr(term, "pos_command_w", None)
        heading_w = getattr(term, "heading_command_w", None)
        if pos_w is not None and heading_w is not None:
            goal = np.concatenate([pos_w[0].cpu().numpy(), [float(heading_w[0].item())]])
            pose_goals_w[f"command_{name}_goal_w"] = goal.astype(np.float64)

    def record_state(terminated: bool, truncated: bool) -> None:
        append_state(rec, robot.data)
        rec["applied_torque"].append(robot.data.applied_torque[0].cpu().numpy().copy())
        rec["terminated"].append(np.array(terminated))
        rec["truncated"].append(np.array(truncated))
        if contact_sensor is not None:
            rec["contact_forces_w"].append(contact_sensor.data.net_forces_w[0].cpu().numpy().copy())
            if contact_phys_per_policy > 1:
                # History is most-recent-first; take this policy step's sub-steps, oldest first.
                hist = contact_sensor.data.net_forces_w_history[0, :contact_phys_per_policy]
                rec["contact_forces_w_phys"].append(hist.flip(0).cpu().numpy().copy())
            elif contact_phys_per_policy == 1:
                rec["contact_forces_w_phys"].append(contact_sensor.data.net_forces_w[0].cpu().numpy().copy()[None])

    num_resets = 0
    for _ in range(args_cli.num_steps):
        obs_t = _policy_obs(obs_dict)
        with torch.no_grad():
            action = policy(obs_t)
        rec["obs"].append(obs_t[0].cpu().numpy().copy())
        rec["action_raw"].append(action[0].cpu().numpy().copy())
        for name in cmd_term_names:
            rec[f"command_{name}"].append(command_manager.get_command(name)[0].cpu().numpy().copy())
        if ll_action_term is not None:
            # Last low-level fire's raw action before this policy step's decimation loop
            # runs — the value the sim2sim low-level parity metric compares against.
            # The buffer holds the frozen policy's output, which still requires grad.
            rec.setdefault("ll_action", []).append(ll_action_term.low_level_actions[0].detach().cpu().numpy().copy())

        # ManagerBasedRLEnv returns per-env tensors, not the scalar bools gymnasium's stubs declare.
        step_out = cast(tuple[Any, Any, torch.Tensor, torch.Tensor, Any], env.step(action))
        obs_dict, _, terminated_t, truncated_t, _ = step_out

        if processed_available:
            try:
                mgr = menv.action_manager
                processed = torch.cat(
                    [mgr.get_term(name).processed_actions.reshape(1, -1) for name in mgr.active_terms], dim=-1
                )
                rec["processed_action"].append(processed[0].cpu().numpy().copy())
            except NotImplementedError:
                processed_available = False
                rec["processed_action"].clear()
                print("[WARN] An action term does not expose processed_actions; omitting 'processed_action'.")

        terminated = bool(terminated_t[0].item())
        truncated = bool(truncated_t[0].item())
        record_state(terminated, truncated)
        if terminated or truncated:
            num_resets += 1

    if num_resets > 0:
        print(f"[WARN] Env reset {num_resets} time(s) during the rollout; the reference is not a single episode.")

    arrays: dict[str, np.ndarray] = {
        "task": np.array(args_cli.task),
        "policy_path": np.array(os.path.abspath(args_cli.policy)),
        "seed": np.array(args_cli.seed),
        "strict": np.array(bool(args_cli.strict)),
        "command": np.array(vel_cmd, dtype=np.float64),
        **plant_record(robot),
        "physics_dt": np.array(menv.cfg.sim.dt),
        "decimation": np.array(menv.cfg.decimation),
        "episode_length_s": np.array(menv.cfg.episode_length_s),
        "env_origin_w": menv.scene.env_origins[0].cpu().numpy(),
        "gravity_w": np.array(menv.cfg.sim.gravity, dtype=np.float64),
        "obs0": obs0,
        "settle_steps": np.array(args_cli.settle_steps),
        **init_state,
    }
    if pose_cmd is not None:
        arrays["pose_command"] = np.array(pose_cmd, dtype=np.float64)
    for key in step_keys:
        if key == "processed_action" and not processed_available:
            continue
        if key == "contact_forces_w" and contact_sensor is None:
            continue
        arrays[key] = np.stack(rec[key])
    for name in cmd_term_names:
        arrays[f"command_{name}"] = np.stack(rec[f"command_{name}"])
    if "ll_action" in rec:
        arrays["ll_action"] = np.stack(rec["ll_action"])
    arrays.update(pose_goals_w)
    if contact_sensor is not None:
        arrays["contact_flag"] = np.linalg.norm(arrays["contact_forces_w"], axis=-1) > 1.0
        arrays["contact_body_names"] = np.array(contact_body_names)
    arrays.update(measured_terrain)
    arrays["record_physics_steps"] = np.array(bool(args_cli.record_physics_steps))
    if contact_phys_per_policy > 0:
        arrays["contact_forces_w_phys"] = np.concatenate(rec["contact_forces_w_phys"], axis=0)
        arrays["contact_phys_per_policy"] = np.array(contact_phys_per_policy)

    out_dir = os.path.dirname(os.path.abspath(args_cli.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args_cli.out, **arrays)
    print(f"[INFO] Dumped {args_cli.num_steps} policy steps for '{args_cli.task}' to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    shutdown(simulation_app)
