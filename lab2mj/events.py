"""Domain-randomization events for the MuJoCo runtime, mirroring isaaclab.envs.mdp.events.

Single-env numpy ports of the IsaacLab v2.3.0 event terms used by the velocity tasks.
Semantics notes vs Isaac/PhysX:

* ``randomize_rigid_body_material``: Isaac samples ``num_buckets`` (static, dynamic,
  restitution) materials once, then assigns a random bucket per collision shape. PhysX
  combines the bucket's static friction with the terrain material per the terrain's
  ``friction_combine_mode`` (:func:`lab2mj.terrain.pair_friction` — the same
  rule the scene builder uses for the un-randomized pair friction). MuJoCo has a single
  sliding-friction coefficient per contact, so we write the pair STATIC value into the
  robot geom's ``friction[0]``: it reproduces PhysX's no-slip breakaway threshold
  exactly (the dominant regime for stance feet), while the force during sustained slip
  follows the static instead of the pair dynamic value — an engine capability gap that
  a construction-time warning flags whenever the bucket columns can differ. The scene
  builder must make the robot geom win MuJoCo's pair combination (give robot geoms
  higher ``priority`` than the terrain, or author terrain friction below the smallest
  bucket value) — which is why the converter disables the self-collision ``robot_self_pair``
  friction scheme (terrain at priority 2) whenever this event exists. Restitution is
  ignored: MuJoCo contacts are solref-based and both robots use restitution 0.
* ``randomize_rigid_body_mass``: operates on *default* masses (repeat applications do
  not compound) and rescales the diagonal body inertia by the mass ratio
  (``recompute_inertia``), exactly like Isaac. Samples are drawn per matched *Isaac*
  body (RNG draw-count parity with Isaac); ``add`` deltas aggregate exactly onto a
  welded (merged) MuJoCo body, while ``scale``/``abs`` on a merged body would need the
  per-link mass split the bundle does not carry and raise instead.
* ``apply_external_force_torque``: Isaac holds the sampled *link-frame* wrench and
  PhysX re-applies it in the link frame at every step; the stored wrenches here are
  re-rotated into ``xfrc_applied`` before every MuJoCo step for the same behavior
  (:meth:`EventSet.refresh_external_wrenches`). A nonzero wrench on a body welded away
  by the converter has no tracked frame and raises.
* Root velocities follow Isaac's ``write_root_velocity_to_sim``: the sampled 6-vector is
  the root **CoM** velocity in world frame. The MuJoCo free joint stores
  ``[lin vel of body-frame origin (world), ang vel (body frame)]``, so we convert.
* ``randomize_joint_default_pos`` (contact_lab encoder calibration bias): offsets the default
  joint positions in place; the runtime shifts the joint-position action offset by
  :attr:`EventSet.joint_default_pos_offset` after the startup events, as the Isaac term does.
* contact_lab's ``check_joint_friction`` (an Isaac-side config check) is accepted and does
  nothing at runtime.
* All events no-op in strict mode.

``RobotMap`` is the runtime-provided addressing/default-state bundle documented on the
dataclass. Interval clocks mirror the EventManager: resampled on episode reset, ticked by
``step_interval`` at the policy rate, applied when the clock crosses zero.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from lab2mj.actuators import resolve_matching_names
from lab2mj.env_yaml import class_name
from lab2mj.ir import EventIR
from lab2mj.quat import quat_from_euler_xyz, quat_mul, quat_to_rotmat
from lab2mj.terrain import pair_friction

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass
class RobotMap:
    """MuJoCo addressing + default state for the converted robot (runtime provides).

    Body dictionaries are keyed by *Isaac* body name in Isaac (BFS) body order. Because
    usd2mjcf folds welded USD bodies into their parent, several Isaac names may map to
    the same MuJoCo body id (multimap).
    """

    body_ids: dict[str, list[int]]
    """Isaac body name -> MuJoCo body id(s)."""
    geom_ids: dict[str, list[int]]
    """Isaac body name -> MuJoCo collision geom ids."""
    root_body_id: int
    root_qpos_adr: int
    """qpos address of the free joint (7 numbers: pos + wxyz quat)."""
    root_dof_adr: int
    """qvel address of the free joint (6 numbers)."""
    joint_names: list[str]
    """Actuated joint names in Isaac order."""
    qpos_adr: np.ndarray
    """(J,) qpos address per joint, Isaac order."""
    dof_adr: np.ndarray
    """(J,) qvel/dof address per joint, Isaac order."""
    default_joint_pos: np.ndarray
    default_joint_vel: np.ndarray
    joint_pos_limits: np.ndarray
    """(J, 2) *soft* joint position limits (Isaac clamps reset samples to these)."""
    joint_vel_limits: np.ndarray
    """(J,) soft joint velocity limits."""
    default_root_pose_env: np.ndarray
    """(7,) env-local default root position + wxyz quaternion (Isaac init_state)."""
    default_root_vel_w: np.ndarray
    """(6,) default root CoM lin/ang velocity in world frame."""
    env_origin_w: np.ndarray
    """(3,) env origin translation."""
    default_body_mass: np.ndarray
    """(nbody,) pristine model.body_mass."""
    default_body_inertia: np.ndarray
    """(nbody, 3) pristine model.body_inertia (principal diagonal)."""
    default_body_ipos: np.ndarray
    """(nbody, 3) pristine model.body_ipos (CoM offset in body frame)."""


def resolve_names(patterns: Any, names: list[str] | Any) -> list[str]:
    """Resolve regex pattern(s) against ``names``, preserving the order of ``names``.

    Mirrors IsaacLab's ``resolve_matching_names`` for the non-``preserve_order``
    case via :func:`lab2mj.actuators.resolve_matching_names` (same contract:
    a pattern matching nothing raises — a typo'd name would otherwise silently
    shrink the event's target set — and a name matched by more than one pattern
    raises); ``None`` matches everything.
    """
    if patterns is None:
        return list(names)
    if isinstance(patterns, str):
        patterns = [patterns]
    names = list(names)
    return [names[i] for i in resolve_matching_names(list(patterns), names)]


# SceneEntityCfg id-based selectors have no name -> id map on the runtime side; a cfg
# authored with any of them would otherwise silently resolve as match-ALL below.
_UNSUPPORTED_SELECTORS = (
    "body_ids",
    "joint_ids",
    "fixed_tendon_names",
    "fixed_tendon_ids",
    "object_collection_names",
    "object_collection_ids",
)


def check_scene_entity_cfg(cfg: Any, label: str, *, entity: str | None = "robot", check_selectors: bool = True) -> None:
    """Reject ``SceneEntityCfg`` forms the runtime does not implement, loudly.

    Mirrors the obs pipeline's construction-time checks: the resolvers below read
    only name-based selectors on the robot entity, so an id-based selector or a
    different scene entity must raise instead of silently applying to every robot
    body/joint. ``entity=None`` skips the entity-name check (sensor cfgs resolve
    their name against the manifest's contact sensors instead);
    ``check_selectors=False`` skips the selector checks for terms whose Isaac
    implementation ignores them (``reset_joints_around_default``).
    """
    cfg = cfg or {}
    name = cfg.get("name")
    if entity is not None and name not in (None, entity):
        raise ValueError(f"{label}: targets scene entity '{name}'; only '{entity}' is supported")
    if not check_selectors:
        return
    for key in _UNSUPPORTED_SELECTORS:
        if cfg.get(key) is not None:
            raise ValueError(f"{label}: SceneEntityCfg '{key}' selectors are not supported (author name patterns)")
    if cfg.get("preserve_order"):
        raise ValueError(f"{label}: SceneEntityCfg preserve_order is not supported")


def resolve_body_ids(patterns: Any, body_map: dict[str, list[int]]) -> list[int]:
    """Resolve body-name regex pattern(s) to deduplicated MuJoCo body ids.

    ``body_map`` is the multimap ``isaac body name -> mj body id(s)`` in Isaac body
    order; ``None`` matches every body; welded Isaac bodies sharing an mj body count
    once. Pattern semantics (including the raises) come from :func:`resolve_names`.
    """
    ids: list[int] = []
    for name in resolve_names(patterns, list(body_map)):
        for bid in body_map[name]:
            if bid not in ids:
                ids.append(bid)
    return ids


def _matched_body_names(params: dict[str, Any], robot_map: RobotMap) -> list[str]:
    asset_cfg = params.get("asset_cfg") or {}
    return resolve_names(asset_cfg.get("body_names"), list(robot_map.body_ids))


def _matched_body_ids(params: dict[str, Any], robot_map: RobotMap) -> list[int]:
    asset_cfg = params.get("asset_cfg") or {}
    return resolve_body_ids(asset_cfg.get("body_names"), robot_map.body_ids)


def _merged_body_ids(robot_map: RobotMap) -> set[int]:
    """MuJoCo body ids that carry more than one Isaac body (weld merges)."""
    seen: set[int] = set()
    merged: set[int] = set()
    for ids in robot_map.body_ids.values():
        for bid in ids:
            (merged if bid in seen else seen).add(bid)
    return merged


def _matched_joint_indices(params: dict[str, Any], robot_map: RobotMap) -> np.ndarray:
    asset_cfg = params.get("asset_cfg") or {}
    matched = resolve_names(asset_cfg.get("joint_names"), robot_map.joint_names)
    index = {n: i for i, n in enumerate(robot_map.joint_names)}
    return np.array([index[n] for n in matched], dtype=np.int64)


def _sample_range_dict(rng: np.random.Generator, ranges: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    """Sample one uniform value per key; missing keys are (0, 0), exactly like Isaac."""
    lo = np.array([(ranges.get(k) or (0.0, 0.0))[0] for k in keys], dtype=np.float64)
    hi = np.array([(ranges.get(k) or (0.0, 0.0))[1] for k in keys], dtype=np.float64)
    return rng.uniform(lo, hi)


def _sample_distribution(rng: np.random.Generator, params: tuple[float, float], size: Any, distribution: str):
    if distribution == "uniform":
        return rng.uniform(params[0], params[1], size)
    if distribution == "log_uniform":
        return np.exp(rng.uniform(np.log(params[0]), np.log(params[1]), size))
    if distribution == "gaussian":
        return rng.normal(params[0], params[1], size)
    raise ValueError(f"unknown distribution '{distribution}' (use 'uniform', 'log_uniform', 'gaussian')")


def _refresh_model_constants(model: mujoco.MjModel) -> None:
    """Refresh mass/inertia-derived model constants after a mass / CoM event.

    ``mj_setConst`` evaluates the model at ``qpos0`` and leaves that configuration in the
    MjData it is given, so it runs on scratch data: on the live data it would teleport
    the robot to ``qpos0`` (the env origin) mid-episode.
    """
    mujoco.mj_setConst(model, mujoco.MjData(model))


def write_root_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_map: RobotMap,
    pos_w: np.ndarray,
    quat_wxyz: np.ndarray,
    vel_com_w: np.ndarray,
) -> None:
    """Write a world root pose + CoM velocity into the free joint (world->body conversion).

    ``vel_com_w`` is ``[lin, ang]`` of the root CoM in world frame (Isaac convention).
    MuJoCo free-joint qvel is ``[lin vel of the body-frame origin in world, ang vel in
    body frame]``: ``v_link_w = v_com_w - w_w x (R_wb @ ipos_b)`` and
    ``w_b = R_wb^T @ w_w``.
    """
    qa, da = robot_map.root_qpos_adr, robot_map.root_dof_adr
    data.qpos[qa : qa + 3] = pos_w
    data.qpos[qa + 3 : qa + 7] = quat_wxyz
    rot_wb = quat_to_rotmat(np.asarray(quat_wxyz, dtype=np.float64))
    ang_vel_w = np.asarray(vel_com_w[3:6], dtype=np.float64)
    ipos_b = model.body_ipos[robot_map.root_body_id]
    data.qvel[da : da + 3] = vel_com_w[0:3] - np.cross(ang_vel_w, rot_wb @ ipos_b)
    data.qvel[da + 3 : da + 6] = rot_wb.T @ ang_vel_w


def read_root_com_vel_w(model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap) -> np.ndarray:
    """Root CoM ``[lin, ang]`` velocity in world frame from the free joint (inverse of write)."""
    qa, da = robot_map.root_qpos_adr, robot_map.root_dof_adr
    rot_wb = quat_to_rotmat(data.qpos[qa + 3 : qa + 7].astype(np.float64))
    ang_vel_w = rot_wb @ data.qvel[da + 3 : da + 6]
    ipos_b = model.body_ipos[robot_map.root_body_id]
    lin_vel_com_w = data.qvel[da : da + 3] + np.cross(ang_vel_w, rot_wb @ ipos_b)
    return np.concatenate([lin_vel_com_w, ang_vel_w])


class EventSet:
    """Applies a list of :class:`EventIR` terms to a MuJoCo model/data pair.

    Modes mirror the IsaacLab EventManager: ``startup`` once after model build,
    ``reset`` on every episode reset (which also resamples interval clocks), and
    ``interval`` ticked at the policy rate via :meth:`step_interval`.
    """

    _KNOWN_FUNCS = (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass",
        "randomize_rigid_body_com",
        "reset_root_state_uniform",
        "reset_joints_by_offset",
        "reset_joints_by_scale",
        "reset_joints_around_default",
        "push_by_setting_velocity",
        "apply_external_force_torque",
        "randomize_joint_default_pos",
        "check_joint_friction",
    )
    # Accepted without a runtime effect (see the module docstring).
    _NOOP_FUNCS = ("check_joint_friction",)
    # Actuator-level randomization (PD gains held by the runtime env, joint friction / armature
    # split across the stiction switch and the implicit-PD damping) is not ported: accepted
    # only in strict mode, where no event applies.
    _STRICT_ONLY_FUNCS = ("randomize_actuator_gains", "randomize_joint_parameters")
    _MODEL_CONST_FUNCS = ("randomize_rigid_body_mass", "randomize_rigid_body_com")
    _KNOWN_MODES = ("startup", "reset", "interval")

    def __init__(
        self,
        events: list[EventIR],
        *,
        terrain_static_friction: float = 1.0,
        friction_combine_mode: str = "average",
        strict: bool = False,
        rng: np.random.Generator | None = None,
        entity: str = "robot",
    ) -> None:
        has_material_event = False
        for event in events:
            func = class_name(event.func)
            if func in self._STRICT_ONLY_FUNCS:
                if not strict:
                    raise NotImplementedError(
                        f"event '{event.name}' ({event.func}) is not ported to the MuJoCo runtime; "
                        "it is only accepted in strict mode, where events do not apply"
                    )
                continue
            if func not in self._KNOWN_FUNCS:
                raise ValueError(
                    f"unsupported event func '{event.func}' "
                    f"(known: {list(self._KNOWN_FUNCS)}; strict-only: {list(self._STRICT_ONLY_FUNCS)})"
                )
            if event.mode not in self._KNOWN_MODES:
                raise ValueError(
                    f"event '{event.name}' has unsupported mode '{event.mode}' (known: {list(self._KNOWN_MODES)}); "
                    "it would otherwise silently never fire"
                )
            if event.mode == "interval" and event.interval_range_s is None:
                raise ValueError(f"interval event '{event.name}' has no interval_range_s")
            check_scene_entity_cfg(
                event.params.get("asset_cfg"),
                f"event '{event.name}'",
                entity=entity,
                # Isaac's reset_joints_around_default ignores asset_cfg joint selection
                # and writes the full joint state (see _reset_joints_around_default).
                check_selectors=func != "reset_joints_around_default",
            )
            if func == "randomize_rigid_body_material":
                has_material_event = True
                static_range = tuple(event.params.get("static_friction_range", (1.0, 1.0)))
                dynamic_range = tuple(event.params.get("dynamic_friction_range", (1.0, 1.0)))
                degenerate = static_range[0] == static_range[1] == dynamic_range[0] == dynamic_range[1]
                if not degenerate and not event.params.get("make_consistent", False):
                    warnings.warn(
                        f"material event '{event.name}': PhysX applies the bucket's DYNAMIC friction "
                        f"(range {dynamic_range}) to sliding contacts while MuJoCo's single coefficient "
                        f"carries the STATIC value (range {static_range}, the no-slip breakaway "
                        "threshold); slip-phase friction forces will follow the static draw",
                        UserWarning,
                        stacklevel=2,
                    )
        self._events = [e for e in events if class_name(e.func) not in self._STRICT_ONLY_FUNCS + self._NOOP_FUNCS]
        self.joint_default_pos_offset: np.ndarray | None = None
        """Accumulated ``randomize_joint_default_pos`` offset (J,), Isaac order; None if never applied."""
        self._terrain_static_friction = float(terrain_static_friction)
        self._friction_combine_mode = str(friction_combine_mode)
        if has_material_event:
            pair_friction(1.0, 1.0, self._friction_combine_mode)  # validate the mode eagerly
        self._strict = bool(strict)
        self._interval_events = [e for e in self._events if e.mode == "interval"]
        self._interval_time_left: list[float | None] = [None] * len(self._interval_events)
        # IsaacLab samples the material buckets ONCE at term init and only draws bucket
        # *indices* per application. With a construction rng the buckets are fixed here;
        # otherwise they are sampled once on the first application and cached.
        self._material_buckets: dict[str, np.ndarray] = {}
        # Persistent link-frame wrenches (body id -> (force_b, torque_b)), held between
        # resets and re-rotated into xfrc_applied before every MuJoCo step.
        self._external_wrench_b: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if rng is not None:
            for event in self._events:
                if class_name(event.func) == "randomize_rigid_body_material":
                    self._material_buckets[event.name] = self._sample_material_buckets(event.params, rng)

    def apply_startup(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator
    ) -> None:
        if self._strict:
            return
        self._apply_mode("startup", model, data, robot_map, rng)

    def apply_reset(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator
    ) -> None:
        if self._strict:
            return
        # EventManager.reset(): resample the interval clocks at episode start. Held
        # external wrenches clear (Isaac zeroes them on reset) before reset-mode
        # terms re-sample; the caller has already zeroed xfrc_applied.
        self._external_wrench_b.clear()
        for i, event in enumerate(self._interval_events):
            assert event.interval_range_s is not None
            self._interval_time_left[i] = float(rng.uniform(*event.interval_range_s))
        self._apply_mode("reset", model, data, robot_map, rng)

    def step_interval(
        self,
        dt: float,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_map: RobotMap,
        rng: np.random.Generator,
    ) -> bool:
        """Tick interval clocks by ``dt``; apply and resample expired terms.

        Returns True when at least one interval event fired this step.
        """
        if self._strict:
            return False
        applied = False
        touched_model = False
        for i, event in enumerate(self._interval_events):
            assert event.interval_range_s is not None
            time_left = self._interval_time_left[i]
            if time_left is None:  # first tick before any reset: initialize like the manager
                time_left = float(rng.uniform(*event.interval_range_s))
            time_left -= dt
            if time_left < 1e-6:
                time_left = float(rng.uniform(*event.interval_range_s))
                self._apply_event(event, model, data, robot_map, rng)
                applied = True
                touched_model |= class_name(event.func) in self._MODEL_CONST_FUNCS
            self._interval_time_left[i] = time_left
        if touched_model:
            _refresh_model_constants(model)
        return applied

    # -- internals ------------------------------------------------------------------

    def _apply_mode(
        self, mode: str, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator
    ) -> None:
        touched_model = False
        for event in self._events:
            if event.mode != mode:
                continue
            self._apply_event(event, model, data, robot_map, rng)
            touched_model |= class_name(event.func) in self._MODEL_CONST_FUNCS
        if touched_model:
            _refresh_model_constants(model)

    def _apply_event(
        self, event: EventIR, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator
    ) -> None:
        func = class_name(event.func)
        if func == "randomize_rigid_body_material":
            self._randomize_rigid_body_material(model, data, robot_map, rng, event)
            return
        handler = getattr(self, f"_{func}")
        handler(model, data, robot_map, rng, event.params)

    @staticmethod
    def _sample_material_buckets(params: dict, rng: np.random.Generator) -> np.ndarray:
        """Sample the ``(num_buckets, 3)`` material table (static, dynamic, restitution)."""
        static_range = params.get("static_friction_range", (1.0, 1.0))
        dynamic_range = params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(params.get("num_buckets", 1))
        lo = np.array([static_range[0], dynamic_range[0], restitution_range[0]], dtype=np.float64)
        hi = np.array([static_range[1], dynamic_range[1], restitution_range[1]], dtype=np.float64)
        buckets = rng.uniform(lo, hi, (num_buckets, 3))
        if params.get("make_consistent", False):
            buckets[:, 1] = np.minimum(buckets[:, 0], buckets[:, 1])
        return buckets

    def _randomize_rigid_body_material(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, event: EventIR
    ) -> None:
        params = event.params
        buckets = self._material_buckets.get(event.name)
        if buckets is None:  # no construction rng: sample once on first application
            buckets = self._sample_material_buckets(params, rng)
            self._material_buckets[event.name] = buckets
        num_buckets = buckets.shape[0]
        for name in _matched_body_names(params, robot_map):
            for geom_id in robot_map.geom_ids.get(name, []):
                bucket = buckets[int(rng.integers(0, num_buckets))]
                # Pair value; restitution (bucket[2]) ignored — see module docstring.
                model.geom_friction[geom_id, 0] = pair_friction(
                    float(bucket[0]), self._terrain_static_friction, self._friction_combine_mode
                )

    def _randomize_rigid_body_mass(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        # One draw per matched ISAAC body (RNG parity), aggregated onto the surviving
        # MuJoCo bodies — see module docstring (randomize_rigid_body_mass).
        names = _matched_body_names(params, robot_map)
        operation = params["operation"]
        distribution = params.get("distribution", "uniform")
        dist_params = tuple(params["mass_distribution_params"])
        recompute_inertia = bool(params.get("recompute_inertia", True))
        samples = _sample_distribution(rng, dist_params, len(names), distribution)
        merged = _merged_body_ids(robot_map)
        new_mass_of: dict[int, float] = {}
        for name, sample in zip(names, samples):
            for bid in robot_map.body_ids[name]:
                default_mass = float(robot_map.default_body_mass[bid])
                if operation == "add":
                    new_mass_of[bid] = new_mass_of.get(bid, default_mass) + float(sample)
                elif operation in ("scale", "abs"):
                    if bid in merged:
                        raise NotImplementedError(
                            f"mass randomization '{operation}' on body '{name}': its MuJoCo body is a weld "
                            "merge of several Isaac links and the bundle carries no per-link mass split"
                        )
                    new_mass_of[bid] = float(default_mass * sample) if operation == "scale" else float(sample)
                else:
                    raise ValueError(f"unsupported mass randomization operation '{operation}'")
        for bid, new_mass in new_mass_of.items():
            default_mass = float(robot_map.default_body_mass[bid])
            model.body_mass[bid] = new_mass
            if recompute_inertia:
                # On a merged body the ratio rescales the merged inertia while Isaac
                # rescales only the matched link's — exact for unmerged bodies.
                model.body_inertia[bid] = robot_map.default_body_inertia[bid] * (new_mass / default_mass)

    def _randomize_rigid_body_com(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        com_range = params["com_range"]
        offset = _sample_range_dict(rng, com_range, ("x", "y", "z"))
        # Isaac draws one offset and applies it to every matched body (broadcast).
        for bid in _matched_body_ids(params, robot_map):
            model.body_ipos[bid] += offset

    def _reset_root_state_uniform(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        pose_samples = _sample_range_dict(rng, params["pose_range"], _POSE_KEYS)
        vel_samples = _sample_range_dict(rng, params["velocity_range"], _POSE_KEYS)
        pos_w = robot_map.default_root_pose_env[:3] + robot_map.env_origin_w + pose_samples[:3]
        delta_quat = quat_from_euler_xyz(pose_samples[3], pose_samples[4], pose_samples[5])
        quat_wxyz = quat_mul(robot_map.default_root_pose_env[3:7], delta_quat)
        vel_com_w = robot_map.default_root_vel_w + vel_samples
        write_root_state(model, data, robot_map, pos_w, quat_wxyz, vel_com_w)

    def _reset_joints_by_offset(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        self._reset_joints(model, data, robot_map, rng, params, operation="add")

    def _reset_joints_by_scale(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        self._reset_joints(model, data, robot_map, rng, params, operation="scale")

    def _reset_joints(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_map: RobotMap,
        rng: np.random.Generator,
        params: dict,
        *,
        operation: str,
    ) -> None:
        joint_idx = _matched_joint_indices(params, robot_map)
        pos_lo, pos_hi = params["position_range"]
        vel_lo, vel_hi = params["velocity_range"]
        pos_samples = rng.uniform(pos_lo, pos_hi, len(joint_idx))
        vel_samples = rng.uniform(vel_lo, vel_hi, len(joint_idx))
        if operation == "add":
            joint_pos = robot_map.default_joint_pos[joint_idx] + pos_samples
            joint_vel = robot_map.default_joint_vel[joint_idx] + vel_samples
        else:
            joint_pos = robot_map.default_joint_pos[joint_idx] * pos_samples
            joint_vel = robot_map.default_joint_vel[joint_idx] * vel_samples
        limits = robot_map.joint_pos_limits[joint_idx]
        joint_pos = np.clip(joint_pos, limits[:, 0], limits[:, 1])
        vel_limits = robot_map.joint_vel_limits[joint_idx]
        joint_vel = np.clip(joint_vel, -vel_limits, vel_limits)
        data.qpos[robot_map.qpos_adr[joint_idx]] = joint_pos
        data.qvel[robot_map.dof_adr[joint_idx]] = joint_vel

    def _randomize_joint_default_pos(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        joint_idx = _matched_joint_indices(params, robot_map)
        lo, hi = params["offset_range"]
        offset = np.zeros(len(robot_map.default_joint_pos), dtype=np.float64)
        offset[joint_idx] = rng.uniform(lo, hi, len(joint_idx))
        robot_map.default_joint_pos += offset
        self.joint_default_pos_offset = (
            offset if self.joint_default_pos_offset is None else self.joint_default_pos_offset + offset
        )

    def _reset_joints_around_default(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        # Spot-mdp term (isaaclab_tasks velocity/config/spot/mdp/events.py): the sampling
        # interval ENDPOINTS (default + range bounds) are clamped to the soft limits and
        # the uniform draw happens inside the clamped interval — unlike _reset_joints,
        # which clamps the sampled values. Positions are drawn before velocities, and the
        # term always writes every joint (the Isaac implementation ignores asset_cfg
        # joint selection and writes the full joint state).
        pos_lo, pos_hi = params["position_range"]
        vel_lo, vel_hi = params["velocity_range"]
        limits = robot_map.joint_pos_limits
        joint_min_pos = np.clip(robot_map.default_joint_pos + pos_lo, limits[:, 0], limits[:, 1])
        joint_max_pos = np.clip(robot_map.default_joint_pos + pos_hi, limits[:, 0], limits[:, 1])
        vel_limits = robot_map.joint_vel_limits
        joint_min_vel = np.clip(robot_map.default_joint_vel + vel_lo, -vel_limits, vel_limits)
        joint_max_vel = np.clip(robot_map.default_joint_vel + vel_hi, -vel_limits, vel_limits)
        data.qpos[robot_map.qpos_adr] = rng.uniform(joint_min_pos, joint_max_pos)
        data.qvel[robot_map.dof_adr] = rng.uniform(joint_min_vel, joint_max_vel)

    def _push_by_setting_velocity(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        vel_com_w = read_root_com_vel_w(model, data, robot_map)
        vel_com_w += _sample_range_dict(rng, params["velocity_range"], _POSE_KEYS)
        qa = robot_map.root_qpos_adr
        pos_w = data.qpos[qa : qa + 3].copy()
        quat_wxyz = data.qpos[qa + 3 : qa + 7].copy()
        write_root_state(model, data, robot_map, pos_w, quat_wxyz, vel_com_w)

    def _apply_external_force_torque(
        self, model: mujoco.MjModel, data: mujoco.MjData, robot_map: RobotMap, rng: np.random.Generator, params: dict
    ) -> None:
        # Link-frame wrenches held until the next reset and re-rotated per step
        # (see module docstring: apply_external_force_torque). One draw per matched
        # Isaac body keeps RNG parity.
        force_lo, force_hi = params["force_range"]
        torque_lo, torque_hi = params["torque_range"]
        names = _matched_body_names(params, robot_map)
        forces_b = rng.uniform(force_lo, force_hi, (len(names), 3))
        torques_b = rng.uniform(torque_lo, torque_hi, (len(names), 3))
        for name, force_b, torque_b in zip(names, forces_b, torques_b):
            for bid in robot_map.body_ids[name]:
                if not np.any(force_b) and not np.any(torque_b):
                    self._external_wrench_b.pop(bid, None)
                    continue
                if model.body(bid).name != name:
                    raise NotImplementedError(
                        f"external wrench on body '{name}': the converter welded it into "
                        f"'{model.body(bid).name}' and its link frame is not tracked at runtime"
                    )
                self._external_wrench_b[bid] = (force_b.copy(), torque_b.copy())
        self.refresh_external_wrenches(model, data)

    def clear_external_wrenches(self) -> None:
        """Drop the held DR wrenches (Isaac zeroes them in every ``_reset_idx``).

        ``reset()`` paths clear via :meth:`apply_reset` before re-sampling;
        exact-state resets run no reset events and call this instead so a
        previous episode's wrench is not re-applied to the restored state.
        """
        self._external_wrench_b.clear()

    def refresh_external_wrenches(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Re-rotate the stored link-frame wrenches into ``xfrc_applied``.

        Call before every ``mj_step`` (no-op when no wrench is active): PhysX applies
        Isaac's held link-frame wrench at the CURRENT orientation each step, while
        ``xfrc_applied`` is a fixed world-frame wrench at the body CoM.
        """
        if not self._external_wrench_b:
            return
        mujoco.mj_kinematics(model, data)  # xquat/xpos may be stale after qpos writes
        for bid, (force_b, torque_b) in self._external_wrench_b.items():
            rot_wb = quat_to_rotmat(data.xquat[bid].astype(np.float64))
            force_w = rot_wb @ force_b
            # Shift the reduction point from the link origin to the body CoM (xfrc acts there).
            torque_w = rot_wb @ torque_b + np.cross(-(rot_wb @ model.body_ipos[bid]), force_w)
            data.xfrc_applied[bid, :3] = force_w
            data.xfrc_applied[bid, 3:] = torque_w
