"""Intermediate representation shared by the IsaacLab -> MuJoCo converter.

Plain dataclasses (stdlib + numpy only) that describe everything the MuJoCo
runtime needs from a resolved IsaacLab env config. All types round-trip
through ``to_dict()``/``from_dict()`` with JSON-serializable dicts (numpy
arrays and tuples become lists), so any subset can be embedded in the bundle
manifest.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ActuatorModel = Literal[
    "implicit_pd", "ideal_pd", "delayed_pd", "remotized_pd", "dc_motor", "actuator_net_lstm", "actuator_net_mlp"
]

# Scalar gain/limit values may instead be dicts keyed by joint-name regex,
# resolved against the actual joint list at build time.
ScalarOrRegexDict = float | dict[str, float] | None

# Variadic aliases so parsed sequences type-check; arity is asserted where it matters.
Vec3 = tuple[float, ...]
Quat = tuple[float, ...]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _vec(value: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in value)


class _IRBase:
    def to_dict(self) -> dict[str, Any]:
        assert dataclasses.is_dataclass(self) and not isinstance(self, type)
        return _jsonable(dataclasses.asdict(self))


@dataclass
class TimingIR(_IRBase):
    physics_dt: float
    decimation: int
    episode_length_s: float

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimingIR":
        return cls(
            physics_dt=float(data["physics_dt"]),
            decimation=int(data["decimation"]),
            episode_length_s=float(data["episode_length_s"]),
        )


@dataclass
class RobotInitIR(_IRBase):
    """Nominal root + joint state (Isaac ``init_state``).

    Root pose/velocity are **env-local** (relative to the per-env origin), not
    world-frame; the runtime must add the env origin before treating them as
    world quantities (env origins are pure translations, so the quaternion and
    velocities transfer unchanged).
    """

    root_pos_env: Vec3 = (0.0, 0.0, 0.0)
    root_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    root_lin_vel_env: Vec3 = (0.0, 0.0, 0.0)
    root_ang_vel_env: Vec3 = (0.0, 0.0, 0.0)
    # Regex -> value; resolved against the robot's joint names at build time.
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RobotInitIR":
        return cls(
            root_pos_env=_vec(data["root_pos_env"]),
            root_quat_wxyz=_vec(data["root_quat_wxyz"]),
            root_lin_vel_env=_vec(data["root_lin_vel_env"]),
            root_ang_vel_env=_vec(data["root_ang_vel_env"]),
            joint_pos=dict(data["joint_pos"]),
            joint_vel=dict(data["joint_vel"]),
        )


@dataclass
class ActuatorGroupIR(_IRBase):
    name: str
    joint_names_expr: list[str]
    model: ActuatorModel
    stiffness: ScalarOrRegexDict = None
    damping: ScalarOrRegexDict = None
    effort_limit: ScalarOrRegexDict = None
    # PhysX >= 5 joint-friction model (PxJointAxis): `friction` is the STATIC
    # friction effort — a breakaway threshold acting on stationary joints only.
    # A moving joint's friction is `dynamic_friction + viscous_friction * |qd|`.
    friction: ScalarOrRegexDict = None
    dynamic_friction: ScalarOrRegexDict = None
    viscous_friction: ScalarOrRegexDict = None
    armature: ScalarOrRegexDict = None
    # Torque-speed curve input for the DC-motor models (dc_motor / actuator nets);
    # recorded for provenance only on the plain PD models.
    velocity_limit: ScalarOrRegexDict = None
    min_delay: int | None = None
    max_delay: int | None = None
    # (angle, transmission ratio, max torque) rows for remotized_pd.
    joint_parameter_lookup: np.ndarray | None = None
    # Optional remotized_pd torque-speed envelope (contact_lab TorqueSpeedRemotizedPDActuator):
    # (max_torque, min_torque, max_speed, min_speed, max_flat_speed, min_flat_speed).
    torque_speed_envelope: list[float] | None = None
    # Stall torque of the DCMotor torque-speed curve (dc_motor / actuator nets).
    saturation_effort: ScalarOrRegexDict = None
    # Source of the actuator-net weights as recorded in env.yaml (URL or path).
    network_file: str | None = None
    # Bundle-relative path of the copied actuator-net weights (set by convert).
    network_bundle_path: str | None = None
    # MLP actuator-net input/output contract (actuator_net_mlp): input scaling,
    # output scaling, block order, and which history steps feed the network
    # (0 = current physics step, n = n steps in the past).
    pos_scale: float | None = None
    vel_scale: float | None = None
    torque_scale: float | None = None
    input_order: str | None = None
    input_idx: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActuatorGroupIR":
        lut = data.get("joint_parameter_lookup")
        input_idx = data.get("input_idx")
        return cls(
            name=data["name"],
            joint_names_expr=list(data["joint_names_expr"]),
            model=data["model"],
            stiffness=data.get("stiffness"),
            damping=data.get("damping"),
            effort_limit=data.get("effort_limit"),
            friction=data.get("friction"),
            dynamic_friction=data.get("dynamic_friction"),
            viscous_friction=data.get("viscous_friction"),
            armature=data.get("armature"),
            velocity_limit=data.get("velocity_limit"),
            min_delay=data.get("min_delay"),
            max_delay=data.get("max_delay"),
            joint_parameter_lookup=None if lut is None else np.asarray(lut, dtype=np.float64),
            torque_speed_envelope=None
            if data.get("torque_speed_envelope") is None
            else [float(v) for v in data["torque_speed_envelope"]],
            saturation_effort=data.get("saturation_effort"),
            network_file=data.get("network_file"),
            network_bundle_path=data.get("network_bundle_path"),
            pos_scale=None if data.get("pos_scale") is None else float(data["pos_scale"]),
            vel_scale=None if data.get("vel_scale") is None else float(data["vel_scale"]),
            torque_scale=None if data.get("torque_scale") is None else float(data["torque_scale"]),
            input_order=data.get("input_order"),
            input_idx=None if input_idx is None else [int(i) for i in input_idx],
        )


@dataclass
class NoiseIR(_IRBase):
    kind: str
    operation: str = "add"
    params: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoiseIR":
        return cls(kind=data["kind"], operation=data.get("operation", "add"), params=dict(data.get("params") or {}))


@dataclass
class ObsTermIR(_IRBase):
    name: str
    func: str
    params: dict[str, Any] = field(default_factory=dict)
    noise: NoiseIR | None = None
    clip: tuple[float, float] | None = None
    # Scalar, per-element list, or regex dict.
    scale: float | list[float] | dict[str, float] | None = None
    history_length: int = 0
    flatten_history_dim: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObsTermIR":
        noise = data.get("noise")
        clip = data.get("clip")
        return cls(
            name=data["name"],
            func=data["func"],
            params=dict(data.get("params") or {}),
            noise=None if noise is None else NoiseIR.from_dict(noise),
            clip=None if clip is None else (float(clip[0]), float(clip[1])),
            scale=data.get("scale"),
            history_length=int(data.get("history_length") or 0),
            flatten_history_dim=bool(data.get("flatten_history_dim", True)),
        )


@dataclass
class ObsGroupIR(_IRBase):
    name: str
    terms: list[ObsTermIR] = field(default_factory=list)
    enable_corruption: bool = False
    concatenate_terms: bool = True
    # Dimension along which concatenate_terms stacks (IsaacLab default: -1).
    concatenate_dim: int = -1
    # Group-level override; None keeps per-term history lengths. When set,
    # IsaacLab copies both history_length and flatten_history_dim onto every term.
    history_length: int | None = None
    flatten_history_dim: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObsGroupIR":
        return cls(
            name=data["name"],
            terms=[ObsTermIR.from_dict(t) for t in data.get("terms") or []],
            enable_corruption=bool(data.get("enable_corruption", False)),
            concatenate_terms=bool(data.get("concatenate_terms", True)),
            concatenate_dim=-1 if data.get("concatenate_dim") is None else int(data["concatenate_dim"]),
            history_length=data.get("history_length"),
            flatten_history_dim=bool(data.get("flatten_history_dim", True)),
        )


@dataclass
class ActionIR(_IRBase):
    func: str
    joint_names_expr: list[str] = field(default_factory=lambda: [".*"])
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    use_default_offset: bool = False
    clip: dict[str, tuple[float, float]] | tuple[float, float] | None = None
    preserve_order: bool = False
    # PreTrainedPolicyAction (navigation tasks) wraps a frozen TorchScript low-level
    # policy: it consumes `low_level_obs` (with the held high-level action in the
    # group's `velocity_commands` slot) every `low_level_decimation` physics steps
    # and drives the embedded `low_level_action` term. The per-joint fields above
    # stay at their defaults on such a wrapper term.
    policy_path: str | None = None
    low_level_decimation: int | None = None
    low_level_action: "ActionIR | None" = None
    low_level_obs: ObsGroupIR | None = None
    # Bundle-relative path of the copied low-level policy (set by convert).
    policy_bundle_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionIR":
        low_action = data.get("low_level_action")
        low_obs = data.get("low_level_obs")
        low_decimation = data.get("low_level_decimation")
        return cls(
            func=data["func"],
            joint_names_expr=list(data.get("joint_names_expr") or [".*"]),
            scale=data.get("scale", 1.0),
            offset=data.get("offset", 0.0),
            use_default_offset=bool(data.get("use_default_offset", False)),
            clip=data.get("clip"),
            preserve_order=bool(data.get("preserve_order", False)),
            policy_path=data.get("policy_path"),
            low_level_decimation=None if low_decimation is None else int(low_decimation),
            low_level_action=None if low_action is None else ActionIR.from_dict(low_action),
            low_level_obs=None if low_obs is None else ObsGroupIR.from_dict(low_obs),
            policy_bundle_path=data.get("policy_bundle_path"),
        )


@dataclass
class CommandIR(_IRBase):
    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommandIR":
        return cls(name=data["name"], type=data["type"], params=dict(data.get("params") or {}))


@dataclass
class EventIR(_IRBase):
    name: str
    func: str
    mode: Literal["startup", "reset", "interval"]
    params: dict[str, Any] = field(default_factory=dict)
    interval_range_s: tuple[float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventIR":
        interval = data.get("interval_range_s")
        return cls(
            name=data["name"],
            func=data["func"],
            mode=data["mode"],
            params=dict(data.get("params") or {}),
            interval_range_s=None if interval is None else (float(interval[0]), float(interval[1])),
        )


@dataclass
class TerminationIR(_IRBase):
    name: str
    func: str
    params: dict[str, Any] = field(default_factory=dict)
    time_out: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerminationIR":
        return cls(
            name=data["name"],
            func=data["func"],
            params=dict(data.get("params") or {}),
            time_out=bool(data.get("time_out", False)),
        )


@dataclass
class TerrainIR(_IRBase):
    terrain_type: str
    generator: dict[str, Any] | None = None
    physics_material: dict[str, Any] = field(default_factory=dict)
    # Scene layout inputs of TerrainImporter (terrain.build_terrain parameters).
    num_envs: int | None = None
    env_spacing: float | None = None
    max_init_terrain_level: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerrainIR":
        num_envs = data.get("num_envs")
        env_spacing = data.get("env_spacing")
        max_init_terrain_level = data.get("max_init_terrain_level")
        return cls(
            terrain_type=data["terrain_type"],
            generator=data.get("generator"),
            physics_material=dict(data.get("physics_material") or {}),
            num_envs=None if num_envs is None else int(num_envs),
            env_spacing=None if env_spacing is None else float(env_spacing),
            max_init_terrain_level=None if max_init_terrain_level is None else int(max_init_terrain_level),
        )


@dataclass
class ContactSensorIR(_IRBase):
    name: str
    prim_path: str
    history_length: int = 0
    track_air_time: bool = False
    force_threshold: float = 1.0
    update_period: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContactSensorIR":
        return cls(
            name=data["name"],
            prim_path=data["prim_path"],
            history_length=int(data.get("history_length", 0)),
            track_air_time=bool(data.get("track_air_time", False)),
            force_threshold=float(data.get("force_threshold", 1.0)),
            update_period=float(data.get("update_period", 0.0)),
        )


@dataclass
class HeightScannerIR(_IRBase):
    """A ``RayCasterCfg`` height-scan sensor attached to a robot body.

    ``pattern`` is the raw ``pattern_cfg`` dict from the dump (``func`` plus the
    pattern parameters); only grid patterns are servable by the runtime
    provider, but any pattern parses so that unused sensors never block
    conversion.
    """

    name: str
    prim_path: str
    # Leaf prim of prim_path: the articulation link whose pose carries the sensor.
    attach_body_name: str
    pattern: dict[str, Any] = field(default_factory=dict)
    offset_pos: Vec3 = (0.0, 0.0, 0.0)
    offset_quat_wxyz: Quat = (1.0, 0.0, 0.0, 0.0)
    ray_alignment: str = "base"
    max_distance: float = 1.0e6
    # World-frame sensor-position drift, one scalar range drawn per axis.
    drift_range: tuple[float, float] = (0.0, 0.0)
    # Per-axis drift in the ray projection frame: x/y shift the ray origins,
    # z shifts the hit points.
    ray_cast_drift_range: dict[str, tuple[float, float]] = field(default_factory=dict)
    update_period: float = 0.0
    # Prim paths the rays are cast against (recorded for provenance; the MuJoCo
    # runtime always casts against the bundle terrain).
    mesh_prim_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeightScannerIR":
        drift = data.get("drift_range") or (0.0, 0.0)
        return cls(
            name=data["name"],
            prim_path=data["prim_path"],
            attach_body_name=data["attach_body_name"],
            pattern=dict(data.get("pattern") or {}),
            offset_pos=_vec(data.get("offset_pos") or (0.0, 0.0, 0.0)),
            offset_quat_wxyz=_vec(data.get("offset_quat_wxyz") or (1.0, 0.0, 0.0, 0.0)),
            ray_alignment=str(data.get("ray_alignment", "base")),
            max_distance=float(data.get("max_distance", 1.0e6)),
            drift_range=(float(drift[0]), float(drift[1])),
            ray_cast_drift_range={
                str(k): (float(v[0]), float(v[1])) for k, v in (data.get("ray_cast_drift_range") or {}).items()
            },
            update_period=float(data.get("update_period", 0.0)),
            mesh_prim_paths=[str(p) for p in data.get("mesh_prim_paths") or []],
        )


@dataclass
class EnvIR(_IRBase):
    timing: TimingIR
    robot_init: RobotInitIR
    action: ActionIR
    # Scene key of the robot articulation — the entity name SceneEntityCfgs refer to.
    robot_entity: str = "robot"
    seed: int | None = None
    gravity_w: Vec3 = (0.0, 0.0, -9.81)
    usd_path: str | None = None
    actuators: list[ActuatorGroupIR] = field(default_factory=list)
    obs_groups: list[ObsGroupIR] = field(default_factory=list)
    commands: list[CommandIR] = field(default_factory=list)
    events: list[EventIR] = field(default_factory=list)
    terminations: list[TerminationIR] = field(default_factory=list)
    terrain: TerrainIR | None = None
    contact_sensors: list[ContactSensorIR] = field(default_factory=list)
    height_scanners: list[HeightScannerIR] = field(default_factory=list)
    sim_physics_material: dict[str, Any] | None = None
    # The robot spawn's physics material; falls back to the sim default material
    # (sim_physics_material) when unset, exactly like PhysX binds materials.
    robot_physics_material: dict[str, Any] | None = None
    # PhysX TGS positional iterations of the articulation (spawn.articulation_props);
    # sets how finely PhysX resolves contacts within one physics step.
    solver_position_iterations: int | None = None

    def obs_group(self, name: str) -> ObsGroupIR:
        for group in self.obs_groups:
            if group.name == name:
                return group
        raise KeyError(f"observation group '{name}' not found (have {[g.name for g in self.obs_groups]})")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvIR":
        terrain = data.get("terrain")
        return cls(
            timing=TimingIR.from_dict(data["timing"]),
            robot_init=RobotInitIR.from_dict(data["robot_init"]),
            action=ActionIR.from_dict(data["action"]),
            robot_entity=str(data.get("robot_entity") or "robot"),
            seed=data.get("seed"),
            gravity_w=_vec(data.get("gravity_w") or (0.0, 0.0, -9.81)),
            usd_path=data.get("usd_path"),
            actuators=[ActuatorGroupIR.from_dict(a) for a in data.get("actuators") or []],
            obs_groups=[ObsGroupIR.from_dict(g) for g in data.get("obs_groups") or []],
            commands=[CommandIR.from_dict(c) for c in data.get("commands") or []],
            events=[EventIR.from_dict(e) for e in data.get("events") or []],
            terminations=[TerminationIR.from_dict(t) for t in data.get("terminations") or []],
            terrain=None if terrain is None else TerrainIR.from_dict(terrain),
            contact_sensors=[ContactSensorIR.from_dict(s) for s in data.get("contact_sensors") or []],
            height_scanners=[HeightScannerIR.from_dict(s) for s in data.get("height_scanners") or []],
            sim_physics_material=data.get("sim_physics_material"),
            robot_physics_material=data.get("robot_physics_material"),
            solver_position_iterations=data.get("solver_position_iterations"),
        )
