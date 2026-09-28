"""Parse a fully-resolved IsaacLab ``params/env.yaml`` dump into :class:`EnvIR`.

Both contact_lab's ``train.py`` and IsaacLab's rsl_rl ``train.py`` dump
``env_cfg.to_dict()`` through ``dump_yaml``, so class references appear as
``module.path:Name`` strings (occasionally ``<class 'module.path.Name'>``),
python tuples as ``!!python/tuple`` and index slices as
``!!python/object/apply:builtins.slice``. Parsing stays yaml + numpy only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lab2mj.ir import (
    ActionIR,
    ActuatorGroupIR,
    ActuatorModel,
    CommandIR,
    ContactSensorIR,
    EnvIR,
    EventIR,
    HeightScannerIR,
    NoiseIR,
    ObsGroupIR,
    ObsTermIR,
    RobotInitIR,
    TerminationIR,
    TerrainIR,
    TimingIR,
)

_ACTUATOR_MODELS: dict[str, ActuatorModel] = {
    "ImplicitActuator": "implicit_pd",
    "IdealPDActuator": "ideal_pd",
    "DelayedPDActuator": "delayed_pd",
    "RemotizedPDActuator": "remotized_pd",
    "SyncFreeDelayedPDActuator": "delayed_pd",
    "SyncFreeRemotizedPDActuator": "remotized_pd",
    "DCMotor": "dc_motor",
    "ActuatorNetLSTM": "actuator_net_lstm",
    "ActuatorNetMLP": "actuator_net_mlp",
}

_CONTACT_SENSOR_CLASSES = {"ContactSensor", "SyncFreeContactSensor"}

_OBS_GROUP_META_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}

_COMMAND_SKIP_KEYS = {"class_type", "debug_vis"}


class IsaacEnvYamlLoader(yaml.SafeLoader):
    """SafeLoader that additionally understands the python tags dump_yaml emits."""


def _construct_python_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_python_slice(loader: yaml.Loader, node: yaml.Node) -> slice:
    return slice(*loader.construct_sequence(node, deep=True))


IsaacEnvYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_python_tuple)
IsaacEnvYamlLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", _construct_python_slice)


def load_env_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.load(f, Loader=IsaacEnvYamlLoader)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at the top level of {path}, got {type(data).__name__}")
    return data


def _plain(value: Any) -> Any:
    """Reduce loaded yaml values to JSON-friendly python (slices -> None/list).

    Tuples become lists so a freshly parsed IR is structurally identical to one
    rebuilt via ``EnvIR.from_dict(ir.to_dict())`` (JSON has no tuples).
    """
    if isinstance(value, slice):
        if value == slice(None):
            return None
        return [value.start, value.stop, value.step]
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def class_name(ref: str) -> str:
    """Extract the bare class/function name from a dump_yaml class reference."""
    match = re.fullmatch(r"<class '([^']+)'>", ref)
    if match:
        ref = match.group(1)
    return ref.split(":")[-1].rsplit(".", 1)[-1]


def actuator_model_from_class(ref: str) -> ActuatorModel:
    name = class_name(ref)
    name = name.removesuffix("Cfg")
    if name not in _ACTUATOR_MODELS:
        raise ValueError(f"unsupported actuator class '{ref}' (known: {sorted(_ACTUATOR_MODELS)})")
    return _ACTUATOR_MODELS[name]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_noise(cfg: dict[str, Any] | None) -> NoiseIR | None:
    if cfg is None:
        return None
    if "class_type" in cfg:
        raise ValueError(
            f"NoiseModelCfg-style noise ({cfg['class_type']!r}) is not supported; "
            "only func-based NoiseCfg noise terms convert"
        )
    func = cfg.get("func")
    kind = class_name(func).removesuffix("_noise") if isinstance(func, str) else "unknown"
    params = {k: _plain(v) for k, v in cfg.items() if k not in ("func", "operation")}
    return NoiseIR(kind=kind, operation=cfg.get("operation", "add"), params=params)


def _parse_obs_groups(cfg: dict[str, Any] | None) -> list[ObsGroupIR]:
    groups = []
    for group_name, group_cfg in (cfg or {}).items():
        if group_cfg is None:
            continue
        terms = []
        for term_name, term_cfg in group_cfg.items():
            if term_name in _OBS_GROUP_META_KEYS or term_cfg is None:
                continue
            if not isinstance(term_cfg, dict) or "func" not in term_cfg:
                continue
            if term_cfg.get("modifiers") is not None:
                raise ValueError(
                    f"obs term '{group_name}/{term_name}' uses modifiers, which the converter does not "
                    "support; the MuJoCo obs pipeline would silently diverge from IsaacLab"
                )
            clip = term_cfg.get("clip")
            terms.append(
                ObsTermIR(
                    name=term_name,
                    func=term_cfg["func"],
                    params=_plain(term_cfg.get("params") or {}),
                    noise=_parse_noise(term_cfg.get("noise")),
                    clip=None if clip is None else (float(clip[0]), float(clip[1])),
                    scale=_plain(term_cfg.get("scale")),
                    history_length=int(term_cfg.get("history_length") or 0),
                    flatten_history_dim=bool(term_cfg.get("flatten_history_dim", True)),
                )
            )
        groups.append(
            ObsGroupIR(
                name=group_name,
                terms=terms,
                enable_corruption=bool(group_cfg.get("enable_corruption", False)),
                concatenate_terms=bool(group_cfg.get("concatenate_terms", True)),
                concatenate_dim=-1 if group_cfg.get("concatenate_dim") is None else int(group_cfg["concatenate_dim"]),
                history_length=group_cfg.get("history_length"),
                flatten_history_dim=bool(group_cfg.get("flatten_history_dim", True)),
            )
        )
    return groups


def _parse_action(cfg: dict[str, Any] | None) -> ActionIR:
    terms = {name: term for name, term in (cfg or {}).items() if isinstance(term, dict) and "class_type" in term}
    if len(terms) != 1:
        raise ValueError(f"expected exactly one action term, found {sorted(terms)}")
    term = next(iter(terms.values()))
    if class_name(term["class_type"]).removesuffix("Cfg") == "PreTrainedPolicyAction":
        return _parse_pre_trained_policy_action(term)
    return _parse_action_term(term)


def _parse_action_term(term: dict[str, Any]) -> ActionIR:
    return ActionIR(
        func=term["class_type"],
        joint_names_expr=list(term.get("joint_names") or [".*"]),
        scale=_plain(term.get("scale", 1.0)),
        offset=_plain(term.get("offset", 0.0)),
        use_default_offset=bool(term.get("use_default_offset", False)),
        clip=_plain(term.get("clip")),
        preserve_order=bool(term.get("preserve_order", False)),
    )


def _parse_pre_trained_policy_action(term: dict[str, Any]) -> ActionIR:
    """Parse a ``PreTrainedPolicyActionCfg`` dump into a wrapper :class:`ActionIR`.

    The embedded low-level obs group parses like any dumped observation group.
    In dumps recorded after env construction, the ``actions`` and
    ``velocity_commands`` term funcs are the ``lambda ...`` strings
    ``PreTrainedPolicyAction.__init__`` wrote into the cfg; the runtime remaps
    those two terms by name, so their func strings are carried as-is.
    """
    policy_path = term.get("policy_path")
    if not policy_path:
        raise ValueError("PreTrainedPolicyAction term has no policy_path")
    low_actions = term.get("low_level_actions")
    low_obs = term.get("low_level_observations")
    if not isinstance(low_actions, dict) or not isinstance(low_obs, dict):
        raise ValueError(
            "PreTrainedPolicyAction term must embed low_level_actions and low_level_observations cfg dicts"
        )
    # "ll_policy" is the group name PreTrainedPolicyAction registers on its
    # standalone low-level ObservationManager.
    (low_group,) = _parse_obs_groups({"ll_policy": low_obs})
    return ActionIR(
        func=term["class_type"],
        policy_path=str(policy_path),
        low_level_decimation=int(term.get("low_level_decimation", 4)),
        low_level_action=_parse_action_term(low_actions),
        low_level_obs=low_group,
    )


def _parse_actuators(cfg: dict[str, Any] | None) -> list[ActuatorGroupIR]:
    groups = []
    for name, actuator in (cfg or {}).items():
        lut = actuator.get("joint_parameter_lookup")
        input_idx = actuator.get("input_idx")
        groups.append(
            ActuatorGroupIR(
                name=name,
                joint_names_expr=list(actuator["joint_names_expr"]),
                model=actuator_model_from_class(actuator["class_type"]),
                stiffness=_plain(actuator.get("stiffness")),
                damping=_plain(actuator.get("damping")),
                # *_sim variants are the applied values when the plain field is unset.
                effort_limit=_plain(_coalesce(actuator.get("effort_limit"), actuator.get("effort_limit_sim"))),
                friction=_plain(actuator.get("friction")),
                dynamic_friction=_plain(actuator.get("dynamic_friction")),
                viscous_friction=_plain(actuator.get("viscous_friction")),
                armature=_plain(actuator.get("armature")),
                velocity_limit=_plain(_coalesce(actuator.get("velocity_limit"), actuator.get("velocity_limit_sim"))),
                min_delay=actuator.get("min_delay"),
                max_delay=actuator.get("max_delay"),
                joint_parameter_lookup=None if lut is None else np.asarray(lut, dtype=np.float64),
                saturation_effort=_plain(actuator.get("saturation_effort")),
                network_file=actuator.get("network_file"),
                pos_scale=None if actuator.get("pos_scale") is None else float(actuator["pos_scale"]),
                vel_scale=None if actuator.get("vel_scale") is None else float(actuator["vel_scale"]),
                torque_scale=None if actuator.get("torque_scale") is None else float(actuator["torque_scale"]),
                input_order=actuator.get("input_order"),
                input_idx=None if input_idx is None else [int(i) for i in input_idx],
            )
        )
    return groups


def _parse_robot_init(cfg: dict[str, Any] | None) -> RobotInitIR:
    cfg = cfg or {}

    def vec(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
        value = cfg.get(key)
        return default if value is None else tuple(float(v) for v in value)

    return RobotInitIR(
        root_pos_env=vec("pos", (0.0, 0.0, 0.0)),
        root_quat_wxyz=vec("rot", (1.0, 0.0, 0.0, 0.0)),
        root_lin_vel_env=vec("lin_vel", (0.0, 0.0, 0.0)),
        root_ang_vel_env=vec("ang_vel", (0.0, 0.0, 0.0)),
        joint_pos={k: float(v) for k, v in (cfg.get("joint_pos") or {}).items()},
        joint_vel={k: float(v) for k, v in (cfg.get("joint_vel") or {}).items()},
    )


def _parse_commands(cfg: dict[str, Any] | None) -> list[CommandIR]:
    commands = []
    for name, command in (cfg or {}).items():
        if command is None:
            continue
        params = {
            k: _plain(v) for k, v in command.items() if k not in _COMMAND_SKIP_KEYS and not k.endswith("visualizer_cfg")
        }
        commands.append(CommandIR(name=name, type=class_name(command["class_type"]), params=params))
    return commands


def _parse_events(cfg: dict[str, Any] | None) -> list[EventIR]:
    events = []
    for name, event in (cfg or {}).items():
        if event is None:
            continue
        interval = event.get("interval_range_s")
        events.append(
            EventIR(
                name=name,
                func=event["func"],
                mode=event["mode"],
                params=_plain(event.get("params") or {}),
                interval_range_s=None if interval is None else (float(interval[0]), float(interval[1])),
            )
        )
    return events


def _parse_terminations(cfg: dict[str, Any] | None) -> list[TerminationIR]:
    terms = []
    for name, term in (cfg or {}).items():
        if term is None:
            continue
        terms.append(
            TerminationIR(
                name=name,
                func=term["func"],
                params=_plain(term.get("params") or {}),
                time_out=bool(term.get("time_out", False)),
            )
        )
    return terms


def _parse_terrain(cfg: dict[str, Any] | None, scene: dict[str, Any] | None = None) -> TerrainIR | None:
    if cfg is None:
        return None
    scene = scene or {}
    material = cfg.get("physics_material") or {}
    # InteractiveScene copies num_envs/env_spacing onto the TerrainImporter cfg before the
    # dump; fall back to the scene-level values for yamls that only carry those.
    num_envs = _coalesce(cfg.get("num_envs"), scene.get("num_envs"))
    env_spacing = _coalesce(cfg.get("env_spacing"), scene.get("env_spacing"))
    max_init_terrain_level = cfg.get("max_init_terrain_level")
    return TerrainIR(
        terrain_type=cfg.get("terrain_type", "plane"),
        generator=_plain(cfg.get("terrain_generator")),
        physics_material={k: _plain(v) for k, v in material.items() if k != "func"},
        num_envs=None if num_envs is None else int(num_envs),
        env_spacing=None if env_spacing is None else float(env_spacing),
        max_init_terrain_level=None if max_init_terrain_level is None else int(max_init_terrain_level),
    )


def find_robot_cfg(scene: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The robot articulation's scene entry: ``(scene key, cfg dict)``.

    The scene key is the entity name ``SceneEntityCfg``s refer to; it is
    recorded in the IR/manifest so runtime checks match the actual name
    instead of hardcoding ``'robot'``.
    """
    robot = scene.get("robot")
    if isinstance(robot, dict) and "actuators" in robot:
        return "robot", robot
    for key, entry in scene.items():
        if isinstance(entry, dict) and isinstance(entry.get("class_type"), str):
            if class_name(entry["class_type"]) == "Articulation":
                return str(key), entry
    raise ValueError("no articulation (robot) entry found in scene config")


def _parse_contact_sensors(scene: dict[str, Any]) -> list[ContactSensorIR]:
    sensors = []
    for name, entry in scene.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("class_type"), str):
            continue
        if class_name(entry["class_type"]) not in _CONTACT_SENSOR_CLASSES:
            continue
        sensors.append(
            ContactSensorIR(
                name=name,
                prim_path=entry.get("prim_path", ""),
                history_length=int(entry.get("history_length") or 0),
                track_air_time=bool(entry.get("track_air_time", False)),
                force_threshold=float(entry.get("force_threshold", 1.0)),
                update_period=float(entry.get("update_period", 0.0)),
            )
        )
    return sensors


def _parse_height_scanners(scene: dict[str, Any]) -> list[HeightScannerIR]:
    scanners = []
    for name, entry in scene.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("class_type"), str):
            continue
        if class_name(entry["class_type"]) != "RayCaster":
            continue
        prim_path = str(entry.get("prim_path", ""))
        offset = entry.get("offset") or {}
        offset_pos = offset.get("pos") or (0.0, 0.0, 0.0)
        offset_quat = offset.get("rot") or (1.0, 0.0, 0.0, 0.0)
        # attach_yaw_only is the deprecated spelling of ray_alignment and wins when
        # set, exactly like RayCaster._update_buffers_impl resolves it.
        ray_alignment = str(entry.get("ray_alignment", "base"))
        attach_yaw_only = entry.get("attach_yaw_only")
        if attach_yaw_only is not None:
            ray_alignment = "yaw" if attach_yaw_only else "base"
        drift = entry.get("drift_range") or (0.0, 0.0)
        scanners.append(
            HeightScannerIR(
                name=name,
                prim_path=prim_path,
                attach_body_name=prim_path.rstrip("/").rsplit("/", 1)[-1],
                pattern={k: _plain(v) for k, v in (entry.get("pattern_cfg") or {}).items()},
                offset_pos=tuple(float(v) for v in offset_pos),
                offset_quat_wxyz=tuple(float(v) for v in offset_quat),
                ray_alignment=ray_alignment,
                max_distance=float(entry.get("max_distance", 1.0e6)),
                drift_range=(float(drift[0]), float(drift[1])),
                ray_cast_drift_range={
                    str(k): (float(v[0]), float(v[1])) for k, v in (entry.get("ray_cast_drift_range") or {}).items()
                },
                update_period=float(entry.get("update_period", 0.0)),
                mesh_prim_paths=[str(p) for p in entry.get("mesh_prim_paths") or []],
            )
        )
    return scanners


def _required(mapping: dict[str, Any], key: str, where: str) -> Any:
    try:
        return mapping[key]
    except KeyError as err:
        raise ValueError(f"env.yaml has no '{where}' entry — unsupported env.yaml layout?") from err


def _material_dict(cfg: Any) -> dict[str, Any] | None:
    if cfg is None:
        return None
    return {k: _plain(v) for k, v in cfg.items() if k != "func"}


def parse_env_dict(data: dict[str, Any]) -> EnvIR:
    sim = data.get("sim") or {}
    scene = data.get("scene") or {}
    robot_entity, robot = find_robot_cfg(scene)
    spawn = robot.get("spawn") or {}
    scale = spawn.get("scale")
    if scale is not None:
        values = [float(v) for v in (scale if isinstance(scale, (list, tuple)) else (scale, scale, scale))]
        if any(v != 1.0 for v in values):
            raise NotImplementedError(
                f"robot spawn.scale={scale}: PhysX simulates the scaled asset, but the converter does not "
                "implement geometric scaling — converting would silently produce a full-size model"
            )

    gravity = sim.get("gravity") or (0.0, 0.0, -9.81)
    return EnvIR(
        timing=TimingIR(
            physics_dt=float(_required(sim, "dt", "sim.dt")),
            decimation=int(_required(data, "decimation", "decimation")),
            episode_length_s=float(_required(data, "episode_length_s", "episode_length_s")),
        ),
        robot_init=_parse_robot_init(robot.get("init_state")),
        robot_entity=robot_entity,
        action=_parse_action(data.get("actions")),
        seed=data.get("seed"),
        gravity_w=tuple(float(g) for g in gravity),
        usd_path=spawn.get("usd_path"),
        actuators=_parse_actuators(robot.get("actuators")),
        obs_groups=_parse_obs_groups(data.get("observations")),
        commands=_parse_commands(data.get("commands")),
        events=_parse_events(data.get("events")),
        terminations=_parse_terminations(data.get("terminations")),
        terrain=_parse_terrain(scene.get("terrain"), scene),
        contact_sensors=_parse_contact_sensors(scene),
        height_scanners=_parse_height_scanners(scene),
        sim_physics_material=_material_dict(sim.get("physics_material")),
        robot_physics_material=_material_dict(spawn.get("physics_material")),
        solver_position_iterations=_solver_position_iterations(spawn),
    )


def _solver_position_iterations(spawn: dict[str, Any]) -> int | None:
    props = spawn.get("articulation_props") or {}
    count = props.get("solver_position_iteration_count")
    return None if count is None else int(count)


def parse_env_yaml(path: str | Path) -> EnvIR:
    return parse_env_dict(load_env_yaml(path))
