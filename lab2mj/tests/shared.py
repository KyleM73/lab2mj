"""Shared paths, skip markers, and builders for the lab2mj test suite."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lab2mj.bundle import MANIFEST_SCHEMA
from lab2mj.commands import RobotState
from lab2mj.env_yaml import _plain, load_env_yaml, parse_env_yaml
from lab2mj.ir import EnvIR
from lab2mj.quat import heading_w_from_quat

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DATA = REPO_ROOT / "data"
SPOT_USD = DATA / "usd_cache" / "spot.usd"
G1_USD = DATA / "usd_cache" / "g1_minimal.usd"
SPOT_ENV_YAML = FIXTURES / "spot_velocity_env.yaml"
G1_ENV_YAML = FIXTURES / "g1_flat_env.yaml"

HAS_PXR = importlib.util.find_spec("pxr") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

needs_spot = pytest.mark.skipif(
    not (HAS_PXR and SPOT_USD.exists()), reason="requires pxr (usd-core) and data/usd_cache/spot.usd"
)
needs_g1 = pytest.mark.skipif(
    not (HAS_PXR and G1_USD.exists()), reason="requires pxr (usd-core) and data/usd_cache/g1_minimal.usd"
)

_PARSE_CACHE: dict[Path, EnvIR] = {}


def parse_fixture(name: str | Path) -> EnvIR:
    """Parse a fixture env.yaml once per session.

    Returns a deepcopy of the cached parse so tests may mutate the IR freely.
    """
    path = Path(name)
    if not path.is_absolute():
        path = FIXTURES / path
    if path not in _PARSE_CACHE:
        _PARSE_CACHE[path] = parse_env_yaml(path)
    return copy.deepcopy(_PARSE_CACHE[path])


def deep_merge(base: dict, delta: dict) -> dict:
    """Recursively merge ``delta`` onto ``base`` (delta leaves win); returns a new dict."""
    out = dict(base)
    for key, value in delta.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def g1_rough_env_dict() -> dict:
    """The rough-G1 env dict: the real flat dump with the rough-task deltas merged on.

    JSON-plain (``_plain``), so callers may ``yaml.safe_dump`` it for file-based
    entry points like ``convert_run``.
    """
    flat = _plain(load_env_yaml(G1_ENV_YAML))
    deltas = load_env_yaml(FIXTURES / "g1_rough_env_deltas.yaml")
    return deep_merge(flat, deltas)


def make_state(
    pos=(0.0, 0.0, 0.5),
    quat=(1.0, 0.0, 0.0, 0.0),
    heading=None,
    origin=(0.0, 0.0, 0.0),
) -> RobotState:
    quat = np.asarray(quat, dtype=np.float64)
    if heading is None:
        heading = heading_w_from_quat(quat)
    return RobotState(
        root_pos_w=np.asarray(pos), root_quat_w_wxyz=quat, heading_w=heading, env_origin_w=np.asarray(origin)
    )


def minimal_manifest(**sections: Any) -> dict[str, Any]:
    """A structurally valid plain-JSON manifest (the superset of required keys).

    Dict-valued keyword arguments are merged into the matching top-level section
    (``robot={"total_mass": 3.0}`` updates just that key); non-dict values
    (lists, ``None``) replace the section. A new required manifest key needs
    adding here only.
    """
    isaac_order = ["j_a", "j_b", "j_c"]
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "robot": {
            "isaac_joint_order": isaac_order,
            "mj_joint_order": ["j_b", "j_a", "j_c"],
            "isaac_to_mj": [1, 0, 2],
            "mj_to_isaac": [1, 0, 2],
            "root_body": "base",
            "isaac_body_names": ["base", "foot"],
            "body_map": {"base": "base", "foot": "base"},
            "geom_map": {"base": ["base_col0"], "foot": []},
            "default_qpos": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3],
            "default_qvel": [0.0] * 9,
            "default_joint_pos_isaac": [0.1, -0.2, 0.3],
            "default_joint_vel_isaac": [0.0] * 3,
            "joint_pos_limits_isaac": [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]],
            "joint_vel_limits_isaac": [20.0, 20.0, 30.0],
            "soft_joint_pos_limit_factor": 0.9,
            "total_mass": 12.5,
        },
        "timing": {"physics_dt": 0.005, "decimation": 4, "episode_length_s": 20.0},
        "obs": {
            "groups": {"policy": {"name": "policy", "terms": []}},
            "layouts": {"policy": []},
            "obs_dims": {"policy": 9},
            "policy_group": "policy",
        },
        "action": {
            "ir": {"func": "JointPositionAction", "scale": 0.5},
            "dim": 3,
            "joint_ids_isaac": [0, 1, 2],
            "scale": [0.5] * 3,
            "offset": [0.1, -0.2, 0.3],
            "clip": None,
        },
        "actuators": [{"name": "all", "joint_names_expr": [".*"], "model": "implicit_pd"}],
        "commands": [{"name": "base_velocity", "type": "UniformVelocityCommand", "params": {}, "dim": 3}],
        "events": [],
        "terminations": [],
        "terrain": {"terrain_type": "plane", "generator": None, "physics_material": {"static_friction": 1.0}},
        "contact_sensors": [],
        "height_scanners": [],
        "sim": {"gravity_w": [0.0, 0.0, -9.81], "physics_material": None, "robot_physics_material": None},
        "init": {
            "root_pos_env": [0.0, 0.0, 0.5],
            "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "root_lin_vel_env": [0.0, 0.0, 0.0],
            "root_ang_vel_env": [0.0, 0.0, 0.0],
            "joint_pos": {".*": 0.0},
            "joint_vel": {".*": 0.0},
            "env_origin_w": [0.0, 0.0, 0.0],
            "seed": 42,
        },
        "sources": {"env_yaml_sha256": "a" * 64, "usd_sha256": "b" * 64},
    }
    for key, override in sections.items():
        if key not in manifest:
            raise KeyError(f"unknown manifest section {key!r}")
        if isinstance(manifest[key], dict) and isinstance(override, dict):
            manifest[key] = {**manifest[key], **override}
        else:
            manifest[key] = override
    return manifest
