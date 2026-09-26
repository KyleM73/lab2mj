"""Bundle manifest read/write for the converted MuJoCo environment.

A bundle directory is fully self-contained::

    <bundle>/
      robot.xml       # robot-only MJCF (provenance/debugging)
      scene.xml       # robot + terrain + lights; the runtime entry point
      assets/*.obj    # exported meshes referenced by both XMLs
      assets/actuator_nets/*.pt  # TorchScript actuator nets (only when used)
      manifest.json   # everything the runtime needs beyond the compiled model

``manifest.json`` is versioned (``"schema": 1``) and must be sufficient to run
the environment **without re-reading** ``env.yaml`` or the USD. It stores:

* ``robot``: Isaac and MJCF joint orders plus the index permutations between
  them (``q_mj = q_isaac[isaac_to_mj]``, ``q_isaac = q_mj[mj_to_isaac]``), the
  Isaac body-name list (PhysX breadth-first, including bodies welded away by
  the USD converter) with the ``body_map`` multimap onto MuJoCo body names and
  ``geom_map`` onto collision geom names, the full default ``qpos``/``qvel``,
  default joint state and soft joint limits in Isaac order.
* ``timing``: physics dt, decimation, episode length, plus ``physics_substeps``
  (MuJoCo steps per Isaac physics step; optional, absent means 1 — the compiled
  model's timestep is ``physics_dt / physics_substeps``).
* ``obs``: every observation group's IR (noise/clip/scale/history per term),
  the per-term flat-vector layout, and total dims. ``height_scan`` terms backed
  by a recorded height scanner resolve fully; groups with other extras-backed
  terms (wrench ground truth, ...) carry a ``null`` layout/dim: they are
  recorded but not runnable by the runtime.
* ``action``: the action-term IR plus fully resolved per-joint scale/offset/clip
  arrays and the action joint indices (into the Isaac joint order). A
  ``PreTrainedPolicyAction`` term instead carries ``dim`` (the high-level
  action), ``low_level_decimation``, the resolved arrays of the wrapped
  low-level term under ``low_level``, and — inside its ``ir`` — the embedded
  low-level obs group plus ``policy_bundle_path`` (the copied TorchScript
  low-level policy under ``assets/policies/``).
* ``actuators``: per-group actuator IRs (model type, gains, limits, delay
  range, remotized torque LUT, DC-motor saturation parameters, and — for
  actuator-net groups — the bundle-relative path of the copied network file).
* ``commands`` (with per-term command dims), ``events``, ``terminations``,
  ``terrain``, ``contact_sensors``: IR dicts as parsed from env.yaml.
* ``height_scanners`` (optional; absent on bundles converted before it existed):
  RayCaster sensor IRs the runtime's height-scan provider is built from.
* ``sim``: gravity (asserted against the compiled model by the runtime, and
  the source of the projected-gravity direction) plus the sim-default and
  robot spawn physics materials (baked into the compiled geom friction;
  recorded for provenance).
* ``init``: env-local default root state, resolved env origin, seed.
* ``sources``: sha256 hashes of the source env.yaml and USD for staleness
  detection.

The file is written with python's ``json`` module, which serializes infinite
joint/effort limits as ``Infinity`` (non-strict JSON); it round-trips through
``json.load`` unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
SCENE_XML_NAME = "scene.xml"
ROBOT_XML_NAME = "robot.xml"

# Relative tolerance for bundle-XML float checks: MjSpec.to_xml() writes 6 significant
# digits, so a round-tripped value can differ from the manifest float64 by up to 5e-7
# relative. Consumers check at this tolerance and stamp the exact manifest value back.
XML_RTOL = 1e-5

_REQUIRED_TOP_KEYS = (
    "schema",
    "robot",
    "timing",
    "obs",
    "action",
    "actuators",
    "commands",
    "events",
    "terminations",
    "terrain",
    "contact_sensors",
    "sim",
    "init",
    "sources",
)
_REQUIRED_ROBOT_KEYS = (
    "isaac_joint_order",
    "mj_joint_order",
    "isaac_to_mj",
    "mj_to_isaac",
    "root_body",
    "isaac_body_names",
    "body_map",
    "geom_map",
    "default_qpos",
    "default_qvel",
    "default_joint_pos_isaac",
    "default_joint_vel_isaac",
    "joint_pos_limits_isaac",
    "joint_vel_limits_isaac",
    "soft_joint_pos_limit_factor",
)
_REQUIRED_TIMING_KEYS = ("physics_dt", "decimation", "episode_length_s")
_REQUIRED_OBS_KEYS = ("groups", "layouts", "obs_dims", "policy_group")
_REQUIRED_ACTION_KEYS = ("ir", "dim", "joint_ids_isaac", "scale", "offset", "clip")
_REQUIRED_PRE_TRAINED_ACTION_KEYS = ("ir", "dim", "low_level", "low_level_decimation")
_REQUIRED_SOURCE_KEYS = ("env_yaml_sha256", "usd_sha256")


def jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and tuples to plain JSON types."""
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{where} missing keys: {missing}")


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Raise ``ValueError`` when required manifest structure is missing or inconsistent."""
    _require_keys(manifest, _REQUIRED_TOP_KEYS, "manifest")
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported manifest schema {manifest['schema']!r} (expected {MANIFEST_SCHEMA})")

    robot = manifest["robot"]
    _require_keys(robot, _REQUIRED_ROBOT_KEYS, "manifest.robot")
    _require_keys(manifest["timing"], _REQUIRED_TIMING_KEYS, "manifest.timing")
    obs = manifest["obs"]
    _require_keys(obs, _REQUIRED_OBS_KEYS, "manifest.obs")
    action = manifest["action"]
    if "low_level" in action:
        _require_keys(action, _REQUIRED_PRE_TRAINED_ACTION_KEYS, "manifest.action")
        _require_keys(action["low_level"], _REQUIRED_ACTION_KEYS, "manifest.action.low_level")
        if int(action["low_level_decimation"]) < 1:
            raise ValueError("manifest.action.low_level_decimation must be >= 1")
        if not (action["ir"] or {}).get("policy_bundle_path"):
            raise ValueError("manifest.action.ir has no policy_bundle_path (low-level policy not bundled)")
    else:
        _require_keys(action, _REQUIRED_ACTION_KEYS, "manifest.action")
    _require_keys(manifest["sources"], _REQUIRED_SOURCE_KEYS, "manifest.sources")

    isaac_order = robot["isaac_joint_order"]
    mj_order = robot["mj_joint_order"]
    if sorted(isaac_order) != sorted(mj_order):
        raise ValueError("isaac_joint_order and mj_joint_order contain different joint names")
    num_joints = len(isaac_order)
    isaac_to_mj = robot["isaac_to_mj"]
    mj_to_isaac = robot["mj_to_isaac"]
    if sorted(isaac_to_mj) != list(range(num_joints)) or sorted(mj_to_isaac) != list(range(num_joints)):
        raise ValueError("isaac_to_mj / mj_to_isaac must be permutations of the joint indices")
    for j in range(num_joints):
        if mj_order[j] != isaac_order[isaac_to_mj[j]] or isaac_order[j] != mj_order[mj_to_isaac[j]]:
            raise ValueError("joint permutations are inconsistent with the recorded joint orders")

    for key in ("default_joint_pos_isaac", "default_joint_vel_isaac", "joint_vel_limits_isaac"):
        if len(robot[key]) != num_joints:
            raise ValueError(f"manifest.robot.{key} has {len(robot[key])} entries, expected {num_joints}")
    if len(robot["joint_pos_limits_isaac"]) != num_joints:
        raise ValueError("manifest.robot.joint_pos_limits_isaac must have one (lo, hi) pair per joint")

    policy_group = obs["policy_group"]
    if policy_group not in obs["groups"]:
        raise ValueError(f"manifest.obs.policy_group '{policy_group}' not in obs.groups {sorted(obs['groups'])}")

    arrays = action["low_level"] if "low_level" in action else action
    label = "action.low_level" if "low_level" in action else "action"
    dim = arrays["dim"]
    for key in ("joint_ids_isaac", "scale", "offset"):
        if len(arrays[key]) != dim:
            raise ValueError(f"manifest.{label}.{key} has {len(arrays[key])} entries, expected {dim}")
    if arrays["clip"] is not None and len(arrays["clip"]) != dim:
        raise ValueError(f"manifest.{label}.clip must be null or one (lo, hi) pair per action joint")


def write_manifest(bundle_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Normalize, validate, and write ``manifest.json``; returns the normalized dict.

    The returned dict is exactly what :func:`read_manifest` will produce (all
    tuples/arrays converted to lists), so callers can keep using it in place.
    """
    normalized = jsonable(manifest)
    validate_manifest(normalized)
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with open(bundle_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
        f.write("\n")
    return normalized


def read_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    """Load and validate ``manifest.json`` from a bundle directory."""
    path = Path(bundle_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_NAME} in bundle directory {bundle_dir}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)
    return manifest


def check_dump_joint_order(manifest: dict[str, Any], dump: Any) -> None:
    """Raise when a reference dump's ``joint_names`` differ from the bundle's Isaac order.

    Every dump consumer (validate gates, replays, one-step parity) must run this
    before indexing joint arrays: a mismatch means the dump was recorded on a
    different robot variant than the bundle was converted from.
    """
    dump_joints = [str(n) for n in dump["joint_names"]]
    if dump_joints != manifest["robot"]["isaac_joint_order"]:
        raise ValueError(
            "dump joint order != bundle Isaac joint order (wrong robot variant?); re-convert with --dump to inspect"
        )


def scene_xml_path(bundle_dir: str | Path) -> Path:
    """Path of the compiled scene MJCF inside a bundle directory."""
    path = Path(bundle_dir) / SCENE_XML_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {SCENE_XML_NAME} in bundle directory {bundle_dir}")
    return path
