"""Tests for the bundle manifest read/write layer (offline, no Isaac)."""

from __future__ import annotations

import numpy as np
import pytest

from lab2mj.bundle import (
    jsonable,
    read_manifest,
    validate_manifest,
    write_manifest,
)

from .shared import minimal_manifest


def _minimal_manifest() -> dict:
    """The shared minimal manifest with numpy/tuple/inf values to exercise normalization."""
    return minimal_manifest(
        robot={
            "isaac_to_mj": np.array([1, 0, 2]),
            "default_qpos": np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.3]),
            "default_qvel": np.zeros(9),
            "default_joint_pos_isaac": (0.1, -0.2, 0.3),
            "default_joint_vel_isaac": np.zeros(3),
            "joint_pos_limits_isaac": [(-1.0, 1.0), (-2.0, 2.0), (-np.inf, np.inf)],
            "joint_vel_limits_isaac": [20.0, 20.0, np.inf],
        },
        action={"scale": np.full(3, 0.5), "offset": (0.1, -0.2, 0.3)},
        sim={"gravity_w": (0.0, 0.0, -9.81)},
        init={"root_pos_env": (0.0, 0.0, 0.5), "env_origin_w": np.zeros(3)},
    )


class TestManifestRoundTrip:
    def test_write_read_equality(self, tmp_path):
        manifest = _minimal_manifest()
        normalized = write_manifest(tmp_path, manifest)
        loaded = read_manifest(tmp_path)
        assert loaded == normalized

    def test_infinite_limits_round_trip(self, tmp_path):
        write_manifest(tmp_path, _minimal_manifest())
        loaded = read_manifest(tmp_path)
        assert loaded["robot"]["joint_vel_limits_isaac"][2] == float("inf")
        assert loaded["robot"]["joint_pos_limits_isaac"][2] == [-float("inf"), float("inf")]


class TestManifestValidation:
    def test_missing_top_key_raises(self):
        manifest = jsonable(_minimal_manifest())
        del manifest["actuators"]
        with pytest.raises(ValueError, match="actuators"):
            validate_manifest(manifest)

    def test_wrong_schema_raises(self):
        manifest = jsonable(_minimal_manifest())
        manifest["schema"] = 99
        with pytest.raises(ValueError, match="schema"):
            validate_manifest(manifest)

    def test_inconsistent_permutation_raises(self):
        manifest = jsonable(_minimal_manifest())
        manifest["robot"]["isaac_to_mj"] = [0, 1, 2]  # inconsistent with the reordered mj order
        with pytest.raises(ValueError, match="permutation"):
            validate_manifest(manifest)

    def test_action_length_mismatch_raises(self):
        manifest = jsonable(_minimal_manifest())
        manifest["action"]["scale"] = [0.5, 0.5]
        with pytest.raises(ValueError, match="action.scale"):
            validate_manifest(manifest)

    def test_missing_policy_group_raises(self):
        manifest = jsonable(_minimal_manifest())
        manifest["obs"]["policy_group"] = "actor"
        with pytest.raises(ValueError, match="policy_group"):
            validate_manifest(manifest)
