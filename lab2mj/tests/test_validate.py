"""Offline tests for the validation gate math (`lab2mj.validate`) — no assets."""

from __future__ import annotations

import numpy as np
import pytest

from lab2mj.validate import (
    GATE_JOINT_TOL_RAD,
    GATE_ROOT_DZ_TOL_M,
    base_vel_b,
    dump_fixed_commands,
    trajectory_metrics,
)


def _identity_quats(n: int) -> np.ndarray:
    quats = np.zeros((n, 4), dtype=np.float64)
    quats[:, 0] = 1.0
    return quats


class TestTrajectoryMetrics:
    def _rec_and_dump(self, n_steps=10, n_joints=3, physics_dt=0.005, decimation=4):
        rec = {
            "joint_pos": np.zeros((n_steps, n_joints)),
            "root_pos_w": np.zeros((n_steps, 3)),
            "root_quat_w": _identity_quats(n_steps),
        }
        dump = {
            "joint_pos": np.zeros((n_steps, n_joints)),
            "root_pos_w": np.zeros((n_steps, 3)),
            "root_quat_w": _identity_quats(n_steps),
            "physics_dt": np.array(physics_dt),
            "decimation": np.array(decimation),
        }
        return rec, dump

    def test_error_outside_gate_window_does_not_gate(self):
        rec, dump = self._rec_and_dump()
        gate_steps = 5
        # Large divergence strictly after the gate window.
        rec["joint_pos"][gate_steps:, 0] = 1.0
        rec["root_pos_w"][gate_steps:, 2] = 1.0
        metrics, curves = trajectory_metrics(rec, dump, n_steps=10, gate_steps=gate_steps)
        assert metrics["max_joint_err_rad_gate_window"] == 0.0
        assert metrics["max_root_dz_m_gate_window"] == 0.0
        assert metrics["max_joint_err_rad_full"] == 1.0
        assert metrics["max_root_dz_m_full"] == 1.0
        assert curves["dq"].shape == (10, 3)
        # The pre-registered gate would pass on the window and fail on the full horizon.
        assert metrics["max_joint_err_rad_gate_window"] <= GATE_JOINT_TOL_RAD
        assert metrics["max_joint_err_rad_full"] > GATE_JOINT_TOL_RAD
        assert metrics["max_root_dz_m_full"] > GATE_ROOT_DZ_TOL_M

    def test_error_inside_gate_window_gates(self):
        rec, dump = self._rec_and_dump()
        rec["joint_pos"][2, 1] = 0.06
        metrics, _ = trajectory_metrics(rec, dump, n_steps=10, gate_steps=5)
        assert metrics["max_joint_err_rad_gate_window"] == 0.06
        assert metrics["max_joint_err_rad_gate_window"] > GATE_JOINT_TOL_RAD

    def test_window_and_horizon_seconds(self):
        rec, dump = self._rec_and_dump(physics_dt=0.005, decimation=4)
        metrics, _ = trajectory_metrics(rec, dump, n_steps=10, gate_steps=5)
        assert metrics["gate_window_s"] == 5 * 0.02
        assert metrics["horizon_s"] == 10 * 0.02


def test_base_vel_b_yaw_invariant_xy():
    yaw = np.pi / 2
    quats = np.array([[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]])
    lin_vel_w = np.array([[0.0, 1.0, 0.0]])  # world +y == body +x after a +90 deg yaw
    ang_vel_w = np.array([[0.0, 0.0, 0.5]])
    out = base_vel_b(quats, lin_vel_w, ang_vel_w)
    np.testing.assert_allclose(out[0], [1.0, 0.0, 0.5], atol=1e-12)


class TestDumpFixedCommands:
    def _manifest(self, commands):
        return {"commands": commands}

    def test_velocity_term_pins_cli_command(self):
        manifest = self._manifest([{"name": "base_velocity", "type": "UniformVelocityCommand"}])
        dump = {"command": np.array([1.0, 0.0, 0.2])}
        fixed = dump_fixed_commands(manifest, dump)
        np.testing.assert_array_equal(fixed["base_velocity"], [1.0, 0.0, 0.2])

    def test_pose_term_pins_world_goal_when_recorded(self):
        manifest = self._manifest(
            [
                {"name": "base_velocity", "type": "UniformVelocityCommand"},
                {"name": "pose_command", "type": "TerrainBasedPose2dCommand"},
            ]
        )
        dump = {
            "command": np.array([0.5, 0.0, 0.0]),
            "pose_command": np.array([2.0, 1.0, 0.6, 0.3]),
            # Body-frame per-step tensor must NOT be used for the world-frame pin.
            "command_pose_command": np.zeros((7, 4)),
        }
        fixed = dump_fixed_commands(manifest, dump)
        np.testing.assert_array_equal(fixed["pose_command"], [2.0, 1.0, 0.6, 0.3])
        np.testing.assert_array_equal(fixed["base_velocity"], [0.5, 0.0, 0.0])

    def test_pose_term_without_recorded_goal_stays_unpinned(self):
        manifest = self._manifest([{"name": "pose_command", "type": "UniformPose2dCommand"}])
        fixed = dump_fixed_commands(manifest, {"command": np.zeros(3)})
        assert "pose_command" not in fixed  # strict MjEnv pins it to zeros

    def test_non_strict_dump_rejected(self):
        # The dumper applies the CLI command only in strict mode; pinning it for a
        # --no-strict dump would score every gate against a phantom command.
        manifest = self._manifest([{"name": "base_velocity", "type": "UniformVelocityCommand"}])
        dump = {"command": np.zeros(3), "strict": np.array(False)}
        with pytest.raises(ValueError, match="no-strict"):
            dump_fixed_commands(manifest, dump)

    def test_scalar_term_pins_first_recorded_row(self):
        manifest = self._manifest([{"name": "vel_limit", "type": "VelocityLimitCommand"}])
        dump = {"command": np.zeros(3), "command_vel_limit": np.full((5, 1), 1.7)}
        fixed = dump_fixed_commands(manifest, dump)
        np.testing.assert_array_equal(fixed["vel_limit"], [1.7])
