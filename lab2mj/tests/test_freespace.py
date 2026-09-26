"""Unit tests for the free-space plant-parity replay helpers (offline; skip without bundles)."""

from __future__ import annotations

import numpy as np
import pytest

from lab2mj.freespace import (
    assert_actuator_order,
    build_stiction,
    disable_contacts,
    disable_joint_limits,
    divergence_stats,
    implicit_pd_damping_selection,
    load_freespace_model,
    quat_angle_rad,
    replay_phase,
    zero_implicit_pd_damping,
)

from .shared import REPO_ROOT

SPOT_BUNDLE = REPO_ROOT / "data" / "mj_bundles" / "spot_velocity"
G1_BUNDLE = REPO_ROOT / "data" / "mj_bundles" / "g1_flat"

needs_spot_bundle = pytest.mark.skipif(
    not (SPOT_BUNDLE / "manifest.json").exists(), reason="requires data/mj_bundles/spot_velocity"
)
needs_g1_bundle = pytest.mark.skipif(
    not (G1_BUNDLE / "manifest.json").exists(), reason="requires data/mj_bundles/g1_flat"
)


@pytest.fixture()
def spot():
    return load_freespace_model(SPOT_BUNDLE)


@pytest.fixture()
def g1():
    return load_freespace_model(G1_BUNDLE)


@needs_spot_bundle
class TestDampingSelectionSpot:
    def test_selection_empty_without_implicit_pd(self, spot):
        _, _, manifest = spot
        names, kd, viscous = implicit_pd_damping_selection(manifest)
        assert names == []
        assert kd.shape == (0,)
        assert viscous.shape == (0,)


@needs_spot_bundle
class TestFrictionConversionSpot:
    def test_model_frictionloss_is_zero(self, spot):
        # Spot authors only the STATIC friction effort (PhysX >= 5 breakaway
        # threshold); a moving PhysX joint has zero Coulomb friction, so the
        # compiled model must carry dof_frictionloss = dynamic_friction = 0.
        model, robot_map, _ = spot
        np.testing.assert_array_equal(model.dof_frictionloss[robot_map.dof_adr], 0.0)

    def test_build_stiction_covers_all_joints(self, spot):
        model, robot_map, manifest = spot
        stiction = build_stiction(model, robot_map, manifest)
        assert stiction.active
        assert stiction.dof_adr.size == len(manifest["robot"]["isaac_joint_order"])


@needs_g1_bundle
class TestFrictionConversionG1:
    def test_stiction_inactive_without_static_friction(self, g1):
        model, robot_map, manifest = g1
        stiction = build_stiction(model, robot_map, manifest)
        assert not stiction.active


@needs_g1_bundle
class TestDampingSelectionG1:
    def test_selection_covers_all_joints_with_manifest_kd(self, g1):
        model, robot_map, manifest = g1
        names, kd, viscous = implicit_pd_damping_selection(manifest)
        assert sorted(names) == sorted(manifest["robot"]["isaac_joint_order"])
        for joint_name, kd_value, viscous_value in zip(names, kd, viscous):
            dof = int(model.jnt_dofadr[model.joint(joint_name).id])
            assert model.dof_damping[dof] == pytest.approx(kd_value + viscous_value)

    def test_zeroing_clears_all_joint_damping(self, g1):
        model, robot_map, manifest = g1
        zeroed = zero_implicit_pd_damping(model, manifest)
        assert sorted(zeroed) == sorted(manifest["robot"]["isaac_joint_order"])
        np.testing.assert_array_equal(model.dof_damping[robot_map.dof_adr], 0.0)

    def test_zeroing_rejects_stale_model_damping(self, g1):
        model, robot_map, manifest = g1
        model.dof_damping[robot_map.dof_adr[0]] += 1.0
        with pytest.raises(ValueError, match="dof_damping"):
            zero_implicit_pd_damping(model, manifest)


@needs_spot_bundle
class TestModelEdits:
    def test_actuator_order_matches_manifest(self, spot):
        model, _, manifest = spot
        assert_actuator_order(model, manifest)

    def test_actuator_order_rejects_permuted_manifest(self, spot):
        model, _, manifest = spot
        corrupted = dict(manifest)
        corrupted["robot"] = dict(manifest["robot"])
        mj_order = list(manifest["robot"]["mj_joint_order"])
        corrupted["robot"]["mj_joint_order"] = mj_order[::-1]
        with pytest.raises(ValueError, match="ctrl slot"):
            assert_actuator_order(model, corrupted)


class TestQuatAngle:
    def test_identical_and_antipodal_are_zero(self):
        q = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
        np.testing.assert_allclose(quat_angle_rad(q, q), 0.0, atol=1e-7)
        np.testing.assert_allclose(quat_angle_rad(q, -q), 0.0, atol=1e-7)

    def test_quarter_turn(self):
        q_id = np.array([1.0, 0.0, 0.0, 0.0])
        q_90z = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
        assert quat_angle_rad(q_id, q_90z) == pytest.approx(np.pi / 2)


class TestDivergenceStats:
    def test_extremes_and_marks(self):
        dt = 0.005
        num_steps, num_joints = 400, 3
        dq = np.zeros((num_steps, num_joints))
        dq[99, 1] = -2.0e-3  # state after step 99 == the 0.5 s mark
        dqd = np.full((num_steps, num_joints), 1.0e-4)
        pos_err = np.linspace(0.0, 0.01, num_steps)
        quat_err = np.zeros(num_steps)
        stats = divergence_stats(
            dq=dq, dqd=dqd, root_pos_err_m=pos_err, root_quat_err_rad=quat_err, dt=dt, marks_s=(0.5, 1.0, 5.0)
        )
        assert stats["duration_s"] == pytest.approx(2.0)
        assert stats["max_abs_dq_rad"] == pytest.approx(2.0e-3)
        assert stats["final_root_pos_err_m"] == pytest.approx(0.01)
        assert set(stats["at"]) == {"0.5s", "1s"}  # 5 s is beyond the horizon
        assert stats["at"]["0.5s"]["step"] == 99
        assert stats["at"]["0.5s"]["max_abs_dq_rad"] == pytest.approx(2.0e-3)
        assert stats["at"]["1s"]["max_abs_dq_rad"] == 0.0


@needs_spot_bundle
class TestReplayPhase:
    @staticmethod
    def _freespace_setup():
        model, robot_map, manifest = load_freespace_model(SPOT_BUNDLE)
        model.opt.gravity[:] = (0.0, 0.0, -9.81)
        disable_contacts(model)
        disable_joint_limits(model)
        zero_implicit_pd_damping(model, manifest)
        return model, robot_map, manifest

    @staticmethod
    def _rest_init(manifest):
        num_joints = len(manifest["robot"]["isaac_joint_order"])
        return {
            "root_pos_w": np.zeros(3),
            "root_quat_w": np.array([1.0, 0.0, 0.0, 0.0]),
            "root_link_lin_vel_w": np.zeros(3),
            "root_ang_vel_w": np.zeros(3),
            "joint_pos": np.asarray(manifest["robot"]["default_joint_pos_isaac"], dtype=np.float64),
            "joint_vel": np.zeros(num_joints),
        }

    def test_torque_free_fall_matches_closed_form(self):
        model, robot_map, manifest = self._freespace_setup()
        substeps = int(manifest["timing"]["physics_substeps"])
        num_steps, num_joints = 100, len(manifest["robot"]["isaac_joint_order"])
        init = self._rest_init(manifest)
        result = replay_phase(
            model, robot_map, manifest, init=init, tau_isaac=np.zeros((num_steps, num_joints)), substeps=substeps
        )
        # Semi-implicit Euler from rest: z_N = -g h^2 N (N + 1) / 2 with h the substep dt.
        h = model.opt.timestep
        n_sub = num_steps * substeps
        z_expected = -9.81 * h * h * n_sub * (n_sub + 1) / 2.0
        assert result.root_pos_w[-1, 2] == pytest.approx(z_expected, rel=1e-6)
        np.testing.assert_allclose(result.root_pos_w[:, :2], 0.0, atol=1e-9)
        # Uniform free fall exerts no joint-space forces; frictionloss keeps the joints at rest.
        np.testing.assert_allclose(result.joint_pos - init["joint_pos"][None, :], 0.0, atol=1e-9)
        np.testing.assert_allclose(result.joint_vel, 0.0, atol=1e-9)
        # Recorded CoM and link-origin velocities agree at identity attitude with zero angular velocity.
        np.testing.assert_allclose(result.root_lin_vel_w, result.root_link_lin_vel_w, atol=1e-9)
