"""Tests for lab2mj.actuators against the Spot and G1 env.yaml fixtures."""

import warnings
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from lab2mj.actuators import ActuatorSet, resolve_matching_names, resolve_matching_names_values
from lab2mj.ir import ActuatorGroupIR

from .shared import G1_ENV_YAML, SPOT_ENV_YAML, parse_fixture

SPOT_JOINT_ORDER = [
    "fl_hx", "fr_hx", "hl_hx", "hr_hx",
    "fl_hy", "fr_hy", "hl_hy", "hr_hy",
    "fl_kn", "fr_kn", "hl_kn", "hr_kn",
]  # fmt: skip

# Articulation joint order of the Isaac-Velocity-Flat-G1-v0 robot (37 joints,
# PhysX breadth-first). The ActuatorSet math only needs a fixed order, not this
# exact one, but the manifest will carry the true order so the test uses it.
G1_JOINT_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint",
    "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint",
    "left_six_joint", "left_four_joint", "left_one_joint",
    "right_six_joint", "right_four_joint", "right_one_joint",
    "left_two_joint", "right_two_joint",
]  # fmt: skip


def _spot_actuator_set() -> ActuatorSet:
    ir = parse_fixture(SPOT_ENV_YAML)
    return ActuatorSet.from_ir(ir.actuators, SPOT_JOINT_ORDER)


def _delay_group_ir(min_delay: int, max_delay: int, kp: float = 1.0, kd: float = 0.0) -> ActuatorGroupIR:
    return ActuatorGroupIR(
        name="grp",
        joint_names_expr=["a", "b"],
        model="delayed_pd",
        stiffness=kp,
        damping=kd,
        min_delay=min_delay,
        max_delay=max_delay,
    )


class TestSpotFixture:
    def test_groups_and_gains(self):
        aset = _spot_actuator_set()
        models = {g.name: g.model for g in aset.groups}
        assert models == {"spot_hip": "delayed_pd", "spot_knee": "remotized_pd"}
        np.testing.assert_allclose(aset.kp, 60.0)
        np.testing.assert_allclose(aset.kd, 1.5)
        # Hips: DelayedPD effort box of 90; knees: RemotizedPD forces the box to inf.
        np.testing.assert_allclose(aset.effort_limit[:8], 90.0)
        assert np.all(np.isinf(aset.effort_limit[8:]))

    def test_group_resolution_follows_joint_list_order(self):
        aset = _spot_actuator_set()
        by_name = {g.name: g for g in aset.groups}
        np.testing.assert_array_equal(by_name["spot_hip"].joint_ids, np.arange(8))
        np.testing.assert_array_equal(by_name["spot_knee"].joint_ids, np.arange(8, 12))
        assert by_name["spot_knee"].joint_names == ["fl_kn", "fr_kn", "hl_kn", "hr_kn"]

    def test_knee_lut_and_delay_ranges(self):
        ir = parse_fixture(SPOT_ENV_YAML)
        knee_ir = next(a for a in ir.actuators if a.name == "spot_knee")
        assert knee_ir.joint_parameter_lookup is not None
        assert knee_ir.joint_parameter_lookup.shape == (101, 3)
        aset = _spot_actuator_set()
        by_name = {g.name: g for g in aset.groups}
        assert (by_name["spot_hip"].min_delay, by_name["spot_hip"].max_delay) == (0, 2)
        assert (by_name["spot_knee"].min_delay, by_name["spot_knee"].max_delay) == (0, 1)
        assert by_name["spot_knee"].lut_angle is not None
        assert by_name["spot_knee"].lut_angle.shape == (101,)
        assert np.all(np.diff(by_name["spot_knee"].lut_angle) > 0)

    def test_static_friction_does_not_convert(self):
        # Spot authors only the static friction effort (PhysX >= 5: a breakaway
        # threshold on stationary joints) — no dynamic/viscous friction and no
        # armature, so no model overrides and a warning about the dropped breakaway.
        with pytest.warns(UserWarning, match="static friction"):
            aset = _spot_actuator_set()
        assert aset.builder_overrides() == {}
        np.testing.assert_allclose(aset.viscous_friction, 0.0)

    def test_dynamic_and_viscous_friction_convert(self):
        groups = [
            ActuatorGroupIR(
                name="grp",
                joint_names_expr=["a", "b"],
                model="ideal_pd",
                stiffness=10.0,
                damping=1.0,
                effort_limit=50.0,
                friction=0.2,
                dynamic_friction=0.2,
                viscous_friction=0.05,
            )
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # static == dynamic: no breakaway warning
            aset = ActuatorSet.from_ir(groups, ["a", "b"])
        overrides = aset.builder_overrides()
        for name in ("a", "b"):
            assert overrides[name] == {
                "frictionloss": pytest.approx(0.2),
                "damping": pytest.approx(0.05),
            }
        np.testing.assert_allclose(aset.viscous_friction, 0.05)


class TestRemotizedCrossCheck:
    """Cross-check torque clamping against the legacy isaac_parity implementation.

    States are constructed so the float32 arithmetic inside
    ``compute_remotized_torques`` is exact (dyadic rationals; knee angles on
    LUT knots whenever the knee clamp is active), making a 1e-10 comparison
    against the float64 ActuatorSet meaningful.
    """

    N_STATES = 200

    def test_agrees_with_isaac_parity(self):
        pytest.importorskip("mujoco")
        pytest.importorskip("torch")
        isaac_parity = pytest.importorskip("contact_lab.utils.mj.isaac_parity")

        ir = parse_fixture(SPOT_ENV_YAML)
        knee_ir = next(a for a in ir.actuators if a.name == "spot_knee")
        assert knee_ir.joint_parameter_lookup is not None
        # The legacy module stores the same LUT decimals rounded to float32;
        # substitute those exact values so the comparison is rounding-free.
        lut = knee_ir.joint_parameter_lookup.copy()
        np.testing.assert_allclose(lut[:, 0], isaac_parity._KNEE_LUT_ANGLES, atol=1e-6, rtol=0)
        np.testing.assert_allclose(lut[:, 2], isaac_parity._KNEE_LUT_MAX_TORQUE, atol=2e-5, rtol=0)
        lut[:, 0] = isaac_parity._KNEE_LUT_ANGLES.astype(np.float64)
        lut[:, 2] = isaac_parity._KNEE_LUT_MAX_TORQUE.astype(np.float64)
        knee_ir.joint_parameter_lookup = lut

        aset = ActuatorSet.from_ir(ir.actuators, SPOT_JOINT_ORDER)
        aset.reset(strict=True)  # lag pinned to 0
        assert all(g.lag == 0 for g in aset.groups)
        kp = float(aset.kp[0])
        kd = float(aset.kd[0])
        assert kp == 60.0 and kd == 1.5

        # Stand-in for a MuJoCo model whose actuator order equals Isaac order.
        model = SimpleNamespace(nu=12, jnt_qposadr=np.arange(12), jnt_dofadr=np.arange(12))
        is_knee = np.array([name.endswith("_kn") for name in SPOT_JOINT_ORDER])
        joint_idx = np.arange(12)
        angles32 = isaac_parity._KNEE_LUT_ANGLES.astype(np.float64)

        rng = np.random.default_rng(42)

        def dyadic(lo: int, hi: int, size: int) -> np.ndarray:
            return rng.integers(lo, hi + 1, size=size).astype(np.float64) / 1024.0

        n_hip_clamped = n_knee_clamped = n_knee_free = 0
        max_err = 0.0
        for _ in range(self.N_STATES):
            q = np.zeros(12)
            qd = np.zeros(12)
            target = np.zeros(12)
            # Hips: dyadic states; torques reach ~246 Nm so the 90 Nm box engages.
            q[:8] = dyadic(-2048, 2048, 8)
            qd[:8] = dyadic(-4096, 4096, 8)
            target[:8] = dyadic(-2048, 2048, 8)
            # Knees: either saturate the LUT clamp (angle on a knot, |tau| >= 144 Nm
            # vs a <= 113.3 Nm envelope) or stay well inside it (|tau| <= 27 Nm vs
            # a >= 75 Nm envelope on the sampled angle range).
            for k in range(8, 12):
                if rng.random() < 0.5:
                    q[k] = angles32[rng.integers(0, 101)]
                    delta = float(rng.choice([-1.0, 1.0])) * float(rng.integers(2560, 4097)) / 1024.0
                    target[k] = q[k] + delta
                    qd[k] = dyadic(-4096, 4096, 1)[0]
                    n_knee_clamped += 1
                else:
                    q[k] = dyadic(-2048, -768, 1)[0]
                    target[k] = q[k] + dyadic(-409, 409, 1)[0]
                    qd[k] = dyadic(-2048, 2048, 1)[0]
                    n_knee_free += 1

            q_des, qd_des = aset.step_delay_buffers(target)
            tau = aset.compute_torques(q, qd, q_des, qd_des)
            data = SimpleNamespace(qpos=q, qvel=qd)
            tau_ref = isaac_parity.compute_remotized_torques(model, data, target, kp, kd, is_knee, joint_idx)
            max_err = max(max_err, float(np.max(np.abs(tau - tau_ref))))
            n_hip_clamped += int(np.sum(np.abs(tau[:8]) == 90.0))

        assert max_err <= 1e-10, f"max |tau - tau_ref| = {max_err}"
        assert n_hip_clamped > 100
        assert n_knee_clamped > 100 and n_knee_free > 100


class TestDelayBuffer:
    @pytest.mark.parametrize("lag", [0, 1, 2])
    def test_lag_semantics(self, lag):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=lag, max_delay=lag)], ["a", "b"])
        aset.reset(np.random.default_rng(0))
        assert aset.groups[0].lag == lag
        pushes = []
        for t in range(6):
            x = np.array([float(t + 1), -float(t + 1)])
            pushes.append(x)
            q_des, qd_des = aset.step_delay_buffers(x, 10.0 * x)
            expected = pushes[max(0, t - lag)]
            np.testing.assert_array_equal(q_des, expected)
            np.testing.assert_array_equal(qd_des, 10.0 * expected)

    def test_reset_refills_with_first_push(self):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=2, max_delay=2)], ["a", "b"])
        rng = np.random.default_rng(0)
        aset.reset(rng)
        for t in range(5):
            aset.step_delay_buffers(np.array([float(t), float(t)]))
        aset.reset(rng)
        # First push after reset fills the whole buffer: no stale pre-reset data.
        q_des, _ = aset.step_delay_buffers(np.array([42.0, -42.0]))
        np.testing.assert_array_equal(q_des, [42.0, -42.0])
        q_des, _ = aset.step_delay_buffers(np.array([43.0, -43.0]))
        np.testing.assert_array_equal(q_des, [42.0, -42.0])
        q_des, _ = aset.step_delay_buffers(np.array([44.0, -44.0]))
        np.testing.assert_array_equal(q_des, [42.0, -42.0])
        q_des, _ = aset.step_delay_buffers(np.array([45.0, -45.0]))
        np.testing.assert_array_equal(q_des, [43.0, -43.0])

    def test_reset_resamples_lag_within_bounds(self):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=0, max_delay=2)], ["a", "b"])
        rng = np.random.default_rng(7)
        lags = set()
        for _ in range(100):
            aset.reset(rng)
            lags.add(aset.groups[0].lag)
        assert lags == {0, 1, 2}

    def test_strict_reset_pins_lag_zero(self):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=2, max_delay=2)], ["a", "b"])
        aset.reset(strict=True)
        assert aset.groups[0].lag == 0
        for t in range(4):
            x = np.array([float(t), -float(t)])
            q_des, _ = aset.step_delay_buffers(x)
            np.testing.assert_array_equal(q_des, x)

    def test_set_lags_overrides_strict_pin(self):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=0, max_delay=2)], ["a", "b"])
        aset.reset(strict=True)
        aset.set_lags({"grp": 2})
        assert aset.groups[0].lag == 2
        first = np.array([1.0, -1.0])
        aset.step_delay_buffers(first)
        aset.step_delay_buffers(np.array([2.0, -2.0]))
        q_des, _ = aset.step_delay_buffers(np.array([3.0, -3.0]))
        np.testing.assert_array_equal(q_des, first)

    def test_set_lags_unknown_group_raises(self):
        aset = ActuatorSet.from_ir([_delay_group_ir(min_delay=0, max_delay=2)], ["a", "b"])
        aset.reset(strict=True)
        with pytest.raises(KeyError, match="unknown group 'nope'"):
            aset.set_lags({"nope": 1})


class TestPartialDictParams:
    """Regex-dict params that cover only part of a group: IsaacLab's
    ``_parse_joint_parameter`` zero-fills the unmatched joints, so the mirror
    must too — but loudly, since a silent 0.0 effort limit disables the joint."""

    def test_partial_effort_limit_dict_zero_fills_with_warning(self):
        groups = [
            ActuatorGroupIR(
                name="g1",
                joint_names_expr=[".*"],
                model="ideal_pd",
                stiffness=10.0,
                damping=1.0,
                effort_limit={"a": 50.0},
            )
        ]
        with pytest.warns(UserWarning, match=r"effort_limit.*\['b'\].*zero-filled"):
            aset = ActuatorSet.from_ir(groups, ["a", "b"])
        np.testing.assert_array_equal(aset.effort_limit, [50.0, 0.0])

    def test_partial_armature_dict_zero_fills_overrides(self):
        # IsaacLab writes the whole zero-filled buffer to sim, so the unmatched
        # joint carries an explicit armature of 0.0 in the builder overrides.
        groups = [
            ActuatorGroupIR(
                name="g1",
                joint_names_expr=[".*"],
                model="ideal_pd",
                stiffness=10.0,
                damping=1.0,
                armature={"a": 0.01},
            )
        ]
        with pytest.warns(UserWarning, match="armature"):
            aset = ActuatorSet.from_ir(groups, ["a", "b"])
        overrides = aset.builder_overrides()
        assert overrides["a"] == {"armature": pytest.approx(0.01)}
        assert overrides["b"] == {"armature": 0.0}

    def test_full_coverage_dict_does_not_warn(self):
        groups = [
            ActuatorGroupIR(
                name="g1",
                joint_names_expr=[".*"],
                model="ideal_pd",
                stiffness={"a": 1.0, "b": 2.0},
                damping=0.1,
                effort_limit=50.0,
            )
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            aset = ActuatorSet.from_ir(groups, ["a", "b"])
        np.testing.assert_array_equal(aset.kp, [1.0, 2.0])


class TestNoneGainsRejected:
    """None stiffness/damping means "use the USD-authored drive gains" in IsaacLab;
    the IR does not carry USD gains, so from_ir must refuse instead of zeroing."""

    @pytest.mark.parametrize("gain", ["stiffness", "damping"])
    def test_none_gain_raises(self, gain):
        group = ActuatorGroupIR(
            name="g1",
            joint_names_expr=[".*"],
            model="implicit_pd",
            stiffness=None if gain == "stiffness" else 10.0,
            damping=None if gain == "damping" else 1.0,
        )
        with pytest.raises(ValueError, match=f"{gain} is None.*USD-authored"):
            ActuatorSet.from_ir([group], ["a", "b"])


class TestRegexResolution:
    def test_joint_in_two_groups_errors(self):
        groups = [
            ActuatorGroupIR(name="g1", joint_names_expr=[".*"], model="ideal_pd", stiffness=1.0, damping=0.1),
            ActuatorGroupIR(name="g2", joint_names_expr=["b"], model="ideal_pd", stiffness=1.0, damping=0.1),
        ]
        with pytest.raises(ValueError, match="claimed by both"):
            ActuatorSet.from_ir(groups, ["a", "b"])

    def test_unmatched_pattern_errors(self):
        groups = [
            ActuatorGroupIR(
                name="g1", joint_names_expr=["a", "missing_.*"], model="ideal_pd", stiffness=1.0, damping=0.1
            )
        ]
        with pytest.raises(ValueError, match="match none"):
            ActuatorSet.from_ir(groups, ["a"])

    def test_joint_matching_two_patterns_in_one_group_errors(self):
        groups = [
            ActuatorGroupIR(name="g1", joint_names_expr=[".*", "b"], model="ideal_pd", stiffness=1.0, damping=0.1)
        ]
        with pytest.raises(ValueError, match="multiple patterns"):
            ActuatorSet.from_ir(groups, ["a", "b"])

    def test_uncovered_joint_errors(self):
        groups = [ActuatorGroupIR(name="g1", joint_names_expr=["a"], model="ideal_pd", stiffness=1.0, damping=0.1)]
        with pytest.raises(ValueError, match="not covered"):
            ActuatorSet.from_ir(groups, ["a", "b"])

    def test_ordering_follows_joint_list(self):
        indices = resolve_matching_names(["b|d", "a|c"], ["a", "b", "c", "d"])
        assert indices == [0, 1, 2, 3]
        indices, values = resolve_matching_names_values({"a|d|e": 1.0, "b|c": 2.0}, ["a", "b", "c", "d", "e"])
        assert indices == [0, 1, 2, 3, 4]
        assert values == [1.0, 2.0, 2.0, 1.0, 1.0]


class TestG1Fixture:
    def _aset(self) -> ActuatorSet:
        ir = parse_fixture(G1_ENV_YAML)
        return ActuatorSet.from_ir(ir.actuators, G1_JOINT_ORDER)

    def test_effort_limits_per_source_cfg(self):
        aset = self._aset()
        legs = ("_hip_yaw_joint", "_hip_roll_joint", "_hip_pitch_joint", "_knee_joint", "torso_joint")
        feet = ("_ankle_pitch_joint", "_ankle_roll_joint")
        for name, limit in zip(aset.joint_names, aset.effort_limit):
            if name.endswith(legs):
                assert limit == 300.0, name
            elif name.endswith(feet):
                assert limit == 20.0, name
            else:
                assert limit == 300.0, name
        assert np.all(np.isfinite(aset.effort_limit))

    def test_per_joint_gains_from_regex_dicts(self):
        aset = self._aset()
        kp = dict(zip(aset.joint_names, aset.kp))
        kd = dict(zip(aset.joint_names, aset.kd))
        assert kp["left_hip_yaw_joint"] == 150.0
        assert kp["right_hip_roll_joint"] == 150.0
        assert kp["left_hip_pitch_joint"] == 200.0
        assert kp["right_knee_joint"] == 200.0
        assert kp["torso_joint"] == 200.0
        assert kp["left_ankle_pitch_joint"] == 20.0
        assert kp["right_ankle_roll_joint"] == 20.0
        assert kp["left_shoulder_yaw_joint"] == 40.0
        assert kd["torso_joint"] == 5.0
        assert kd["right_ankle_pitch_joint"] == 2.0
        assert kd["left_elbow_roll_joint"] == 10.0

    def test_armature_overrides_populated(self):
        overrides = self._aset().builder_overrides()
        assert set(overrides) == set(G1_JOINT_ORDER)
        finger = ("_five_joint", "_three_joint", "_six_joint", "_four_joint", "_zero_joint", "_one_joint", "_two_joint")
        for name, entry in overrides.items():
            assert "frictionloss" not in entry, name
            expected = 0.001 if name.endswith(finger) else 0.01
            assert entry["armature"] == pytest.approx(expected), name

    def test_implicit_torque_clipped_to_effort_limit(self):
        aset = self._aset()
        n = aset.num_joints
        zeros = np.zeros(n)
        tau = aset.compute_torques(zeros, zeros, np.full(n, 100.0))
        np.testing.assert_allclose(tau, aset.effort_limit)
        tau = aset.compute_torques(zeros, zeros, np.full(n, -100.0))
        np.testing.assert_allclose(tau, -aset.effort_limit)
        np.testing.assert_allclose(aset.compute_torques(zeros, zeros, zeros), 0.0)

    def test_pd_formula_with_velocity_target(self):
        aset = self._aset()
        n = aset.num_joints
        q = np.full(n, 0.1)
        qd = np.full(n, -0.2)
        q_t = np.full(n, 0.3)
        qd_t = np.full(n, 0.5)
        expected = np.clip(aset.kp * (q_t - q) + aset.kd * (qd_t - qd), -aset.effort_limit, aset.effort_limit)
        np.testing.assert_allclose(aset.compute_torques(q, qd, q_t, qd_t), expected, rtol=0, atol=1e-14)


class TestContactProfileSelection:
    """The engagement contact profile follows the actuator family (soft-gain DC motors)."""

    @staticmethod
    def _aset(model, **extra) -> ActuatorSet:
        kwargs: dict[str, Any] = dict(name="grp", joint_names_expr=["a", "b"], model=model)
        if model in ("dc_motor", "actuator_net_lstm", "actuator_net_mlp"):
            kwargs.update(effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0)
        if model in ("actuator_net_lstm", "actuator_net_mlp"):
            kwargs.update(network_file="net.pt")
        else:
            kwargs.update(stiffness=25.0, damping=0.5)
        if model == "actuator_net_mlp":
            kwargs.update(pos_scale=1.0, vel_scale=1.0, torque_scale=1.0, input_order="pos_vel", input_idx=[0, 1, 2])
        kwargs.update(extra)
        return ActuatorSet.from_ir([ActuatorGroupIR(**kwargs)], ["a", "b"])

    @pytest.mark.parametrize(
        "model,profile",
        [
            ("dc_motor", "engagement"),
            ("actuator_net_mlp", "engagement"),
            ("actuator_net_lstm", "default"),
            ("implicit_pd", "default"),
        ],
    )
    def test_profile_follows_actuator_family(self, model, profile):
        assert self._aset(model).contact_profile() == profile

    def test_mixed_groups_any_dc_motor_selects_engagement(self):
        groups = [
            ActuatorGroupIR(
                name="legs",
                joint_names_expr=["a"],
                model="dc_motor",
                stiffness=25.0,
                damping=0.5,
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
            ),
            ActuatorGroupIR(
                name="arm", joint_names_expr=["b"], model="implicit_pd", stiffness=100.0, damping=2.0, effort_limit=50.0
            ),
        ]
        aset = ActuatorSet.from_ir(groups, ["a", "b"])
        assert aset.contact_profile() == "engagement"

    def test_spot_fixture_keeps_default(self):
        assert _spot_actuator_set().contact_profile() == "default"


class TestTorqueSpeedEnvelope:
    ENVELOPE = np.array([97.0, -108.79, 25.03, -22.22, 9.48, -8.32])

    def test_clip_corners(self):
        from lab2mj.actuators import torque_speed_envelope_clip

        vel = np.array([-30.0, 0.0, 9.48, 17.255, 25.03, 30.0])
        np.testing.assert_allclose(
            torque_speed_envelope_clip(np.full(6, 200.0), vel, self.ENVELOPE), [97.0, 97.0, 97.0, 48.5, 0.0, 0.0]
        )
        vel = np.array([30.0, 0.0, -8.32, -15.27, -22.22, -30.0])
        np.testing.assert_allclose(
            torque_speed_envelope_clip(np.full(6, -200.0), vel, self.ENVELOPE),
            [-108.79, -108.79, -108.79, -54.395, 0.0, 0.0],
        )

    def test_applied_after_the_lookup(self):
        group = ActuatorGroupIR(
            name="knee",
            joint_names_expr=["a"],
            model="remotized_pd",
            stiffness=1000.0,
            damping=0.0,
            joint_parameter_lookup=np.array([[-1.0, 1.0, 50.0], [1.0, 1.0, 150.0]]),
            torque_speed_envelope=list(self.ENVELOPE),
        )
        aset = ActuatorSet.from_ir([group], ["a"])
        # lookup cap 100 at q=0; the envelope caps 97 at rest and 48.5 at 17.255 rad/s
        assert aset.compute_torques(np.zeros(1), np.zeros(1), np.ones(1))[0] == pytest.approx(97.0)
        assert aset.compute_torques(np.zeros(1), np.full(1, 17.255), np.ones(1))[0] == pytest.approx(48.5)
