"""Unit tests for the PhysX static-friction (stiction) emulation."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from lab2mj.stiction import SLIP_EPS, JointStiction

# A gravity pendulum: hinge about +Y at the body origin, rod CoM 0.5 m out, so
# gravity applies torque m*g*0.5*sin(q) about the joint (zero at the bottom).
PENDULUM_XML = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 1 0" frictionloss="{frictionloss}"/>
      <geom type="capsule" fromto="0 0 0  0 0 -1" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j" gainprm="1 0 0"/>
  </actuator>
</mujoco>
"""

STATIC = 0.5
DYNAMIC = 0.0


def _pendulum(frictionloss: float = DYNAMIC) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML.format(frictionloss=frictionloss))
    return model, mujoco.MjData(model)


def _stiction(model: mujoco.MjModel, static: float = STATIC, dynamic: float = DYNAMIC) -> JointStiction:
    return JointStiction(model, np.array([0]), np.array([static]), np.array([dynamic]), model.opt.timestep)


class TestConstruction:
    def test_inactive_when_static_not_above_dynamic(self):
        model, _ = _pendulum(frictionloss=0.3)
        stiction = JointStiction(model, np.array([0]), np.array([0.3]), np.array([0.3]), model.opt.timestep)
        assert not stiction.active

    def test_stale_frictionloss_raises(self):
        # The model must carry the DYNAMIC friction efforts; anything else is stale.
        model, _ = _pendulum(frictionloss=0.123)
        with pytest.raises(ValueError, match="dof_frictionloss"):
            _stiction(model)

    def test_inactive_apply_is_noop(self):
        model, data = _pendulum(frictionloss=0.3)
        stiction = JointStiction(model, np.array([0]), np.array([0.3]), np.array([0.3]), model.opt.timestep)
        before = model.dof_frictionloss.copy()
        stiction.apply(model, data)
        np.testing.assert_array_equal(model.dof_frictionloss, before)


class TestCaptureHoldRelease:
    @staticmethod
    def _run(model, data, stiction, steps):
        for _ in range(steps):
            stiction.apply(model, data)
            mujoco.mj_step(model, data)

    def test_swinging_pendulum_is_captured_at_the_turning_point(self):
        # Released near the bottom with a small push, the pendulum decelerates
        # naturally; when |qd| enters the capture window the static bound zeroes
        # it, and gravity's holding torque there (< static) cannot break it out.
        model, data = _pendulum()
        stiction = _stiction(model)
        data.qvel[0] = 0.3
        self._run(model, data, stiction, 200)
        assert abs(data.qvel[0]) < SLIP_EPS
        q_stuck = float(data.qpos[0])
        # Holding torque = m g L sin(q) must be below the static bound for the test to hold.
        assert 9.81 * 0.5 * abs(np.sin(q_stuck)) < STATIC
        self._run(model, data, stiction, 200)
        assert abs(data.qpos[0] - q_stuck) < 1e-6

    def test_breakaway_releases_to_dynamic_friction(self):
        model, data = _pendulum()
        stiction = _stiction(model)
        data.qvel[0] = 0.3
        self._run(model, data, stiction, 200)
        assert abs(data.qvel[0]) < SLIP_EPS
        # Torque above static + gravity load: the joint must slip, and once it
        # does the bound must drop back to the dynamic effort.
        data.ctrl[0] = STATIC + 9.81 * 0.5 + 0.5
        self._run(model, data, stiction, 10)
        assert abs(data.qvel[0]) > 0.1
        assert model.dof_frictionloss[0] == pytest.approx(DYNAMIC)

    def test_released_joint_does_not_restick_inside_the_window(self):
        # Constant just-above-static torque, no gravity, substeps=4: after breakaway
        # the joint accelerates from rest and stays inside the capture window for
        # several substeps. It must carry the dynamic bound the whole way (PhysX
        # applies the dynamic effort to any moving joint), not alternate
        # static/dynamic at ~half the breakaway effort. (A joint that DECELERATES
        # back into the window may re-stick — that is a genuine capture.)
        xml = PENDULUM_XML.format(frictionloss=DYNAMIC).replace('gravity="0 0 -9.81"', 'gravity="0 0 0"')
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        physics_dt = float(model.opt.timestep)
        model.opt.timestep = physics_dt / 4
        stiction = JointStiction(model, np.array([0]), np.array([STATIC]), np.array([DYNAMIC]), physics_dt)
        self._run(model, data, stiction, 2)  # capture and hold the resting joint
        assert stiction._held[0]
        data.ctrl[0] = STATIC + 0.02
        released = False
        for _ in range(100):
            stiction.apply(model, data)
            if released and abs(data.qvel[0]) > SLIP_EPS:
                assert model.dof_frictionloss[0] == pytest.approx(DYNAMIC), "released joint re-stuck mid-slip"
            released = released or bool(stiction._sliding[0])
            mujoco.mj_step(model, data)
        assert released
        assert abs(data.qvel[0]) > 0.05  # it kept accelerating instead of grinding

    def test_sub_breakaway_torque_does_not_release(self):
        model, data = _pendulum()
        stiction = _stiction(model)
        data.qvel[0] = 0.3
        self._run(model, data, stiction, 200)
        q_stuck = float(data.qpos[0])
        gravity_force = -9.81 * 0.5 * np.sin(q_stuck)  # gravity's generalized force at the joint
        # Cancel gravity and demand ~0.3 below the static bound: still stuck.
        data.ctrl[0] = -gravity_force + (STATIC - 0.3)
        self._run(model, data, stiction, 100)
        assert abs(data.qvel[0]) < SLIP_EPS
        assert abs(data.qpos[0] - q_stuck) < 1e-4

    def test_substep_capture_holds_static_bound_without_chatter(self):
        # substeps=4: the model integrates at physics_dt/4 while the capture window is
        # computed against the Isaac physics_dt, so one substep removes only ~1/4 of a
        # window-entering velocity and the capture spans several substeps. The bound
        # must stay STATIC for the whole capture (a slip-eps release mid-capture would
        # alternate static/dynamic and complete the capture physics steps late) and
        # the joint must still end up zeroed and held.
        model, data = _pendulum()
        physics_dt = float(model.opt.timestep)
        model.opt.timestep = physics_dt / 4
        stiction = JointStiction(model, np.array([0]), np.array([STATIC]), np.array([DYNAMIC]), physics_dt)
        data.qvel[0] = 0.3
        stuck_seen = False
        for _ in range(200 * 4):
            stiction.apply(model, data)
            if stuck_seen:
                assert model.dof_frictionloss[0] == pytest.approx(STATIC), "capture chattered to the dynamic bound"
            stuck_seen = stuck_seen or bool(stiction._stuck[0])
            mujoco.mj_step(model, data)
            if stuck_seen and abs(data.qvel[0]) < SLIP_EPS:
                break
        assert stuck_seen and abs(data.qvel[0]) < SLIP_EPS

    def test_reset_clears_stick_state(self):
        model, data = _pendulum()
        stiction = _stiction(model)
        data.qvel[0] = 0.3
        self._run(model, data, stiction, 200)
        stiction.reset()
        assert not stiction._stuck.any()


HINGE_XML = """
<mujoco>
  <option timestep="0.000625" gravity="0 0 0"/>
  <worldbody>
    <body name="link">
      <joint name="hinge" type="hinge" axis="0 0 1" frictionloss="1.0"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _coasting_friction_fraction(harden: bool) -> float:
    """Mean friction force / bound while a hinge coasts from 0.5 rad/s with no drive (above 0.05 rad/s)."""
    model = mujoco.MjModel.from_xml_string(HINGE_XML)
    data = mujoco.MjData(model)
    if harden:  # a Coulomb-only joint (static == dynamic): no stiction switch, hardened rows only
        stiction = JointStiction(model, np.array([0]), np.array([1.0]), np.array([1.0]), physics_dt=0.005)
        assert not stiction.active
    data.qvel[0] = 0.5
    fractions = []
    while data.qvel[0] > 0.05:
        mujoco.mj_step(model, data)
        rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
        fractions.append(-float(data.efc_force[rows].sum()) / 1.0)
    return float(np.mean(fractions))


def test_friction_rows_are_hardened_for_coulomb_joints():
    # With MuJoCo's default friction-row regularization a coasting joint feels a
    # velocity-ramped fraction of its Coulomb bound (~0.6 here); PhysX applies the full
    # dynamic effort to a moving joint, which the hardened rows reproduce.
    assert _coasting_friction_fraction(harden=False) < 0.8
    assert _coasting_friction_fraction(harden=True) == pytest.approx(1.0, abs=1e-3)
