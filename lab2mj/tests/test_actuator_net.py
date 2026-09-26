"""Tests for the dc_motor / actuator_net_lstm actuator models (offline).

The LSTM tests synthesize a tiny TorchScript network with the same I/O contract
as IsaacLab's ANYdrive actuator net (input ``(J, 1, 2)`` of ``[pos_error, vel]``,
state ``(layers, J, hidden)``); the real ANYdrive net test is gated on the
cached file under ``data/actuator_nets/``.
"""

from pathlib import Path

import numpy as np
import pytest

from lab2mj.actuators import ActuatorSet, dc_motor_clip
from lab2mj.env_yaml import _parse_actuators, actuator_model_from_class
from lab2mj.ir import ActuatorGroupIR

REPO = Path(__file__).resolve().parents[2]
ANYDRIVE_NET = REPO / "data" / "actuator_nets" / "anydrive_3_lstm_jit.pt"

ANYDRIVE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    "/Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"
)

# ANYdrive 3 parameters (ANYDRIVE_3_LSTM_ACTUATOR_CFG / ANYDRIVE_3_SIMPLE_ACTUATOR_CFG).
SAT, EFFORT_LIM, VEL_LIM = 120.0, 80.0, 7.5


def _lstm_dump_dict(network_file: str = ANYDRIVE_URL) -> dict:
    """Actuator entry shaped like a dump_yaml'd ActuatorNetLSTMCfg (ANYmal legs)."""
    return {
        "legs": {
            "class_type": "isaaclab.actuators.actuator_net:ActuatorNetLSTM",
            "joint_names_expr": [".*HAA", ".*HFE", ".*KFE"],
            "effort_limit": EFFORT_LIM,
            "velocity_limit": VEL_LIM,
            "effort_limit_sim": None,
            "velocity_limit_sim": None,
            "stiffness": None,
            "damping": None,
            "armature": None,
            "friction": None,
            "dynamic_friction": None,
            "viscous_friction": None,
            "network_file": network_file,
            "saturation_effort": SAT,
        }
    }


def _dc_motor_dump_dict() -> dict:
    """Actuator entry shaped like a dump_yaml'd DCMotorCfg (ANYDRIVE_3_SIMPLE)."""
    return {
        "legs": {
            "class_type": "isaaclab.actuators.actuator_pd:DCMotor",
            "joint_names_expr": [".*HAA", ".*HFE", ".*KFE"],
            "effort_limit": EFFORT_LIM,
            "velocity_limit": VEL_LIM,
            "effort_limit_sim": None,
            "velocity_limit_sim": None,
            "stiffness": {".*": 40.0},
            "damping": {".*": 5.0},
            "armature": None,
            "friction": None,
            "dynamic_friction": None,
            "viscous_friction": None,
            "saturation_effort": SAT,
        }
    }


def _dc_group_ir(
    joints: list[str],
    kp: float = 40.0,
    kd: float = 5.0,
    sat: float = SAT,
    effort_limit: float = EFFORT_LIM,
    velocity_limit: float = VEL_LIM,
) -> ActuatorGroupIR:
    return ActuatorGroupIR(
        name="dc",
        joint_names_expr=list(joints),
        model="dc_motor",
        stiffness=kp,
        damping=kd,
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=sat,
    )


def _lstm_group_ir(
    joints: list[str],
    network_file: str,
    sat: float = SAT,
    effort_limit: float = EFFORT_LIM,
    velocity_limit: float = VEL_LIM,
    network_bundle_path: str | None = None,
) -> ActuatorGroupIR:
    return ActuatorGroupIR(
        name="legs",
        joint_names_expr=list(joints),
        model="actuator_net_lstm",
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=sat,
        network_file=network_file,
        network_bundle_path=network_bundle_path,
    )


class TestEnvYamlParsing:
    def test_actuator_class_map(self):
        assert actuator_model_from_class("isaaclab.actuators.actuator_pd:DCMotor") == "dc_motor"
        assert actuator_model_from_class("isaaclab.actuators.actuator_net:ActuatorNetLSTM") == "actuator_net_lstm"
        with pytest.raises(ValueError, match="unsupported actuator class"):
            actuator_model_from_class("isaaclab.actuators.actuator_pd:FancyActuator")

    def test_lstm_dump_dict(self):
        (group,) = _parse_actuators(_lstm_dump_dict())
        assert group.model == "actuator_net_lstm"
        assert group.joint_names_expr == [".*HAA", ".*HFE", ".*KFE"]
        assert group.network_file == ANYDRIVE_URL
        assert group.saturation_effort == SAT
        assert group.effort_limit == EFFORT_LIM
        assert group.velocity_limit == VEL_LIM
        assert group.stiffness is None and group.damping is None
        assert group.network_bundle_path is None

    def test_dc_motor_dump_dict(self):
        (group,) = _parse_actuators(_dc_motor_dump_dict())
        assert group.model == "dc_motor"
        assert group.saturation_effort == SAT
        assert group.stiffness == {".*": 40.0}
        assert group.network_file is None

    def test_ir_round_trip_keeps_net_fields(self):
        group = _lstm_group_ir(["a"], ANYDRIVE_URL, network_bundle_path="assets/actuator_nets/net.pt")
        rebuilt = ActuatorGroupIR.from_dict(group.to_dict())
        assert rebuilt.model == "actuator_net_lstm"
        assert rebuilt.network_file == ANYDRIVE_URL
        assert rebuilt.network_bundle_path == "assets/actuator_nets/net.pt"
        assert rebuilt.saturation_effort == SAT


class TestDCMotorClipFunction:
    """Hand-computed values for sat=120, effort_limit=80, velocity_limit=7.5.

    Corner velocity: 7.5 * (1 + 80/120) = 12.5 rad/s.
    """

    def _clip(self, effort: float, vel: float) -> float:
        out = dc_motor_clip(
            np.array([effort]), np.array([vel]), np.array([SAT]), np.array([EFFORT_LIM]), np.array([VEL_LIM])
        )
        return float(out[0])

    def test_zero_velocity_box(self):
        assert self._clip(1000.0, 0.0) == 80.0
        assert self._clip(-1000.0, 0.0) == -80.0
        assert self._clip(30.0, 0.0) == 30.0

    def test_torque_speed_line_engages(self):
        # vel=5: max = 120*(1 - 5/7.5) = 40, min = max(120*(-1 - 5/7.5), -80) = -80.
        assert self._clip(1000.0, 5.0) == pytest.approx(40.0)
        assert self._clip(-1000.0, 5.0) == -80.0
        assert self._clip(39.0, 5.0) == 39.0
        # Symmetric quadrant, vel=-5.
        assert self._clip(1000.0, -5.0) == 80.0
        assert self._clip(-1000.0, -5.0) == pytest.approx(-40.0)

    def test_velocity_beyond_corner_pins_to_effort_limit(self):
        # vel clamped to 12.5: max = min(120*(1 - 12.5/7.5), 80) = -80 -> tau = -80 always
        # (up to the rounding of 12.5/7.5, which Isaac's own arithmetic shares).
        for effort in (1000.0, 0.0, -1000.0):
            assert self._clip(effort, 15.0) == pytest.approx(-80.0)
            assert self._clip(effort, -15.0) == pytest.approx(80.0)


class TestDCMotorGroup:
    def _aset(self, **kwargs) -> ActuatorSet:
        return ActuatorSet.from_ir([_dc_group_ir(["a", "b", "c"], **kwargs)], ["a", "b", "c"])

    def test_matches_hand_computed_curve(self):
        kp, kd = 40.0, 5.0
        aset = self._aset(kp=kp, kd=kd)
        rng = np.random.default_rng(3)
        for _ in range(50):
            q = rng.uniform(-2.0, 2.0, 3)
            qd = rng.uniform(-20.0, 20.0, 3)  # spans past the 12.5 corner velocity
            q_des = rng.uniform(-3.0, 3.0, 3)
            tau = aset.compute_torques(q, qd, q_des)
            pd = kp * (q_des - q) + kd * (0.0 - qd)
            vel = np.clip(qd, -12.5, 12.5)
            max_e = np.minimum(SAT * (1.0 - vel / VEL_LIM), EFFORT_LIM)
            min_e = np.maximum(SAT * (-1.0 - vel / VEL_LIM), -EFFORT_LIM)
            np.testing.assert_allclose(tau, np.clip(pd, min_e, max_e), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("missing", ["saturation_effort", "velocity_limit", "effort_limit"])
    def test_missing_curve_parameter_raises(self, missing):
        group = _dc_group_ir(["a"])
        setattr(group, missing, None)
        with pytest.raises(ValueError, match=missing):
            ActuatorSet.from_ir([group], ["a"])

    def test_missing_gains_still_raise(self):
        group = _dc_group_ir(["a"])
        group.stiffness = None
        with pytest.raises(ValueError, match="stiffness is None"):
            ActuatorSet.from_ir([group], ["a"])

    def test_nonpositive_curve_parameters_raise(self):
        group = _dc_group_ir(["a"], sat=0.0)
        with pytest.raises(ValueError, match="positive"):
            ActuatorSet.from_ir([group], ["a"])


torch = pytest.importorskip("torch")

HIDDEN_DIM, NUM_LAYERS = 8, 2


class _SeaNet(torch.nn.Module):
    """Tiny stand-in with the ANYdrive net's I/O contract (batch-first LSTM + linear head)."""

    def __init__(self, out_bias: float = 0.0, out_gain: float = 1.0):
        super().__init__()
        self.lstm = torch.nn.LSTM(2, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.out = torch.nn.Linear(HIDDEN_DIM, 1)
        with torch.no_grad():
            self.out.weight *= out_gain
            self.out.bias.fill_(out_bias)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        y, new_state = self.lstm(x, state)
        return self.out(y), new_state


def _write_net(path: Path, **kwargs) -> Path:
    torch.manual_seed(0)
    torch.jit.save(torch.jit.script(_SeaNet(**kwargs)), str(path))
    return path


@pytest.fixture(scope="module")
def tiny_net(tmp_path_factory) -> Path:
    return _write_net(tmp_path_factory.mktemp("nets") / "sea_net.pt")


def _lstm_aset(net_path: Path | str, joints: list[str] | None = None, **kwargs) -> ActuatorSet:
    joints = joints or ["a", "b", "c"]
    aset = ActuatorSet.from_ir([_lstm_group_ir(joints, str(net_path), **kwargs)], joints)
    aset.reset(strict=True)
    return aset


def _reference_rollout(
    net_path: Path, steps: list[tuple[np.ndarray, np.ndarray]], sat: float = SAT, effort_limit: float = EFFORT_LIM
) -> list[np.ndarray]:
    """Manual mirror of ActuatorNetLSTM: persistent state + zero-velocity DC clip."""
    net = torch.jit.load(str(net_path), map_location="cpu").eval()
    state_dict = net.lstm.state_dict()
    num_layers = len(state_dict) // 4
    hidden_dim = state_dict["weight_hh_l0"].shape[1]
    num_joints = steps[0][0].shape[0]
    hidden = torch.zeros(num_layers, num_joints, hidden_dim)
    cell = torch.zeros(num_layers, num_joints, hidden_dim)
    bound = min(sat, effort_limit)
    outs = []
    for pos_error, vel in steps:
        x = torch.zeros(num_joints, 1, 2)
        x[:, 0, 0] = torch.from_numpy(pos_error.astype(np.float32))
        x[:, 0, 1] = torch.from_numpy(vel.astype(np.float32))
        with torch.inference_mode():
            tau, (hidden, cell) = net(x, (hidden, cell))
        outs.append(np.clip(tau.reshape(-1).cpu().numpy().astype(np.float64), -bound, bound))
    return outs


class TestLSTMGroup:
    def test_matches_reference_with_persistent_state(self, tiny_net):
        aset = _lstm_aset(tiny_net)
        rng = np.random.default_rng(11)
        steps = [(rng.uniform(-0.5, 0.5, 3), rng.uniform(-3.0, 3.0, 3)) for _ in range(6)]
        expected = _reference_rollout(tiny_net, steps)
        q = np.zeros(3)
        for (pos_error, vel), ref in zip(steps, expected):
            tau = aset.compute_torques(q, vel, q + pos_error)
            np.testing.assert_allclose(tau, ref, rtol=0, atol=1e-7)
        # The hidden state must actually evolve: a frozen state would repeat outputs.
        assert not np.allclose(expected[0], expected[1])

    @pytest.mark.parametrize("strict", [True, False])
    def test_reset_clears_state(self, tiny_net, strict):
        aset = _lstm_aset(tiny_net)
        q, qd = np.zeros(3), np.zeros(3)
        fresh = aset.compute_torques(q, qd, np.full(3, 0.3))
        for _ in range(3):
            aset.compute_torques(q, qd, np.full(3, -0.7))
        if strict:
            aset.reset(strict=True)
        else:
            aset.reset(np.random.default_rng(0))
        np.testing.assert_array_equal(aset.compute_torques(q, qd, np.full(3, 0.3)), fresh)

    @pytest.mark.parametrize(
        ("sat", "effort_limit", "expected"),
        [(10.0, 4.0, 4.0), (10.0, 100.0, 10.0)],
    )
    def test_clip_is_zero_velocity_box(self, tmp_path, sat, effort_limit, expected):
        # Huge output bias saturates the clip; Isaac's ActuatorNetLSTM never writes the
        # DCMotor._joint_vel buffer, so the bound is +-min(sat, effort_limit) for ANY
        # joint velocity (the curve is evaluated at zero velocity).
        net_path = _write_net(tmp_path / "big.pt", out_bias=1e6)
        aset = _lstm_aset(net_path, sat=sat, effort_limit=effort_limit)
        for qd_value in (0.0, 5.0, -50.0):
            tau = aset.compute_torques(np.zeros(3), np.full(3, qd_value), np.zeros(3))
            np.testing.assert_allclose(tau, expected)
        net_path = _write_net(tmp_path / "big_neg.pt", out_bias=-1e6)
        aset = _lstm_aset(net_path, sat=sat, effort_limit=effort_limit)
        tau = aset.compute_torques(np.zeros(3), np.full(3, 5.0), np.zeros(3))
        np.testing.assert_allclose(tau, -expected)

    def test_set_warm_start_leaves_lstm_cold(self, tiny_net):
        # ActuatorSet.warm_start seeds MLP histories only: the LSTM fixed point
        # is not the state Isaac's settle phase leaves behind, so LSTM groups
        # must behave exactly like a cold reset afterwards.
        q, qd, q_des = np.zeros(3), np.zeros(3), np.full(3, 0.3)
        cold_first = _lstm_aset(tiny_net).compute_torques(q, qd, q_des)
        aset = _lstm_aset(tiny_net)
        aset.warm_start(q, qd, q_des)
        np.testing.assert_array_equal(aset.compute_torques(q, qd, q_des), cold_first)

    def test_mixed_with_pd_group_leaves_pd_joints_untouched(self, tiny_net):
        pd = ActuatorGroupIR(
            name="pd", joint_names_expr=["p1", "p2"], model="ideal_pd", stiffness=10.0, damping=1.0, effort_limit=50.0
        )
        lstm = _lstm_group_ir(["n1", "n2"], str(tiny_net))
        order = ["p1", "n1", "p2", "n2"]
        aset = ActuatorSet.from_ir([pd, lstm], order)
        aset.reset(strict=True)
        q = np.zeros(4)
        qd = np.array([1.0, 0.0, -2.0, 0.0])
        q_des = np.array([0.5, 0.1, 8.0, -0.1])
        tau = aset.compute_torques(q, qd, q_des)
        # PD joints: kp*(q_des - q) - kd*qd, boxed at 50.
        assert tau[0] == pytest.approx(10.0 * 0.5 - 1.0 * 1.0)
        assert tau[2] == pytest.approx(50.0)
        assert abs(tau[1]) <= min(SAT, EFFORT_LIM) and abs(tau[3]) <= min(SAT, EFFORT_LIM)

    def test_bundle_path_resolution(self, tiny_net, tmp_path):
        rel = "assets/actuator_nets/sea_net.pt"
        target = tmp_path / rel
        target.parent.mkdir(parents=True)
        target.write_bytes(Path(tiny_net).read_bytes())
        group = _lstm_group_ir(["a"], ANYDRIVE_URL, network_bundle_path=rel)
        aset = ActuatorSet.from_ir([group], ["a"], net_dir=tmp_path)
        assert aset.has_net
        aset.load_networks()
        tau = aset.compute_torques(np.zeros(1), np.zeros(1), np.full(1, 0.2))
        assert np.isfinite(tau).all()

    def test_missing_network_file_raises_on_load(self):
        group = _lstm_group_ir(["a"], "/nonexistent/net.pt")
        aset = ActuatorSet.from_ir([group], ["a"])
        with pytest.raises(ValueError, match="actuator-net file not found"):
            aset.load_networks()

    def test_group_without_network_raises_at_build(self):
        group = _lstm_group_ir(["a"], ANYDRIVE_URL)
        group.network_file = None
        with pytest.raises(ValueError, match="requires a network_file"):
            ActuatorSet.from_ir([group], ["a"])


@pytest.mark.skipif(not ANYDRIVE_NET.is_file(), reason=f"cached ANYdrive net missing: {ANYDRIVE_NET}")
class TestRealAnydriveNet:
    JOINTS = [f"j{i}" for i in range(12)]

    def test_zero_input_output_shape_and_bounds(self):
        aset = _lstm_aset(ANYDRIVE_NET, joints=self.JOINTS)
        tau = aset.compute_torques(np.zeros(12), np.zeros(12), np.zeros(12))
        assert tau.shape == (12,)
        assert np.isfinite(tau).all()
        assert np.all(np.abs(tau) <= EFFORT_LIM)
        # All joints share the same input, so the torques must be identical.
        np.testing.assert_allclose(tau, tau[0])
