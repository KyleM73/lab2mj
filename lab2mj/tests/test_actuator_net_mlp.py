"""Tests for the actuator_net_mlp actuator model (offline).

The synthetic tests script tiny TorchScript MLPs with IsaacLab's
``ActuatorNetMLP`` I/O contract: input ``(J, 2 * len(input_idx))`` of scaled
pos-error/velocity history columns (``input_idx`` selects which past physics
steps, 0 = current), output one torque per joint, scaled by ``torque_scale``
and clipped by the DC-motor curve at the **current** joint velocity (the MLP,
unlike the LSTM, writes the ``DCMotor._joint_vel`` clip buffer). The real Go1
net test downloads the published TorchScript file into the
``data/actuator_nets/`` cache and skips when the download fails.
"""

from pathlib import Path

import numpy as np
import pytest

from lab2mj.actuators import ActuatorSet
from lab2mj.convert import _resolve_actuator_nets, resolve_network_file
from lab2mj.env_yaml import _parse_actuators
from lab2mj.ir import ActuatorGroupIR

GO1_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
    "/Isaac/IsaacLab/ActuatorNets/Unitree/unitree_go1.pt"
)

# Go1 parameters (GO1_ACTUATOR_CFG in isaaclab_assets).
SAT, EFFORT_LIM, VEL_LIM = 23.7, 23.7, 30.0
POS_SCALE, VEL_SCALE, TORQUE_SCALE = -1.0, 1.0, 1.0
INPUT_ORDER = "pos_vel"
INPUT_IDX = [0, 1, 2]
IN_DIM = 2 * len(INPUT_IDX)


def _mlp_dump_dict(network_file: str = GO1_URL) -> dict:
    """Actuator entry shaped like a dump_yaml'd ActuatorNetMLPCfg (Go1 base_legs)."""
    return {
        "base_legs": {
            "class_type": "isaaclab.actuators.actuator_net:ActuatorNetMLP",
            "joint_names_expr": [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
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
            "pos_scale": POS_SCALE,
            "vel_scale": VEL_SCALE,
            "torque_scale": TORQUE_SCALE,
            "input_order": INPUT_ORDER,
            "input_idx": [0, 1, 2],
        }
    }


def _mlp_group_ir(
    joints: list[str],
    network_file: str,
    sat: float = SAT,
    effort_limit: float = EFFORT_LIM,
    velocity_limit: float = VEL_LIM,
    pos_scale: float = POS_SCALE,
    vel_scale: float = VEL_SCALE,
    torque_scale: float = TORQUE_SCALE,
    input_order: str = INPUT_ORDER,
    input_idx: list[int] | None = None,
    network_bundle_path: str | None = None,
) -> ActuatorGroupIR:
    return ActuatorGroupIR(
        name="base_legs",
        joint_names_expr=list(joints),
        model="actuator_net_mlp",
        effort_limit=effort_limit,
        velocity_limit=velocity_limit,
        saturation_effort=sat,
        network_file=network_file,
        network_bundle_path=network_bundle_path,
        pos_scale=pos_scale,
        vel_scale=vel_scale,
        torque_scale=torque_scale,
        input_order=input_order,
        input_idx=list(INPUT_IDX) if input_idx is None else input_idx,
    )


class TestEnvYamlParsing:
    def test_mlp_dump_dict(self):
        (group,) = _parse_actuators(_mlp_dump_dict())
        assert group.model == "actuator_net_mlp"
        assert group.joint_names_expr == [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]
        assert group.network_file == GO1_URL
        assert group.saturation_effort == SAT
        assert group.effort_limit == EFFORT_LIM
        assert group.velocity_limit == VEL_LIM
        assert group.stiffness is None and group.damping is None
        assert group.pos_scale == POS_SCALE
        assert group.vel_scale == VEL_SCALE
        assert group.torque_scale == TORQUE_SCALE
        assert group.input_order == INPUT_ORDER
        assert group.input_idx == INPUT_IDX
        assert group.network_bundle_path is None

    def test_ir_round_trip_keeps_mlp_fields(self):
        group = _mlp_group_ir(["a"], GO1_URL, network_bundle_path="assets/actuator_nets/net.pt")
        rebuilt = ActuatorGroupIR.from_dict(group.to_dict())
        assert rebuilt.model == "actuator_net_mlp"
        assert rebuilt.network_file == GO1_URL
        assert rebuilt.network_bundle_path == "assets/actuator_nets/net.pt"
        assert rebuilt.pos_scale == POS_SCALE
        assert rebuilt.vel_scale == VEL_SCALE
        assert rebuilt.torque_scale == TORQUE_SCALE
        assert rebuilt.input_order == INPUT_ORDER
        assert rebuilt.input_idx == INPUT_IDX


class TestConvertNetResolution:
    def test_mlp_group_weights_are_bundled(self, tmp_path):
        from lab2mj.ir import ActionIR, EnvIR, RobotInitIR, TimingIR

        net = tmp_path / "unitree_go1.pt"
        net.write_bytes(b"jit")
        group = _mlp_group_ir([".*"], str(net))
        ir = EnvIR(
            timing=TimingIR(physics_dt=0.005, decimation=4, episode_length_s=20.0),
            robot_init=RobotInitIR(),
            action=ActionIR(func="JointPositionAction"),
            actuators=[group],
        )
        files = _resolve_actuator_nets(ir)
        assert group.network_bundle_path == "assets/actuator_nets/unitree_go1.pt"
        assert files == {"assets/actuator_nets/unitree_go1.pt": net}


torch = pytest.importorskip("torch")


class _MlpNet(torch.nn.Module):
    """Linear stand-in with the ActuatorNetMLP I/O contract ((B, in_dim) -> (B, 1))."""

    def __init__(self, weight: list[float] | None = None, out_bias: float = 0.0, in_dim: int = IN_DIM):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, 1)
        with torch.no_grad():
            if weight is not None:
                self.fc.weight.copy_(torch.tensor([weight], dtype=torch.float32))
            self.fc.bias.fill_(out_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _write_net(path: Path, **kwargs) -> Path:
    torch.manual_seed(0)
    torch.jit.save(torch.jit.script(_MlpNet(**kwargs)), str(path))
    return path


def _selector_weight(col: int, in_dim: int = IN_DIM) -> list[float]:
    """Weight row that returns input column ``col`` unchanged."""
    weight = [0.0] * in_dim
    weight[col] = 1.0
    return weight


@pytest.fixture(scope="module")
def random_net(tmp_path_factory) -> Path:
    return _write_net(tmp_path_factory.mktemp("nets") / "mlp_net.pt")


# Wide-open curve so unclipped tests observe the raw network output.
NO_CLIP = {"sat": 1e6, "effort_limit": 1e6, "velocity_limit": 1e6}


def _mlp_aset(net_path: Path | str, joints: list[str] | None = None, **kwargs) -> ActuatorSet:
    joints = joints or ["a", "b", "c"]
    aset = ActuatorSet.from_ir([_mlp_group_ir(joints, str(net_path), **kwargs)], joints)
    aset.reset(strict=True)
    return aset


def _dc_clip(tau: np.ndarray, vel: np.ndarray, sat: float, effort_limit: float, velocity_limit: float) -> np.ndarray:
    vel = np.clip(vel, -velocity_limit * (1.0 + effort_limit / sat), velocity_limit * (1.0 + effort_limit / sat))
    max_e = np.minimum(sat * (1.0 - vel / velocity_limit), effort_limit)
    min_e = np.maximum(sat * (-1.0 - vel / velocity_limit), -effort_limit)
    return np.clip(tau, min_e, max_e)


def _reference_rollout(
    net_path: Path,
    steps: list[tuple[np.ndarray, np.ndarray]],
    input_idx: list[int] = INPUT_IDX,
    input_order: str = INPUT_ORDER,
    pos_scale: float = POS_SCALE,
    vel_scale: float = VEL_SCALE,
    torque_scale: float = TORQUE_SCALE,
    sat: float = SAT,
    effort_limit: float = EFFORT_LIM,
    velocity_limit: float = VEL_LIM,
) -> list[np.ndarray]:
    """Manual mirror of ActuatorNetMLP.compute: rolled histories + current-velocity DC clip."""
    net = torch.jit.load(str(net_path), map_location="cpu").eval()
    num_joints = steps[0][0].shape[0]
    history_length = max(input_idx) + 1
    pos_hist = torch.zeros(1, history_length, num_joints)
    vel_hist = torch.zeros(1, history_length, num_joints)
    outs = []
    for pos_error, vel in steps:
        pos_hist = pos_hist.roll(1, 1)
        pos_hist[:, 0] = torch.from_numpy(pos_error.astype(np.float32))
        vel_hist = vel_hist.roll(1, 1)
        vel_hist[:, 0] = torch.from_numpy(vel.astype(np.float32))
        pos_input = torch.cat([pos_hist[:, i].unsqueeze(2) for i in input_idx], dim=2).view(num_joints, -1)
        vel_input = torch.cat([vel_hist[:, i].unsqueeze(2) for i in input_idx], dim=2).view(num_joints, -1)
        if input_order == "pos_vel":
            x = torch.cat([pos_input * pos_scale, vel_input * vel_scale], dim=1)
        else:
            x = torch.cat([vel_input * vel_scale, pos_input * pos_scale], dim=1)
        with torch.inference_mode():
            tau = net(x).view(num_joints) * torque_scale
        outs.append(_dc_clip(tau.cpu().numpy().astype(np.float64), vel, sat, effort_limit, velocity_limit))
    return outs


class TestMLPGroup:
    def test_matches_reference_rollout(self, random_net):
        aset = _mlp_aset(random_net)
        rng = np.random.default_rng(7)
        steps = [(rng.uniform(-0.5, 0.5, 3), rng.uniform(-3.0, 3.0, 3)) for _ in range(6)]
        expected = _reference_rollout(random_net, steps)
        q = np.zeros(3)
        for (pos_error, vel), ref in zip(steps, expected):
            tau = aset.compute_torques(q, vel, q + pos_error)
            np.testing.assert_allclose(tau, ref, rtol=0, atol=1e-7)
        # The history must actually fill: identical inputs at steps 0/1 would be a
        # coincidence, distinct inputs through the history rows change the output.
        assert not np.allclose(expected[0], expected[1])

    @pytest.mark.parametrize(
        ("col", "expected_of_t"),
        [
            # Input columns are [pos@0, pos@1, pos@2, vel@0, vel@1, vel@2]; pos columns
            # carry pos_scale = -1. At step t (1-based) pos_error = t, vel = 100 t;
            # history rows older than the first step are zero.
            (0, lambda t: -t),
            (1, lambda t: -(t - 1) if t >= 2 else 0.0),
            (2, lambda t: -(t - 2) if t >= 3 else 0.0),
            (3, lambda t: 100.0 * t),
            (4, lambda t: 100.0 * (t - 1) if t >= 2 else 0.0),
            (5, lambda t: 100.0 * (t - 2) if t >= 3 else 0.0),
        ],
    )
    def test_history_indexing_hand_verified(self, tmp_path, col, expected_of_t):
        net_path = _write_net(tmp_path / f"sel{col}.pt", weight=_selector_weight(col))
        base = np.array([1.0, 2.0])
        aset = _mlp_aset(net_path, joints=["a", "b"], **NO_CLIP)
        for t in range(1, 5):
            tau = aset.compute_torques(np.zeros(2), 100.0 * t * base, t * base)
            np.testing.assert_allclose(tau, expected_of_t(t) * base, rtol=0, atol=1e-4)

    def test_input_idx_selects_specific_past_steps(self, tmp_path):
        # input_idx = [0, 2]: only the current and two-steps-ago rows feed the net;
        # columns are [pos@0, pos@2, vel@0, vel@2]. Selector on column 1 (pos@2).
        net_path = _write_net(tmp_path / "sel_gap.pt", weight=_selector_weight(1, in_dim=4), in_dim=4)
        aset = _mlp_aset(net_path, joints=["a"], input_idx=[0, 2], **NO_CLIP)
        for t in range(1, 6):
            tau = aset.compute_torques(np.zeros(1), np.zeros(1), np.array([float(t)]))
            expected = -(t - 2) if t >= 3 else 0.0  # pos_scale = -1
            np.testing.assert_allclose(tau, [expected], rtol=0, atol=1e-6)

    @pytest.mark.parametrize("strict", [True, False])
    def test_reset_zeroes_history_without_seeding(self, tmp_path, strict):
        # Selector on pos@1: right after a reset the one-step-ago row must read 0
        # even though the current pos_error is nonzero (Isaac zero-fills, it never
        # seeds the history with the current state).
        net_path = _write_net(tmp_path / "sel1.pt", weight=_selector_weight(1))
        aset = _mlp_aset(net_path, joints=["a"], **NO_CLIP)
        for _ in range(4):
            aset.compute_torques(np.zeros(1), np.zeros(1), np.array([0.9]))
        if strict:
            aset.reset(strict=True)
        else:
            aset.reset(np.random.default_rng(0))
        tau = aset.compute_torques(np.zeros(1), np.zeros(1), np.array([0.9]))
        np.testing.assert_allclose(tau, [0.0], rtol=0, atol=1e-7)

    @pytest.mark.parametrize(
        ("col", "expected"),
        [
            # Columns are [pos@0, pos@1, pos@2, vel@0, vel@1, vel@2] with pos_scale = -1.
            # Warm start with pos_error 0.5 / vel 2.0, then compute one step at
            # pos_error 0 / vel 0: rows 1 and 2 must still read the warm-start input.
            (0, 0.0),
            (1, -0.5),
            (2, -0.5),
            (3, 0.0),
            (4, 2.0),
            (5, 2.0),
        ],
    )
    def test_warm_start_history_hand_computed(self, tmp_path, col, expected):
        net_path = _write_net(tmp_path / f"ws_sel{col}.pt", weight=_selector_weight(col))
        aset = _mlp_aset(net_path, joints=["a"], **NO_CLIP)
        aset.warm_start(np.array([0.1]), np.array([2.0]), np.array([0.6]))
        tau = aset.compute_torques(np.array([0.1]), np.zeros(1), np.array([0.1]))
        np.testing.assert_allclose(tau, [expected], rtol=0, atol=1e-6)

    @pytest.mark.parametrize(("pos_scale", "sign"), [(-1.0, -1.0), (1.0, 1.0), (2.0, 2.0)])
    def test_pos_scale_applied_to_position_errors(self, tmp_path, pos_scale, sign):
        net_path = _write_net(tmp_path / "sel0.pt", weight=_selector_weight(0))
        aset = _mlp_aset(net_path, joints=["a"], pos_scale=pos_scale, **NO_CLIP)
        tau = aset.compute_torques(np.array([0.1]), np.zeros(1), np.array([0.6]))
        np.testing.assert_allclose(tau, [sign * 0.5], rtol=1e-6, atol=0)

    def test_vel_scale_applied_to_velocities(self, tmp_path):
        net_path = _write_net(tmp_path / "sel3.pt", weight=_selector_weight(3))
        aset = _mlp_aset(net_path, joints=["a"], vel_scale=0.5, **NO_CLIP)
        tau = aset.compute_torques(np.zeros(1), np.array([4.0]), np.zeros(1))
        np.testing.assert_allclose(tau, [2.0], rtol=1e-6, atol=0)

    def test_torque_scale_applied_to_output(self, tmp_path):
        net_path = _write_net(tmp_path / "sel0_ts.pt", weight=_selector_weight(0))
        aset = _mlp_aset(net_path, joints=["a"], torque_scale=2.5, **NO_CLIP)
        tau = aset.compute_torques(np.zeros(1), np.zeros(1), np.array([1.0]))
        # pos_scale = -1 then torque_scale = 2.5.
        np.testing.assert_allclose(tau, [-2.5], rtol=1e-6, atol=0)

    def test_input_order_vel_pos_swaps_blocks(self, tmp_path):
        # Under vel_pos, column 0 is the first velocity column, not pos-error.
        net_path = _write_net(tmp_path / "sel0_vp.pt", weight=_selector_weight(0))
        aset = _mlp_aset(net_path, joints=["a"], input_order="vel_pos", **NO_CLIP)
        tau = aset.compute_torques(np.zeros(1), np.array([3.0]), np.array([9.0]))
        np.testing.assert_allclose(tau, [3.0], rtol=1e-6, atol=0)

    def test_clip_uses_current_velocity(self, tmp_path):
        # Saturating output; sat=120, effort_limit=80, velocity_limit=7.5 (corner at
        # 12.5 rad/s). Unlike the LSTM's zero-velocity box, the MLP clip follows the
        # torque-speed curve at the actual joint velocity.
        net_path = _write_net(tmp_path / "big.pt", out_bias=1e6)
        aset = _mlp_aset(net_path, joints=["a"], sat=120.0, effort_limit=80.0, velocity_limit=7.5)
        for qd_value, expected in [(0.0, 80.0), (5.0, 40.0), (-5.0, 80.0), (15.0, -80.0)]:
            aset.reset(strict=True)
            tau = aset.compute_torques(np.zeros(1), np.array([qd_value]), np.zeros(1))
            np.testing.assert_allclose(tau, [expected], rtol=1e-6, atol=0)

    def test_mixed_with_pd_group_leaves_pd_joints_untouched(self, random_net):
        pd = ActuatorGroupIR(
            name="pd", joint_names_expr=["p1", "p2"], model="ideal_pd", stiffness=10.0, damping=1.0, effort_limit=50.0
        )
        mlp = _mlp_group_ir(["n1", "n2"], str(random_net))
        order = ["p1", "n1", "p2", "n2"]
        aset = ActuatorSet.from_ir([pd, mlp], order)
        aset.reset(strict=True)
        q = np.zeros(4)
        qd = np.array([1.0, 0.0, -2.0, 0.0])
        q_des = np.array([0.5, 0.1, 8.0, -0.1])
        tau = aset.compute_torques(q, qd, q_des)
        assert tau[0] == pytest.approx(10.0 * 0.5 - 1.0 * 1.0)
        assert tau[2] == pytest.approx(50.0)
        assert abs(tau[1]) <= min(SAT, EFFORT_LIM) and abs(tau[3]) <= min(SAT, EFFORT_LIM)

    # The missing-network-file (load) and missing-network_file (build) gates fire in
    # shared ActuatorSet code before any per-model branch; test_actuator_net.py covers
    # them once for all net models.

    @pytest.mark.parametrize("missing", ["pos_scale", "vel_scale", "torque_scale", "input_order", "input_idx"])
    def test_missing_contract_field_raises(self, missing):
        group = _mlp_group_ir(["a"], GO1_URL)
        setattr(group, missing, None)
        with pytest.raises(ValueError, match=f"requires {missing}"):
            ActuatorSet.from_ir([group], ["a"])

    def test_invalid_input_order_raises(self):
        group = _mlp_group_ir(["a"], GO1_URL, input_order="pos_then_vel")
        with pytest.raises(ValueError, match="invalid input order"):
            ActuatorSet.from_ir([group], ["a"])

    @pytest.mark.parametrize("input_idx", [[], [0, -1]])
    def test_invalid_input_idx_raises(self, input_idx):
        group = _mlp_group_ir(["a"], GO1_URL, input_idx=input_idx)
        with pytest.raises(ValueError, match="input_idx"):
            ActuatorSet.from_ir([group], ["a"])

    # The torque-curve parameter loop is likewise shared; test_actuator_net.py's
    # DC-motor class covers the missing-parameter raises.


@pytest.fixture(scope="module")
def go1_net() -> Path:
    try:
        return resolve_network_file(GO1_URL)
    except Exception as err:  # noqa: BLE001 - any network/cache failure skips
        pytest.skip(f"unitree_go1.pt unavailable: {err}")


class TestRealGo1Net:
    JOINTS = [f"j{i}" for i in range(12)]

    def test_warm_start_matches_settled_cold_rollout(self, go1_net):
        # A warm-started net must produce exactly the output a cold net reaches
        # once its history has filled with the same constant input.
        q, qd, q_des = np.zeros(12), np.zeros(12), np.full(12, 0.4)
        cold = _mlp_aset(go1_net, joints=self.JOINTS)
        settled = [cold.compute_torques(q, qd, q_des) for _ in range(4)][-1]
        warm = _mlp_aset(go1_net, joints=self.JOINTS)
        warm.warm_start(q, qd, q_des)
        np.testing.assert_array_equal(warm.compute_torques(q, qd, q_des), settled)

    def test_matches_reference_rollout(self, go1_net):
        aset = _mlp_aset(go1_net, joints=self.JOINTS)
        rng = np.random.default_rng(21)
        steps = [(rng.uniform(-0.5, 0.5, 12), rng.uniform(-3.0, 3.0, 12)) for _ in range(5)]
        expected = _reference_rollout(go1_net, steps)
        q = np.zeros(12)
        for (pos_error, vel), ref in zip(steps, expected):
            tau = aset.compute_torques(q, vel, q + pos_error)
            np.testing.assert_allclose(tau, ref, rtol=0, atol=1e-6)
