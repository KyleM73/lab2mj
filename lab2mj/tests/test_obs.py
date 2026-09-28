"""Tests for the MuJoCo-runtime observation pipeline (`lab2mj.obs`)."""

from typing import Any

import numpy as np
import pytest

from lab2mj.ir import NoiseIR, ObsGroupIR, ObsTermIR
from lab2mj.obs import ObsContext, ObsPipeline

from .shared import parse_fixture

BASE_LIN_VEL = "isaaclab.envs.mdp.observations:base_lin_vel"
BASE_ANG_VEL = "isaaclab.envs.mdp.observations:base_ang_vel"
PROJECTED_GRAVITY = "isaaclab.envs.mdp.observations:projected_gravity"
JOINT_POS_REL = "isaaclab.envs.mdp.observations:joint_pos_rel"
JOINT_VEL_REL = "isaaclab.envs.mdp.observations:joint_vel_rel"
LAST_ACTION = "isaaclab.envs.mdp.observations:last_action"
GENERATED_COMMANDS = "isaaclab.envs.mdp.observations:generated_commands"
COMMAND_TIME_REMAINING = "contact_lab.tasks.base_wrench.mdp.observations:command_time_remaining"
HEIGHT_SCAN = "isaaclab.envs.mdp.observations:height_scan"
EXTERNAL_FORCE_B = "contact_lab.tasks.base_wrench.mdp.observations:external_force_b"


def make_ctx(num_joints: int, action_dim: int, seed: int = 0, **overrides) -> ObsContext:
    rng = np.random.default_rng(seed)
    fields: dict[str, Any] = dict(
        root_lin_vel_b=rng.normal(size=3),
        root_ang_vel_b=rng.normal(size=3),
        projected_gravity_b=np.array([0.05, -0.02, -0.99]),
        joint_pos_isaac=rng.normal(size=num_joints),
        joint_vel_isaac=rng.normal(size=num_joints),
        default_joint_pos_isaac=rng.normal(size=num_joints),
        default_joint_vel_isaac=np.zeros(num_joints),
        last_action_raw=rng.normal(size=action_dim),
        commands={"base_velocity": np.array([0.5, -0.2, 0.1])},
        command_time_left={},
        extras={},
    )
    fields.update(overrides)
    return ObsContext(**fields)


def single_term_group(term: ObsTermIR, *, enable_corruption: bool = True, **group_kwargs) -> ObsGroupIR:
    return ObsGroupIR(name="policy", terms=[term], enable_corruption=enable_corruption, **group_kwargs)


def base_lin_vel_term(**kwargs) -> ObsTermIR:
    return ObsTermIR(name="base_lin_vel", func=BASE_LIN_VEL, **kwargs)


"""
Fixture-driven layout tests.
"""


def test_spot_fixture_policy_layout():
    ir = parse_fixture("spot_velocity_env.yaml")
    group = ir.obs_group("policy")
    # Spot velocity uses a group-level history_length=1 override.
    assert group.history_length == 1
    pipeline = ObsPipeline(group, num_joints=12, action_dim=12, command_dims={"base_velocity": 3})

    assert pipeline.obs_dim == 48
    layout = pipeline.layout()
    names = [name for name, _, _ in layout]
    dims = [dim for _, _, dim in layout]
    assert names == [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert dims == [3, 3, 3, 3, 12, 12, 12]
    # Slices are contiguous and cover the full vector.
    offset = 0
    for _, slc, dim in layout:
        assert slc == slice(offset, offset + dim)
        offset += dim
    assert offset == 48


@pytest.mark.parametrize(
    "fixture,num_joints,obs_dim,history_length",
    [("spot_velocity_env.yaml", 12, 48, 1), ("g1_flat_env.yaml", 37, 123, None)],
)
def test_fixture_policy_values_strict(fixture, num_joints, obs_dim, history_length):
    # Both velocity fixtures share the same 7-term block; Spot adds a group-level
    # history_length=1 override, G1 has none.
    ir = parse_fixture(fixture)
    group = ir.obs_group("policy")
    assert group.history_length == history_length
    pipeline = ObsPipeline(
        group, num_joints=num_joints, action_dim=num_joints, command_dims={"base_velocity": 3}, enable_noise=False
    )
    assert pipeline.obs_dim == obs_dim
    assert [dim for _, _, dim in pipeline.layout()] == [3, 3, 3, 3, num_joints, num_joints, num_joints]
    ctx = make_ctx(num_joints, num_joints, seed=3)
    obs = pipeline.compute(ctx)
    assert obs.dtype == np.float32
    expected = np.concatenate(
        [
            ctx.root_lin_vel_b,
            ctx.root_ang_vel_b,
            ctx.projected_gravity_b,
            ctx.commands["base_velocity"],
            ctx.joint_pos_isaac - ctx.default_joint_pos_isaac,
            ctx.joint_vel_isaac - ctx.default_joint_vel_isaac,
            ctx.last_action_raw,
        ]
    ).astype(np.float32)
    assert obs.shape == (obs_dim,)
    np.testing.assert_array_equal(obs, expected)


def test_spot_fixture_critic_has_no_noise():
    ir = parse_fixture("spot_velocity_env.yaml")
    group = ir.obs_group("critic")
    assert not group.enable_corruption
    pipeline = ObsPipeline(group, num_joints=12, action_dim=12, command_dims={"base_velocity": 3})
    ctx = make_ctx(12, 12, seed=4)
    # Even with an rng and noise force-enabled, the critic group applies no
    # noise (IsaacLab drops term noise when enable_corruption is False).
    obs_a = pipeline.compute(ctx, np.random.default_rng(0), enable_noise=True)
    pipeline.reset()
    obs_b = pipeline.compute(ctx, enable_noise=False)
    np.testing.assert_array_equal(obs_a, obs_b)


"""
Pipeline-order tests (noise -> clip -> scale).
"""


def test_order_noise_then_clip_then_scale():
    # value 1.0: +10 (noise) -> 11, clip to [-2, 2] -> 2, scale 0.5 -> 1.0.
    # scale-before-clip would give 2.0; clip-before-noise would give 5.5.
    term = base_lin_vel_term(
        noise=NoiseIR(kind="constant", operation="add", params={"bias": 10.0}),
        clip=(-2.0, 2.0),
        scale=0.5,
    )
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)
    ctx = make_ctx(1, 1, root_lin_vel_b=np.ones(3))
    obs = pipeline.compute(ctx, np.random.default_rng(0))
    np.testing.assert_allclose(obs, np.full(3, 1.0, dtype=np.float32))


def test_scale_scalar_and_per_element():
    term_scalar = base_lin_vel_term(scale=2.0)
    pipeline = ObsPipeline(single_term_group(term_scalar), num_joints=1, action_dim=1)
    ctx = make_ctx(1, 1, root_lin_vel_b=np.array([1.0, -2.0, 3.0]))
    np.testing.assert_allclose(pipeline.compute(ctx, enable_noise=False), [2.0, -4.0, 6.0])

    term_vec = base_lin_vel_term(scale=[1.0, 2.0, 4.0])
    pipeline = ObsPipeline(single_term_group(term_vec), num_joints=1, action_dim=1)
    np.testing.assert_allclose(pipeline.compute(ctx, enable_noise=False), [1.0, -4.0, 12.0])

    with pytest.raises(ValueError, match="per-element scale"):
        ObsPipeline(single_term_group(base_lin_vel_term(scale=[1.0, 2.0])), num_joints=1, action_dim=1)


"""
Noise tests.
"""


def test_gaussian_and_scale_and_abs_operations():
    value = np.array([1.0, 2.0, 3.0])
    ctx = make_ctx(1, 1, root_lin_vel_b=value)

    term = base_lin_vel_term(noise=NoiseIR(kind="gaussian", operation="add", params={"mean": 0.5, "std": 0.2}))
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)
    obs = pipeline.compute(ctx, np.random.default_rng(7))
    expected = value + 0.5 + 0.2 * np.random.default_rng(7).standard_normal(3)
    np.testing.assert_array_equal(obs, expected.astype(np.float32))

    term = base_lin_vel_term(noise=NoiseIR(kind="uniform", operation="scale", params={"n_min": 0.9, "n_max": 1.1}))
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)
    obs = pipeline.compute(ctx, np.random.default_rng(7))
    expected = value * np.random.default_rng(7).uniform(0.9, 1.1, size=3)
    np.testing.assert_array_equal(obs, expected.astype(np.float32))

    term = base_lin_vel_term(noise=NoiseIR(kind="constant", operation="abs", params={"bias": 4.0}))
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)
    np.testing.assert_array_equal(pipeline.compute(ctx, np.random.default_rng(0)), np.full(3, 4.0, dtype=np.float32))


def test_strict_mode_toggles():
    term = base_lin_vel_term(noise=NoiseIR(kind="uniform", operation="add", params={"n_min": -1.0, "n_max": 1.0}))
    value = np.array([1.0, 2.0, 3.0])
    ctx = make_ctx(1, 1, root_lin_vel_b=value)
    clean = value.astype(np.float32)

    # Construction-time strict mode: no rng needed, reproducible.
    strict = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1, enable_noise=False)
    np.testing.assert_array_equal(strict.compute(ctx), clean)
    np.testing.assert_array_equal(strict.compute(ctx), clean)
    # Call-time override re-enables noise on a strict pipeline.
    assert not np.array_equal(strict.compute(ctx, np.random.default_rng(0), enable_noise=True), clean)

    # Call-time strict mode on a noisy pipeline.
    noisy = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)
    np.testing.assert_array_equal(noisy.compute(ctx, enable_noise=False), clean)
    # Noise enabled but no rng -> error.
    with pytest.raises(ValueError, match="no rng"):
        noisy.compute(ctx)


"""
History tests.
"""


def test_history_fill_roll_and_reset():
    term = base_lin_vel_term(history_length=3)
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1, enable_noise=False)
    assert pipeline.obs_dim == 9

    def ctx_with(v: float) -> ObsContext:
        return make_ctx(1, 1, root_lin_vel_b=np.full(3, v))

    def frames(obs: np.ndarray) -> list[list[float]]:
        return obs.reshape(3, 3).tolist()

    # First push after reset fills all slots with the current value.
    assert frames(pipeline.compute(ctx_with(1.0))) == [[1.0] * 3] * 3
    # Oldest-first ordering as new frames arrive.
    assert frames(pipeline.compute(ctx_with(2.0))) == [[1.0] * 3, [1.0] * 3, [2.0] * 3]
    assert frames(pipeline.compute(ctx_with(3.0))) == [[1.0] * 3, [2.0] * 3, [3.0] * 3]
    assert frames(pipeline.compute(ctx_with(4.0))) == [[2.0] * 3, [3.0] * 3, [4.0] * 3]
    # Reset clears history; the next push fills all slots again.
    pipeline.reset()
    assert frames(pipeline.compute(ctx_with(5.0))) == [[5.0] * 3] * 3


def test_seed_history_from_flat_reproduces_settled_history():
    # A settled reference dump's obs0 carries distinct rolling history frames; after
    # reset + seed, the next compute (evaluating the same t0 state) must reproduce the
    # recorded flat observation exactly instead of tile-filling with the t0 frame.
    term = base_lin_vel_term(history_length=3)
    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1, enable_noise=False)

    def ctx_with(v: float) -> ObsContext:
        return make_ctx(1, 1, root_lin_vel_b=np.full(3, v))

    for v in (1.0, 2.0):
        pipeline.compute(ctx_with(v))
    obs0 = pipeline.compute(ctx_with(3.0))
    pipeline.reset()
    pipeline.seed_history_from_flat(obs0)
    np.testing.assert_array_equal(pipeline.compute(ctx_with(3.0)), obs0)
    # Subsequent frames keep rolling from the seeded history (no tile-fill).
    frames = pipeline.compute(ctx_with(4.0)).reshape(3, 3)
    np.testing.assert_array_equal(frames, np.stack([np.full(3, 2.0), np.full(3, 3.0), np.full(3, 4.0)]))


def test_history_blocks_are_per_term_contiguous():
    group = ObsGroupIR(
        name="policy",
        terms=[
            ObsTermIR(name="base_lin_vel", func=BASE_LIN_VEL, history_length=2),
            ObsTermIR(name="base_ang_vel", func=BASE_ANG_VEL, history_length=2),
        ],
        enable_corruption=False,
    )
    pipeline = ObsPipeline(group, num_joints=1, action_dim=1)
    ctx1 = make_ctx(1, 1, root_lin_vel_b=np.full(3, 1.0), root_ang_vel_b=np.full(3, 10.0))
    ctx2 = make_ctx(1, 1, root_lin_vel_b=np.full(3, 2.0), root_ang_vel_b=np.full(3, 20.0))
    pipeline.compute(ctx1)
    obs = pipeline.compute(ctx2)
    # [lin_old, lin_new, ang_old, ang_new] — per-term contiguous, oldest first.
    expected = np.concatenate([np.full(3, 1.0), np.full(3, 2.0), np.full(3, 10.0), np.full(3, 20.0)])
    np.testing.assert_array_equal(obs, expected.astype(np.float32))
    assert [(name, dim) for name, _, dim in pipeline.layout()] == [("base_lin_vel", 6), ("base_ang_vel", 6)]


def test_group_history_override():
    # Terms declare no history; the group-level override applies H=2 to all.
    group = ObsGroupIR(
        name="policy",
        terms=[
            ObsTermIR(name="base_lin_vel", func=BASE_LIN_VEL, history_length=0),
            ObsTermIR(name="base_ang_vel", func=BASE_ANG_VEL, history_length=5),
        ],
        enable_corruption=False,
        history_length=2,
    )
    pipeline = ObsPipeline(group, num_joints=1, action_dim=1)
    assert pipeline.obs_dim == 12
    assert [dim for _, _, dim in pipeline.layout()] == [6, 6]

    # A group override of history_length=0 disables per-term history too.
    group_zero = ObsGroupIR(
        name="policy",
        terms=[ObsTermIR(name="base_ang_vel", func=BASE_ANG_VEL, history_length=5)],
        enable_corruption=False,
        history_length=0,
    )
    assert ObsPipeline(group_zero, num_joints=1, action_dim=1).obs_dim == 3


def test_group_flatten_history_dim_override():
    # Group override copies flatten_history_dim=False onto terms: outputs are
    # (H, dim) frames concatenated along the last dim -> frame-interleaved.
    group = ObsGroupIR(
        name="policy",
        terms=[
            ObsTermIR(name="base_lin_vel", func=BASE_LIN_VEL, flatten_history_dim=True),
            ObsTermIR(name="base_ang_vel", func=BASE_ANG_VEL, flatten_history_dim=True),
        ],
        enable_corruption=False,
        history_length=2,
        flatten_history_dim=False,
    )
    pipeline = ObsPipeline(group, num_joints=1, action_dim=1)
    assert pipeline.obs_dim == 12
    with pytest.raises(ValueError, match="layout"):
        pipeline.layout()

    ctx1 = make_ctx(1, 1, root_lin_vel_b=np.full(3, 1.0), root_ang_vel_b=np.full(3, 10.0))
    ctx2 = make_ctx(1, 1, root_lin_vel_b=np.full(3, 2.0), root_ang_vel_b=np.full(3, 20.0))
    pipeline.compute(ctx1)
    obs = pipeline.compute(ctx2)
    # Rows are history frames: [lin_old | ang_old], then [lin_new | ang_new].
    expected = np.concatenate([np.full(3, 1.0), np.full(3, 10.0), np.full(3, 2.0), np.full(3, 20.0)])
    np.testing.assert_array_equal(obs, expected.astype(np.float32))


"""
Term-resolution error tests.
"""


def test_extras_terms_require_provider():
    term = ObsTermIR(name="height_scan", func=HEIGHT_SCAN, params={"sensor_cfg": {"name": "height_scanner"}})
    with pytest.raises(ValueError, match="requires a runtime extras provider"):
        ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)

    pipeline = ObsPipeline(single_term_group(term), num_joints=1, action_dim=1, extras_dims={"height_scan": 4})
    assert pipeline.obs_dim == 4
    ctx = make_ctx(1, 1, extras={"height_scan": np.array([0.1, 0.2, 0.3, 0.4])})
    expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    np.testing.assert_array_equal(pipeline.compute(ctx, enable_noise=False), expected)
    # Missing extras entry at compute time is a clear KeyError.
    with pytest.raises(KeyError, match="height_scan"):
        pipeline.compute(make_ctx(1, 1), enable_noise=False)

    term = ObsTermIR(name="ext_force", func=EXTERNAL_FORCE_B)
    with pytest.raises(ValueError, match="ext_force"):
        ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)


def test_command_time_remaining():
    term = ObsTermIR(name="clock", func=COMMAND_TIME_REMAINING, params={"command_name": "pose_command"})
    with pytest.raises(ValueError, match="command_resample_time_s"):
        ObsPipeline(single_term_group(term), num_joints=1, action_dim=1)

    pipeline = ObsPipeline(
        single_term_group(term),
        num_joints=1,
        action_dim=1,
        command_resample_time_s={"pose_command": 10.0},
    )
    assert pipeline.obs_dim == 1
    ctx = make_ctx(1, 1, command_time_left={"pose_command": 4.0})
    np.testing.assert_allclose(pipeline.compute(ctx, enable_noise=False), [0.4])
    # Clipped to [0, 1] at both ends.
    ctx.command_time_left["pose_command"] = 15.0
    np.testing.assert_array_equal(pipeline.compute(ctx, enable_noise=False), np.array([1.0], dtype=np.float32))
    ctx.command_time_left["pose_command"] = -1.0
    np.testing.assert_array_equal(pipeline.compute(ctx, enable_noise=False), np.array([0.0], dtype=np.float32))


def test_concatenate_terms_false_raises():
    group = single_term_group(base_lin_vel_term(), concatenate_terms=False)
    with pytest.raises(ValueError, match="concatenate_terms"):
        ObsPipeline(group, num_joints=1, action_dim=1)


def test_joint_subset_and_named_action_raise():
    term = ObsTermIR(name="joint_pos", func=JOINT_POS_REL, params={"asset_cfg": {"name": "robot", "joint_ids": [0, 1]}})
    with pytest.raises(ValueError, match="joint_ids"):
        ObsPipeline(single_term_group(term), num_joints=4, action_dim=4)

    term = ObsTermIR(name="actions", func=LAST_ACTION, params={"action_name": "joint_pos"})
    with pytest.raises(ValueError, match="named action"):
        ObsPipeline(single_term_group(term), num_joints=4, action_dim=4)


def test_ctx_arrays_not_mutated():
    term = ObsTermIR(name="joint_vel", func=JOINT_VEL_REL, clip=(-0.1, 0.1), scale=100.0)
    pipeline = ObsPipeline(single_term_group(term), num_joints=3, action_dim=3, enable_noise=False)
    ctx = make_ctx(3, 3, joint_vel_isaac=np.array([1.0, -1.0, 0.05]), default_joint_vel_isaac=np.zeros(3))
    before = ctx.joint_vel_isaac.copy()
    np.testing.assert_allclose(pipeline.compute(ctx), [10.0, -10.0, 5.0])
    np.testing.assert_array_equal(ctx.joint_vel_isaac, before)


PHASE = "contact_lab.tasks.base_wrench.g1.mdp.observations:phase"


def test_gait_phase_sin_cos_and_no_gait():
    term = ObsTermIR(name="phase_command", func=PHASE, params={"command_name": "frequency"})
    pipeline = ObsPipeline(
        single_term_group(term), num_joints=1, action_dim=1, command_dims={"frequency": 2}, enable_noise=False
    )
    ctx = make_ctx(1, 1, commands={"frequency": np.array([1.8, 0.25])})
    np.testing.assert_allclose(pipeline.compute(ctx, np.random.default_rng(0)), [1.0, 0.0], atol=1e-6)
    # contact_lab's gait_state reports phase 0 when no gait is commanded.
    ctx = make_ctx(1, 1, commands={"frequency": np.array([0.0, 0.25])})
    np.testing.assert_allclose(pipeline.compute(ctx, np.random.default_rng(0)), [0.0, 1.0], atol=1e-6)


def test_gait_phase_requires_frequency_command_and_command_source():
    term = ObsTermIR(name="phase_command", func=PHASE, params={})
    with pytest.raises(ValueError, match="frequency, phase"):
        ObsPipeline(single_term_group(term), num_joints=1, action_dim=1, command_dims={"base_velocity": 3})
    action_term = ObsTermIR(name="phase_command", func=PHASE, params={"source": "action"})
    with pytest.raises(ValueError, match="action"):
        ObsPipeline(single_term_group(action_term), num_joints=1, action_dim=1, command_dims={"frequency": 2})
