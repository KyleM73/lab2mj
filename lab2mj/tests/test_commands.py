"""Tests for lab2mj.commands (offline, numpy-only)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lab2mj.commands import (
    TerrainBasedPose2dCommand,
    UniformPose2dCommand,
    UniformScalarCommand,
    UniformVelocityCommand,
    build_command,
)
from lab2mj.ir import CommandIR
from lab2mj.quat import quat_from_euler_xyz, quat_mul, wrap_to_pi

from .shared import make_state, parse_fixture

# ---------------------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------------------


def test_wrap_to_pi_matches_isaac_edge_cases():
    assert wrap_to_pi(math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(-math.pi) == pytest.approx(-math.pi)
    assert wrap_to_pi(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(-3 * math.pi) == pytest.approx(-math.pi)
    assert wrap_to_pi(6.0) == pytest.approx(6.0 - 2 * math.pi)
    assert wrap_to_pi(0.3) == pytest.approx(0.3)
    np.testing.assert_allclose(wrap_to_pi(np.array([0.1, 2 * math.pi + 0.1])), [0.1, 0.1], atol=1e-12)


# ---------------------------------------------------------------------------------------
# Resample clock (CommandTerm.compute flow)
# ---------------------------------------------------------------------------------------


def test_resample_clock_and_counter():
    gen = UniformVelocityCommand((10.0, 10.0), lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
    rng = np.random.default_rng(0)
    state = make_state()
    gen.reset(rng, state)
    assert gen.time_left == pytest.approx(10.0)
    assert gen.command_counter == 1

    gen.step(4.0, state, rng)
    assert gen.time_left == pytest.approx(6.0)
    assert gen.command_counter == 1

    gen.step(6.0, state, rng)  # hits exactly zero -> resample (Isaac: time_left <= 0)
    assert gen.time_left == pytest.approx(10.0)
    assert gen.command_counter == 2


def test_resample_draws_within_ranges():
    gen = UniformVelocityCommand((5.0, 8.0), lin_vel_x=(-1.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0))
    rng = np.random.default_rng(42)
    state = make_state()
    for _ in range(20):
        gen.reset(rng, state)
        assert 5.0 <= gen.time_left <= 8.0
        vx, vy, wz = gen.vel_command_b
        assert -1.5 <= vx <= 1.5 and -1.0 <= vy <= 1.0 and -1.0 <= wz <= 1.0


def test_seeded_trajectory_is_reproducible():
    def run():
        gen = UniformVelocityCommand(
            (0.5, 1.5), lin_vel_x=(0.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), rel_standing_envs=0.5
        )
        rng = np.random.default_rng(7)
        state = make_state()
        gen.reset(rng, state)
        traj = []
        for _ in range(50):
            gen.step(0.2, state, rng)
            traj.append(gen.command.copy())
        return np.stack(traj)

    np.testing.assert_array_equal(run(), run())


# ---------------------------------------------------------------------------------------
# UniformVelocityCommand semantics
# ---------------------------------------------------------------------------------------


def test_heading_to_wz_conversion():
    # G1 cfg: stiffness 0.5, ang_vel_z (-1, 1), all envs heading-controlled.
    gen = UniformVelocityCommand(
        (10.0, 10.0),
        lin_vel_x=(0.0, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=1.0,
        rel_standing_envs=0.0,
    )
    rng = np.random.default_rng(3)
    state = make_state(heading=-3.0)
    gen.reset(rng, state)
    gen.heading_target = 3.0
    gen.is_heading_env = True
    gen.is_standing_env = False
    gen.step(0.02, state, rng)
    # wrap_to_pi(3.0 - (-3.0)) = 6 - 2*pi; wz = 0.5 * that, well inside (-1, 1).
    expected = 0.5 * (6.0 - 2 * math.pi)
    assert gen.command[2] == pytest.approx(expected)


def test_heading_wz_is_clipped_to_ang_vel_range():
    gen = UniformVelocityCommand(
        (10.0, 10.0),
        lin_vel_x=(0.0, 1.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0),
        heading=(-math.pi, math.pi),
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_heading_envs=1.0,
    )
    rng = np.random.default_rng(3)
    state = make_state(heading=0.0)
    gen.reset(rng, state)
    gen.heading_target = 2.5  # error 2.5 rad * 1.0 -> clipped to +1
    gen.is_heading_env = True
    gen.is_standing_env = False
    gen.step(0.02, state, rng)
    assert gen.command[2] == pytest.approx(1.0)


def test_non_heading_env_keeps_sampled_wz():
    gen = UniformVelocityCommand(
        (10.0, 10.0),
        lin_vel_x=(0.0, 0.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.7, 0.7),
        heading=(-math.pi, math.pi),
        heading_command=True,
        rel_heading_envs=0.0,  # never heading-controlled
        rel_standing_envs=0.0,
    )
    rng = np.random.default_rng(1)
    state = make_state(heading=1.0)
    gen.reset(rng, state)
    assert gen.is_heading_env is False
    gen.step(0.02, state, rng)
    assert gen.command[2] == pytest.approx(0.7)


def test_standing_env_zeroes_command():
    gen = UniformVelocityCommand(
        (10.0, 10.0), lin_vel_x=(1.0, 1.0), lin_vel_y=(1.0, 1.0), ang_vel_z=(1.0, 1.0), rel_standing_envs=1.0
    )
    rng = np.random.default_rng(0)
    state = make_state()
    gen.reset(rng, state)
    assert gen.is_standing_env is True
    gen.step(0.02, state, rng)
    np.testing.assert_array_equal(gen.command, np.zeros(3))


# ---------------------------------------------------------------------------------------
# Pose 2D commands
# ---------------------------------------------------------------------------------------


def test_pose_command_body_frame_math_with_rotated_root():
    # Root at (1, 2, 0.5), yawed +pi/2 with a roll component that yaw_quat must strip.
    q_yaw = quat_from_euler_xyz(0.0, 0.0, math.pi / 2)
    q_root = quat_mul(q_yaw, quat_from_euler_xyz(math.pi / 6, 0.0, 0.0))
    state = make_state(pos=(1.0, 2.0, 0.5), quat=q_root)
    assert state.heading_w == pytest.approx(math.pi / 2)

    gen = UniformPose2dCommand(
        (10.0, 10.0),
        pos_x=(0.0, 0.0),
        pos_y=(0.0, 0.0),
        heading=(0.0, 0.0),
        strict=True,
        fixed_command=[3.0, 2.0, 0.5, 1.0],  # world goal + heading target
    )
    rng = np.random.default_rng(0)
    gen.reset(rng, state)
    gen.step(0.02, state, rng)
    # target_vec_w = (2, 0, 0); R_z(-pi/2) @ (2, 0, 0) = (0, -2, 0).
    np.testing.assert_allclose(gen.command[:3], [0.0, -2.0, 0.0], atol=1e-12)
    assert gen.command[3] == pytest.approx(wrap_to_pi(1.0 - math.pi / 2))


def test_pose_command_reset_clears_body_frame_and_retarget_revives_it():
    # The body-frame command is written only by _update_command; reset must not
    # serve the previous episode's goal (Isaac's fresh-env buffers are zeros),
    # and retarget() must re-derive it for settled-dump continuation.
    gen = UniformPose2dCommand(
        (10.0, 10.0),
        pos_x=(0.0, 0.0),
        pos_y=(0.0, 0.0),
        heading=(0.0, 0.0),
        strict=True,
        fixed_command=[3.0, 2.0, 0.5, 1.0],
    )
    rng = np.random.default_rng(0)
    state = make_state(pos=(1.0, 2.0, 0.5))
    gen.reset(rng, state)
    gen.step(0.02, state, rng)
    assert np.abs(gen.command).max() > 0.0

    gen.reset(rng, state)
    np.testing.assert_array_equal(gen.command, np.zeros(4))

    gen.retarget(state)
    np.testing.assert_allclose(gen.command[:3], [2.0, 0.0, 0.0], atol=1e-12)
    assert gen.command[3] == pytest.approx(1.0)


def test_pose_command_goal_sampling_is_env_origin_relative():
    gen = UniformPose2dCommand(
        (10.0, 10.0), pos_x=(2.0, 2.0), pos_y=(-1.0, -1.0), heading=(0.25, 0.25), default_root_height=0.42
    )
    rng = np.random.default_rng(0)
    state = make_state(origin=(10.0, 20.0, 0.0))
    gen.reset(rng, state)
    np.testing.assert_allclose(gen.pos_command_w, [12.0, 19.0, 0.42], atol=1e-12)
    assert gen.heading_command_w == pytest.approx(0.25)


def test_pose_command_simple_heading_picks_closest_direction():
    gen = UniformPose2dCommand(
        (10.0, 10.0), pos_x=(2.0, 2.0), pos_y=(0.0, 0.0), heading=(0.0, 0.0), simple_heading=True
    )
    rng = np.random.default_rng(0)
    # Robot faces -x (heading pi); goal straight ahead in +x -> flipped direction (pi) is closer.
    state = make_state(pos=(0.0, 0.0, 0.0), quat=quat_from_euler_xyz(0.0, 0.0, math.pi), heading=math.pi)
    gen.reset(rng, state)
    assert gen.heading_command_w == pytest.approx(math.pi)

    # Robot facing the goal -> direct direction (0) is chosen.
    state = make_state(pos=(0.0, 0.0, 0.0), heading=0.0)
    gen.reset(rng, state)
    assert gen.heading_command_w == pytest.approx(0.0)


def test_terrain_based_pose_command_uses_patch_sampler():
    gen = TerrainBasedPose2dCommand(
        (10.0, 10.0),
        patch_sampler=lambda rng: np.array([5.0, 6.0]),
        heading=(0.25, 0.25),
        default_root_height=0.4,
    )
    rng = np.random.default_rng(0)
    gen.reset(rng, make_state())
    np.testing.assert_allclose(gen.pos_command_w, [5.0, 6.0, 0.4], atol=1e-12)
    assert gen.command.shape == (4,)


# ---------------------------------------------------------------------------------------
# Scalar commands + strict mode
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
def test_fixed_command_pins_in_and_out_of_strict(strict):
    # A fixed command must pin outside strict mode too (play.py relies on this).
    gen = UniformVelocityCommand(
        (10.0, 10.0),
        lin_vel_x=(-1, 1),
        lin_vel_y=(-1, 1),
        ang_vel_z=(-1, 1),
        rel_standing_envs=1.0,  # would zero the command if not pinned
        strict=strict,
        fixed_command=[0.5, 0.0, 0.1],
    )
    rng = np.random.default_rng(0)
    state = make_state()
    gen.reset(rng, state)
    for _ in range(5):
        gen.step(100.0, state, rng)  # far past any resample clock
    np.testing.assert_array_equal(gen.command, [0.5, 0.0, 0.1])
    assert gen.command_counter == 0
    if strict:
        assert gen.time_left == math.inf


def test_strict_mode_requires_fixed_command():
    with pytest.raises(ValueError, match="fixed_command"):
        UniformVelocityCommand((10.0, 10.0), lin_vel_x=(0, 1), lin_vel_y=(0, 1), ang_vel_z=(0, 1), strict=True)


# ---------------------------------------------------------------------------------------
# Factory + fixture coverage
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["spot_velocity_env.yaml", "g1_flat_env.yaml"])
def test_every_fixture_command_builds_and_runs(fixture):
    ir = parse_fixture(fixture)
    assert ir.commands, "fixture should declare at least one command"
    rng = np.random.default_rng(0)
    state = make_state()
    for cmd_ir in ir.commands:
        gen = build_command(cmd_ir, default_root_height=float(ir.robot_init.root_pos_env[2]))
        gen.reset(rng, state)
        for _ in range(3):
            gen.step(ir.timing.policy_dt, state, rng)
        assert gen.command.shape == (3,)  # both fixtures use UniformVelocityCommand
        assert np.isfinite(gen.command).all()


def test_factory_scalar_and_unknown_types():
    scalar_ir = CommandIR(
        name="vel_limit",
        type="VelocityLimitCommand",
        params={"resampling_time_range": [4.0, 6.0], "range": [0.3, 1.2]},
    )
    gen = build_command(scalar_ir)
    assert isinstance(gen, UniformScalarCommand)
    rng = np.random.default_rng(0)
    gen.reset(rng, make_state())
    assert 0.3 <= gen.command[0] <= 1.2

    with pytest.raises(ValueError, match="unsupported command type"):
        build_command(CommandIR(name="x", type="MysteryCommand", params={"resampling_time_range": [1, 1]}))


def test_factory_terrain_pose_requires_sampler():
    ir = CommandIR(
        name="goal",
        type="TerrainBasedPose2dCommand",
        params={"resampling_time_range": [8.0, 8.0], "simple_heading": False, "ranges": {"heading": [-1.0, 1.0]}},
    )
    with pytest.raises(ValueError, match="patch_sampler"):
        build_command(ir)
    gen = build_command(ir, patch_sampler=lambda rng: np.zeros(2))
    assert isinstance(gen, TerrainBasedPose2dCommand)


class TestRegistryAndParamErrors:
    def test_missing_resampling_time_range_names_the_term(self):
        ir = CommandIR(name="base_velocity", type="UniformVelocityCommand", params={})
        with pytest.raises(ValueError, match="base_velocity.*resampling_time_range"):
            build_command(ir)
