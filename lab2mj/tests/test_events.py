"""Tests for lab2mj.events (offline; tiny synthetic MJCF, no Isaac Sim)."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from lab2mj.events import EventSet, RobotMap, read_root_com_vel_w, resolve_names
from lab2mj.ir import EventIR
from lab2mj.quat import quat_from_euler_xyz

from .shared import parse_fixture

TEST_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.5">
      <freejoint name="root"/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.1" mass="4.0"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="j1" axis="0 1 0" range="-1.0 1.0"/>
        <geom name="l1_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1.0"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" axis="0 1 0" range="-1.5 1.5"/>
          <geom name="l2_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def make_model_and_map() -> tuple[mujoco.MjModel, mujoco.MjData, RobotMap]:
    model = mujoco.MjModel.from_xml_string(TEST_XML)
    data = mujoco.MjData(model)
    body = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("base", "link1", "link2")}
    geom = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("base_geom", "l1_geom", "l2_geom")}
    joints = ["j1", "j2"]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joints]
    root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    robot_map = RobotMap(
        body_ids={"base": [body["base"]], "link1": [body["link1"]], "link2": [body["link2"]]},
        geom_ids={"base": [geom["base_geom"]], "link1": [geom["l1_geom"]], "link2": [geom["l2_geom"]]},
        root_body_id=body["base"],
        root_qpos_adr=int(model.jnt_qposadr[root_jid]),
        root_dof_adr=int(model.jnt_dofadr[root_jid]),
        joint_names=joints,
        qpos_adr=np.array([model.jnt_qposadr[j] for j in jids]),
        dof_adr=np.array([model.jnt_dofadr[j] for j in jids]),
        default_joint_pos=np.array([0.3, -0.4]),
        default_joint_vel=np.zeros(2),
        joint_pos_limits=model.jnt_range[jids].copy(),
        joint_vel_limits=np.array([10.0, 10.0]),
        default_root_pose_env=np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]),
        default_root_vel_w=np.zeros(6),
        env_origin_w=np.array([2.0, 0.0, 0.0]),
        default_body_mass=model.body_mass.copy(),
        default_body_inertia=model.body_inertia.copy(),
        default_body_ipos=model.body_ipos.copy(),
    )
    return model, data, robot_map


def event(name, func, mode, params, interval=None) -> EventIR:
    return EventIR(
        name=name, func=f"isaaclab.envs.mdp.events:{func}", mode=mode, params=params, interval_range_s=interval
    )


def test_resolve_names_order_and_none():
    names = ["body", "fl_uleg", "fl_lleg", "fl_foot"]
    assert resolve_names(None, names) == names
    assert resolve_names(".*leg", names) == ["fl_uleg", "fl_lleg"]
    assert resolve_names(["fl_foot", "body"], names) == ["body", "fl_foot"]


def test_resolve_names_overlapping_patterns_raise():
    # Isaac's resolve_matching_names rejects a name matched by more than one pattern
    # (same contract as lab2mj.actuators.resolve_matching_names).
    with pytest.raises(ValueError, match="multiple patterns"):
        resolve_names([".*leg", "fl_.*"], ["body", "fl_uleg", "fl_lleg"])


def test_id_selector_and_foreign_entity_raise():
    # body_ids has no name -> id map on the runtime side and would silently become
    # match-ALL; a non-robot entity would silently apply to the robot instead.
    ev_ids = event(
        "base_mass",
        "randomize_rigid_body_mass",
        "startup",
        {"asset_cfg": {"body_ids": [0]}, "mass_distribution_params": [1.0, 1.0], "operation": "scale"},
    )
    with pytest.raises(ValueError, match="body_ids"):
        EventSet([ev_ids])
    ev_entity = event(
        "object_mass",
        "randomize_rigid_body_mass",
        "startup",
        {
            "asset_cfg": {"name": "object", "body_names": ".*"},
            "mass_distribution_params": [1.0, 1.0],
            "operation": "scale",
        },
    )
    with pytest.raises(ValueError, match="scene entity 'object'"):
        EventSet([ev_entity])


# ---------------------------------------------------------------------------------------
# Startup events
# ---------------------------------------------------------------------------------------


def test_mass_add_scales_inertia_and_uses_defaults():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "base_mass",
        "randomize_rigid_body_mass",
        "startup",
        {
            "asset_cfg": {"body_names": "base"},
            "mass_distribution_params": [-0.5, 0.5],
            "operation": "add",
            "distribution": "uniform",
        },
    )
    events = EventSet([ev])
    bid = robot_map.body_ids["base"][0]
    default_mass = robot_map.default_body_mass[bid]
    default_inertia = robot_map.default_body_inertia[bid].copy()

    events.apply_startup(model, data, robot_map, np.random.default_rng(0))
    mass_1 = float(model.body_mass[bid])
    assert default_mass - 0.5 <= mass_1 <= default_mass + 0.5
    assert mass_1 != pytest.approx(default_mass)
    np.testing.assert_allclose(model.body_inertia[bid], default_inertia * (mass_1 / default_mass), rtol=1e-12)
    # Other bodies untouched.
    other = robot_map.body_ids["link1"][0]
    assert model.body_mass[other] == pytest.approx(robot_map.default_body_mass[other])

    # Re-applying randomizes from the DEFAULT mass, not the previous sample (no compounding).
    events.apply_startup(model, data, robot_map, np.random.default_rng(1))
    assert default_mass - 0.5 <= float(model.body_mass[bid]) <= default_mass + 0.5

    # Model stays steppable after mj_setConst.
    mujoco.mj_step(model, data)


def test_mass_scale_operation():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "scale_mass",
        "randomize_rigid_body_mass",
        "startup",
        {
            "asset_cfg": {"body_names": "link.*"},
            "mass_distribution_params": [2.0, 2.0],
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    EventSet([ev]).apply_startup(model, data, robot_map, np.random.default_rng(0))
    for name in ("link1", "link2"):
        bid = robot_map.body_ids[name][0]
        assert model.body_mass[bid] == pytest.approx(2.0 * robot_map.default_body_mass[bid])


def test_friction_bucket_assignment():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "physics_material",
        "randomize_rigid_body_material",
        "startup",
        {
            "asset_cfg": {"body_names": ".*"},
            "static_friction_range": [0.3, 1.0],
            "dynamic_friction_range": [0.3, 0.8],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 2,
        },
    )
    events = EventSet([ev], terrain_static_friction=0.7, friction_combine_mode="multiply")
    events.apply_startup(model, data, robot_map, np.random.default_rng(0))
    values = {float(model.geom_friction[g, 0]) for gs in robot_map.geom_ids.values() for g in gs}
    # Pair friction = bucket_static * terrain_static, so within [0.3, 1.0] * 0.7.
    for v in values:
        assert 0.3 * 0.7 <= v <= 1.0 * 0.7
    assert len(values) <= 2  # at most num_buckets distinct values
    # Torsional/rolling coefficients untouched.
    g0 = robot_map.geom_ids["base"][0]
    np.testing.assert_array_equal(model.geom_friction[g0, 1:], [0.005, 0.0001])


@pytest.mark.parametrize(
    "combine_mode,expected_pair",
    [
        ("multiply", 0.4 * 0.5),
        # Omitted combine mode falls back to PhysX's default "average": (bucket + terrain) / 2.
        (None, 0.5 * (0.4 + 0.5)),
    ],
)
def test_friction_single_bucket_pair_value(combine_mode, expected_pair):
    model, data, robot_map = make_model_and_map()
    ev = event(
        "physics_material",
        "randomize_rigid_body_material",
        "startup",
        {
            "asset_cfg": {"body_names": "base"},
            "static_friction_range": [0.4, 0.4],
            "dynamic_friction_range": [0.4, 0.4],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 1,
        },
    )
    kwargs = {"friction_combine_mode": combine_mode} if combine_mode is not None else {}
    EventSet([ev], terrain_static_friction=0.5, **kwargs).apply_startup(
        model, data, robot_map, np.random.default_rng(0)
    )
    g0 = robot_map.geom_ids["base"][0]
    assert model.geom_friction[g0, 0] == pytest.approx(expected_pair)
    g1 = robot_map.geom_ids["link1"][0]  # unmatched body keeps MJCF default
    assert model.geom_friction[g1, 0] == pytest.approx(1.0)


def test_friction_buckets_sampled_once_per_eventset():
    """IsaacLab samples the material buckets at term init; re-applying the event only
    redraws bucket *indices*, never the bucket values."""
    model, data, robot_map = make_model_and_map()
    ev = event(
        "physics_material",
        "randomize_rigid_body_material",
        "startup",
        {
            "asset_cfg": {"body_names": "base"},
            "static_friction_range": [0.3, 1.0],
            "dynamic_friction_range": [0.3, 0.8],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 1,
        },
    )
    events = EventSet([ev])
    g0 = robot_map.geom_ids["base"][0]
    events.apply_startup(model, data, robot_map, np.random.default_rng(0))
    first = float(model.geom_friction[g0, 0])
    # A different application rng must reuse the cached bucket (num_buckets=1 pins the index).
    events.apply_startup(model, data, robot_map, np.random.default_rng(123))
    assert float(model.geom_friction[g0, 0]) == first
    # A fresh EventSet resamples the bucket table.
    events_b = EventSet([ev])
    events_b.apply_startup(model, data, robot_map, np.random.default_rng(123))
    assert float(model.geom_friction[g0, 0]) != first


def test_friction_buckets_from_construction_rng():
    model, data, robot_map = make_model_and_map()
    params = {
        "asset_cfg": {"body_names": "base"},
        "static_friction_range": [0.3, 1.0],
        "dynamic_friction_range": [0.3, 0.8],
        "restitution_range": [0.0, 0.0],
        "num_buckets": 1,
    }
    ev = event("physics_material", "randomize_rigid_body_material", "startup", params)
    g0 = robot_map.geom_ids["base"][0]
    expected_bucket = np.random.default_rng(7).uniform([0.3, 0.3, 0.0], [1.0, 0.8, 0.0], (1, 3))
    events = EventSet([ev], friction_combine_mode="multiply", rng=np.random.default_rng(7))
    events.apply_startup(model, data, robot_map, np.random.default_rng(999))
    assert float(model.geom_friction[g0, 0]) == pytest.approx(expected_bucket[0, 0] * 1.0)


def test_com_randomization_shared_offset():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "com",
        "randomize_rigid_body_com",
        "startup",
        {"asset_cfg": {"body_names": ["base", "link1"]}, "com_range": {"x": [0.01, 0.01], "z": [-0.02, -0.02]}},
    )
    EventSet([ev]).apply_startup(model, data, robot_map, np.random.default_rng(0))
    for name in ("base", "link1"):
        bid = robot_map.body_ids[name][0]
        np.testing.assert_allclose(
            model.body_ipos[bid], robot_map.default_body_ipos[bid] + [0.01, 0.0, -0.02], atol=1e-12
        )


# ---------------------------------------------------------------------------------------
# Reset events
# ---------------------------------------------------------------------------------------


def test_reset_root_state_uniform_pose_and_velocity_frames():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "reset_base",
        "reset_root_state_uniform",
        "reset",
        {
            "pose_range": {"x": [1.0, 1.0], "yaw": [math.pi / 2, math.pi / 2]},
            "velocity_range": {"x": [0.5, 0.5], "yaw": [2.0, 2.0]},
        },
    )
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    qa, da = robot_map.root_qpos_adr, robot_map.root_dof_adr
    # pos = default (0,0,0.5) + origin (2,0,0) + sample (1,0,0).
    np.testing.assert_allclose(data.qpos[qa : qa + 3], [3.0, 0.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(data.qpos[qa + 3 : qa + 7], quat_from_euler_xyz(0.0, 0.0, math.pi / 2), atol=1e-12)
    # base body CoM is at its frame origin -> qvel lin equals the sampled world CoM lin vel;
    # angular: R^T (0,0,2) = (0,0,2) for a pure yaw rotation.
    np.testing.assert_allclose(data.qvel[da : da + 3], [0.5, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(data.qvel[da + 3 : da + 6], [0.0, 0.0, 2.0], atol=1e-12)
    # Round-trip through the CoM-velocity reader.
    np.testing.assert_allclose(read_root_com_vel_w(model, data, robot_map), [0.5, 0, 0, 0, 0, 2], atol=1e-12)


def test_reset_joints_by_offset_clamps_to_soft_limits():
    model, data, robot_map = make_model_and_map()
    robot_map.default_joint_pos = np.array([0.95, -1.45])  # near limits so offsets must clamp
    ev = event(
        "reset_joints",
        "reset_joints_by_offset",
        "reset",
        {"position_range": [0.2, 0.2], "velocity_range": [0.0, 0.0]},
    )
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    assert data.qpos[robot_map.qpos_adr[0]] == pytest.approx(1.0)  # clamped to j1 upper limit
    assert data.qpos[robot_map.qpos_adr[1]] == pytest.approx(-1.25)  # -1.45 + 0.2, inside limits


def test_reset_joints_by_scale_identity_matches_g1_cfg():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "reset_joints",
        "reset_joints_by_scale",
        "reset",
        {"position_range": [1.0, 1.0], "velocity_range": [0.0, 0.0]},
    )
    data.qvel[robot_map.dof_adr] = 1.0
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    np.testing.assert_allclose(data.qpos[robot_map.qpos_adr], robot_map.default_joint_pos, atol=1e-12)
    np.testing.assert_allclose(data.qvel[robot_map.dof_adr], 0.0, atol=1e-12)


def _spot_reset_around_default_event(params) -> EventIR:
    # The stock Spot task ships this term from its task-local mdp module; the func
    # reference in a dumped env.yaml carries that full path.
    return EventIR(
        name="reset_joints",
        func="isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp.events:reset_joints_around_default",
        mode="reset",
        params=params,
        interval_range_s=None,
    )


def test_reset_joints_around_default_matches_hand_computed_sampling():
    """Isaac clamps the sampling-interval ENDPOINTS (default + range bounds) to the soft
    limits and then draws uniformly inside, positions before velocities."""
    model, data, robot_map = make_model_and_map()
    robot_map.default_joint_pos = np.array([0.95, -1.45])  # near the j1/j2 limits
    robot_map.joint_vel_limits = np.array([2.0, 10.0])
    ev = _spot_reset_around_default_event({"position_range": [-0.2, 0.3], "velocity_range": [-5.0, 1.0]})
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(42))

    rng = np.random.default_rng(42)
    # j1 limits (-1, 1): [0.75, 1.25] -> [0.75, 1.0]; j2 limits (-1.5, 1.5): [-1.65, -1.15] -> [-1.5, -1.15].
    expected_pos = rng.uniform([0.75, -1.5], [1.0, -1.15])
    # vel endpoints default(0) + (-5, 1), clamped to +-vel_limit: j1 -> [-2, 1]; j2 -> [-5, 1].
    expected_vel = rng.uniform([-2.0, -5.0], [1.0, 1.0])
    np.testing.assert_array_equal(data.qpos[robot_map.qpos_adr], expected_pos)
    np.testing.assert_array_equal(data.qvel[robot_map.dof_adr], expected_vel)


def test_reset_joints_around_default_writes_all_joints():
    # The Isaac implementation ignores asset_cfg joint selection and writes the full
    # joint state; the port must do the same.
    model, data, robot_map = make_model_and_map()
    ev = _spot_reset_around_default_event(
        {"asset_cfg": {"joint_names": ["j1"]}, "position_range": [0.1, 0.1], "velocity_range": [0.0, 0.0]}
    )
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    np.testing.assert_allclose(data.qpos[robot_map.qpos_adr], robot_map.default_joint_pos + 0.1, atol=1e-12)


def test_apply_external_force_torque():
    model, data, robot_map = make_model_and_map()
    zero = event(
        "wrench0",
        "apply_external_force_torque",
        "reset",
        {"asset_cfg": {"body_names": ["base"]}, "force_range": [0.0, 0.0], "torque_range": [0.0, 0.0]},
    )
    EventSet([zero]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    np.testing.assert_array_equal(data.xfrc_applied, 0.0)

    ev = event(
        "wrench",
        "apply_external_force_torque",
        "reset",
        {"asset_cfg": {"body_names": ["base"]}, "force_range": [2.0, 2.0], "torque_range": [0.0, 0.0]},
    )
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    bid = robot_map.body_ids["base"][0]
    # Identity orientation, CoM at frame origin -> the body-frame wrench passes through.
    np.testing.assert_allclose(data.xfrc_applied[bid, :3], [2.0, 2.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(data.xfrc_applied[bid, 3:], 0.0, atol=1e-12)


def test_clear_external_wrenches_stops_reapplication():
    # Exact-state resets run no reset events; a held wrench from the previous
    # episode must not be re-applied to the restored state (Isaac zeroes held
    # wrenches in every _reset_idx).
    model, data, robot_map = make_model_and_map()
    ev = event(
        "wrench",
        "apply_external_force_torque",
        "reset",
        {"asset_cfg": {"body_names": ["base"]}, "force_range": [2.0, 2.0], "torque_range": [0.0, 0.0]},
    )
    events = EventSet([ev])
    events.apply_reset(model, data, robot_map, np.random.default_rng(0))
    events.clear_external_wrenches()
    data.xfrc_applied[:] = 0.0
    events.refresh_external_wrenches(model, data)
    np.testing.assert_array_equal(data.xfrc_applied, 0.0)


# ---------------------------------------------------------------------------------------
# Interval events
# ---------------------------------------------------------------------------------------


def test_push_by_setting_velocity_interval_clock():
    model, data, robot_map = make_model_and_map()
    ev = event(
        "push_robot",
        "push_by_setting_velocity",
        "interval",
        {"asset_cfg": {"body_names": "base"}, "velocity_range": {"x": [1.0, 1.0]}},
        interval=(0.5, 0.5),
    )
    events = EventSet([ev])
    rng = np.random.default_rng(0)
    events.apply_reset(model, data, robot_map, rng)  # samples the interval clock (0.5 s)
    da = robot_map.root_dof_adr
    data.qvel[:] = 0.0

    assert events.step_interval(0.25, model, data, robot_map, rng) is False
    assert data.qvel[da] == pytest.approx(0.0)
    assert events.step_interval(0.25, model, data, robot_map, rng) is True  # clock hits zero
    assert data.qvel[da] == pytest.approx(1.0)  # x velocity ADDED to current (zero)
    assert events.step_interval(0.25, model, data, robot_map, rng) is False  # clock resampled to 0.5
    assert events.step_interval(0.25, model, data, robot_map, rng) is True
    assert data.qvel[da] == pytest.approx(2.0)  # additive on top of the previous push


def test_interval_mass_randomization_refreshes_model_constants():
    """Interval-fired mass/CoM events must run the same mj_setConst refresh as
    startup/reset applications, or derived constants (subtree masses) go stale."""
    model, data, robot_map = make_model_and_map()
    ev = event(
        "add_mass",
        "randomize_rigid_body_mass",
        "interval",
        {
            "asset_cfg": {"body_names": "base"},
            "mass_distribution_params": [1.0, 1.0],
            "operation": "add",
            "distribution": "uniform",
        },
        interval=(0.5, 0.5),
    )
    events = EventSet([ev])
    rng = np.random.default_rng(0)
    events.apply_reset(model, data, robot_map, rng)
    bid = robot_map.body_ids["base"][0]
    subtree_before = float(model.body_subtreemass[bid])

    assert events.step_interval(0.5, model, data, robot_map, rng) is True
    assert model.body_mass[bid] == pytest.approx(robot_map.default_body_mass[bid] + 1.0)
    # mj_setConst propagated the new mass into the derived subtree mass.
    assert model.body_subtreemass[bid] == pytest.approx(subtree_before + 1.0)


# ---------------------------------------------------------------------------------------
# Strict mode, validation, fixture coverage
# ---------------------------------------------------------------------------------------


def test_strict_mode_noops_everything():
    model, data, robot_map = make_model_and_map()
    evs = [
        event(
            "mass",
            "randomize_rigid_body_mass",
            "startup",
            {
                "asset_cfg": {"body_names": "base"},
                "mass_distribution_params": [-2.0, 2.0],
                "operation": "add",
                "distribution": "uniform",
            },
        ),
        event(
            "reset_base",
            "reset_root_state_uniform",
            "reset",
            {"pose_range": {"x": [1.0, 1.0]}, "velocity_range": {"x": [1.0, 1.0]}},
        ),
        event("push", "push_by_setting_velocity", "interval", {"velocity_range": {"x": [1, 1]}}, interval=(0.1, 0.1)),
    ]
    events = EventSet(evs, strict=True)
    rng = np.random.default_rng(0)
    qpos_before = data.qpos.copy()
    events.apply_startup(model, data, robot_map, rng)
    events.apply_reset(model, data, robot_map, rng)
    assert events.step_interval(10.0, model, data, robot_map, rng) is False
    np.testing.assert_array_equal(model.body_mass, robot_map.default_body_mass)
    np.testing.assert_array_equal(data.qpos, qpos_before)


def test_unknown_event_mode_raises():
    # e.g. IsaacLab's 'prestartup' mode: accepted silently before, then never applied.
    ev = event(
        "spawn_mass",
        "randomize_rigid_body_mass",
        "prestartup",
        {"asset_cfg": {"body_names": "base"}, "mass_distribution_params": [1.0, 1.0], "operation": "scale"},
    )
    with pytest.raises(ValueError, match="unsupported mode"):
        EventSet([ev])


def test_seeded_events_are_deterministic():
    def run():
        model, data, robot_map = make_model_and_map()
        ir = parse_fixture("spot_velocity_env.yaml")
        # Retarget fixture body/geom patterns onto the toy model's names.
        events = EventSet(
            [e for e in ir.events if e.mode == "reset"],
        )
        events.apply_reset(model, data, robot_map, np.random.default_rng(99))
        return data.qpos.copy(), data.qvel.copy()

    (qpos_a, qvel_a), (qpos_b, qvel_b) = run(), run()
    np.testing.assert_array_equal(qpos_a, qpos_b)
    np.testing.assert_array_equal(qvel_a, qvel_b)


@pytest.mark.parametrize("fixture", ["spot_velocity_env.yaml", "g1_flat_env.yaml"])
def test_every_fixture_event_is_supported(fixture):
    ir = parse_fixture(fixture)
    assert ir.events, "fixture should declare events"
    events = EventSet(ir.events)  # unknown funcs would raise
    assert len(ir.events) == len(events._events)


def test_g1_fixture_events_apply_on_toy_model():
    # G1's reset events use default asset_cfg (all bodies/joints) except the external
    # wrench on torso_link; alias that name onto the toy model's base body.
    model, data, robot_map = make_model_and_map()
    robot_map.body_ids["torso_link"] = robot_map.body_ids["base"]
    robot_map.geom_ids["torso_link"] = robot_map.geom_ids["base"]
    ir = parse_fixture("g1_flat_env.yaml")
    assert ir.terrain is not None
    events = EventSet(
        ir.events,
        terrain_static_friction=float(ir.terrain.physics_material["static_friction"]),
        friction_combine_mode=str(ir.terrain.physics_material["friction_combine_mode"]),
    )
    rng = np.random.default_rng(0)
    events.apply_startup(model, data, robot_map, rng)
    events.apply_reset(model, data, robot_map, rng)
    # G1 velocity_range is all zeros and joint scale is (1,1): deterministic reset state.
    np.testing.assert_allclose(data.qpos[robot_map.qpos_adr], robot_map.default_joint_pos, atol=1e-12)
    np.testing.assert_array_equal(data.xfrc_applied, 0.0)
    qa = robot_map.root_qpos_adr
    pos = data.qpos[qa : qa + 3] - robot_map.env_origin_w - robot_map.default_root_pose_env[:3]
    assert -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and pos[2] == pytest.approx(0.0)
    # Friction: static range (0.8, 0.8) x terrain 1.0 -> exactly 0.8 on every robot geom.
    for gs in robot_map.geom_ids.values():
        for g in gs:
            assert model.geom_friction[g, 0] == pytest.approx(0.8)


def test_reset_mass_event_keeps_the_robot_state():
    # mj_setConst evaluates at qpos0 and leaves it in the data it is given: the refresh
    # after a reset/interval mass event must not teleport the robot.
    model, data, robot_map = make_model_and_map()
    data.qpos[:3] = [1.0, 2.0, 3.0]
    data.qpos[robot_map.qpos_adr] = [0.2, -0.3]
    before = data.qpos.copy()
    ev = event(
        "base_mass",
        "randomize_rigid_body_mass",
        "reset",
        {"asset_cfg": {"body_names": "base"}, "mass_distribution_params": [1.0, 1.0], "operation": "add"},
    )
    EventSet([ev]).apply_reset(model, data, robot_map, np.random.default_rng(0))
    np.testing.assert_array_equal(data.qpos, before)
    bid = robot_map.body_ids["base"][0]
    assert model.body_subtreemass[bid] == pytest.approx(robot_map.default_body_mass[bid : bid + 3].sum() + 1.0)


@pytest.mark.parametrize("func", ["randomize_actuator_gains", "randomize_joint_parameters"])
def test_actuator_randomization_is_strict_only(func):
    model, data, robot_map = make_model_and_map()
    ev = event(func, func, "startup", {"asset_cfg": {"joint_names": [".*"]}})
    events = EventSet([ev], strict=True)
    before = (model.dof_damping.copy(), model.dof_frictionloss.copy(), model.dof_armature.copy())
    events.apply_startup(model, data, robot_map, np.random.default_rng(0))
    for a, b in zip(before, (model.dof_damping, model.dof_frictionloss, model.dof_armature)):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(NotImplementedError, match="strict mode"):
        EventSet([ev])
