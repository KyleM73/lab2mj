"""Tests for lab2mj.env_yaml parsing against the real Spot and G1 dumps."""

import numpy as np
import pytest

from lab2mj.env_yaml import (
    _parse_noise,
    _parse_obs_groups,
    _parse_terrain,
    actuator_model_from_class,
    class_name,
    load_env_yaml,
    parse_env_dict,
)
from lab2mj.ir import EnvIR

from .shared import FIXTURES, G1_ENV_YAML, SPOT_ENV_YAML, parse_fixture

NAV_ENV_YAML = FIXTURES / "anymal_c_nav_env_like.yaml"


@pytest.fixture(scope="module")
def spot_ir() -> EnvIR:
    if not SPOT_ENV_YAML.exists():
        pytest.skip(f"missing fixture {SPOT_ENV_YAML}")
    return parse_fixture(SPOT_ENV_YAML)


@pytest.fixture(scope="module")
def g1_ir() -> EnvIR:
    return parse_fixture(G1_ENV_YAML)


@pytest.fixture(scope="module")
def nav_ir() -> EnvIR:
    return parse_fixture(NAV_ENV_YAML)


class TestClassNameParsing:
    def test_actuator_model_map(self):
        assert actuator_model_from_class("isaaclab.actuators.actuator_pd:ImplicitActuator") == "implicit_pd"
        bracket_form = "<class 'isaaclab.actuators.actuator_pd.ImplicitActuatorCfg'>"
        assert actuator_model_from_class(bracket_form) == "implicit_pd"
        assert actuator_model_from_class("isaaclab.actuators.actuator_pd:IdealPDActuator") == "ideal_pd"
        assert actuator_model_from_class("isaaclab.actuators.actuator_pd:DelayedPDActuator") == "delayed_pd"
        assert actuator_model_from_class("isaaclab.actuators.actuator_pd:RemotizedPDActuator") == "remotized_pd"
        assert actuator_model_from_class("contact_lab.assets.actuators:HostSyncFreeDelayedPDActuator") == "delayed_pd"
        assert (
            actuator_model_from_class("contact_lab.assets.actuators:HostSyncFreeRemotizedPDActuator") == "remotized_pd"
        )


class TestSpotTiming:
    def test_timing(self, spot_ir: EnvIR):
        assert spot_ir.timing.physics_dt == 0.005
        assert spot_ir.timing.decimation == 4
        assert spot_ir.timing.episode_length_s == 20.0
        assert spot_ir.timing.policy_dt == pytest.approx(0.02)


class TestSpotObservations:
    def test_policy_terms_in_order(self, spot_ir: EnvIR):
        policy = spot_ir.obs_group("policy")
        assert [t.name for t in policy.terms] == [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        assert policy.enable_corruption is True
        assert policy.concatenate_terms is True
        assert policy.history_length == 1

    def test_noise_ranges(self, spot_ir: EnvIR):
        policy = spot_ir.obs_group("policy")
        expected = {
            "base_lin_vel": (-0.1, 0.1),
            "base_ang_vel": (-0.1, 0.1),
            "projected_gravity": (-0.05, 0.05),
            "velocity_commands": None,
            "joint_pos": (-0.05, 0.05),
            "joint_vel": (-0.5, 0.5),
            "actions": None,
        }
        for term in policy.terms:
            noise = term.noise
            if expected[term.name] is None:
                assert noise is None, term.name
            else:
                assert noise is not None, term.name
                assert noise.kind == "uniform"
                assert noise.operation == "add"
                assert (noise.params["n_min"], noise.params["n_max"]) == expected[term.name]

    def test_critic_group_uncorrupted(self, spot_ir: EnvIR):
        critic = spot_ir.obs_group("critic")
        assert critic.enable_corruption is False
        assert len(critic.terms) == 7
        assert all(t.noise is None for t in critic.terms)


class TestSpotAction:
    def test_action_term(self, spot_ir: EnvIR):
        action = spot_ir.action
        assert class_name(action.func) == "JointPositionAction"
        assert action.joint_names_expr == [".*"]
        assert action.scale == 0.2
        assert action.offset == 0.0
        assert action.use_default_offset is True
        assert action.preserve_order is False
        assert action.clip is None


class TestSpotActuators:
    def test_knee_group(self, spot_ir: EnvIR):
        knee = next(a for a in spot_ir.actuators if a.name == "spot_knee")
        assert knee.model == "remotized_pd"
        assert knee.joint_names_expr == [".*_kn"]
        assert knee.stiffness == 60.0
        assert knee.damping == 1.5
        assert knee.effort_limit is None
        assert knee.friction == 0.18
        assert (knee.min_delay, knee.max_delay) == (0, 1)
        lut = knee.joint_parameter_lookup
        assert lut is not None and lut.shape == (101, 3)
        np.testing.assert_allclose(lut[0], [-2.7929, -24.776718, 37.165077])
        np.testing.assert_allclose(lut[-1], [-0.2471, -20.401667, 30.6025])
        # Angle column must be monotonic for np.interp-based torque clamping.
        assert np.all(np.diff(lut[:, 0]) > 0)


class TestSpotInitAndScene:
    def test_contact_sensor(self, spot_ir: EnvIR):
        (sensor,) = spot_ir.contact_sensors
        assert sensor.name == "contact_forces"
        assert sensor.prim_path == "/World/envs/env_.*/Robot/.*"
        assert sensor.history_length == 4
        assert sensor.track_air_time is True
        assert sensor.force_threshold == 1.0
        assert sensor.update_period == 0.005

    def test_sync_free_contact_sensor(self):
        data = load_env_yaml(SPOT_ENV_YAML)
        data["scene"]["contact_forces"]["class_type"] = "contact_lab.sensors:HostSyncFreeContactSensor"
        (sensor,) = parse_env_dict(data).contact_sensors
        assert sensor.name == "contact_forces" and sensor.history_length == 4


class TestSpotCommands:
    def test_base_velocity(self, spot_ir: EnvIR):
        (cmd,) = spot_ir.commands
        assert cmd.name == "base_velocity"
        assert cmd.type == "UniformVelocityCommand"
        assert cmd.params["resampling_time_range"] == [10.0, 10.0]
        assert cmd.params["rel_standing_envs"] == 0.1
        assert cmd.params["rel_heading_envs"] == 0.0
        assert cmd.params["heading_command"] is False
        ranges = cmd.params["ranges"]
        assert ranges["lin_vel_x"] == [-1.5, 1.5]
        assert ranges["lin_vel_y"] == [-1.0, 1.0]
        assert ranges["ang_vel_z"] == [-1.0, 1.0]
        assert ranges["heading"] is None
        # Visualizer-only sub-configs must not leak into params.
        assert not any(k.endswith("visualizer_cfg") for k in cmd.params)


class TestSpotEvents:
    def test_base_mass(self, spot_ir: EnvIR):
        event = next(e for e in spot_ir.events if e.name == "base_mass")
        assert event.func == "isaaclab.envs.mdp.events:randomize_rigid_body_mass"
        assert event.params["mass_distribution_params"] == [-2.5, 2.5]
        assert event.params["operation"] == "add"
        assert event.params["distribution"] == "uniform"
        assert event.params["asset_cfg"]["body_names"] == "body"
        # Full slices from the dump are sanitized to None ("all").
        assert event.params["asset_cfg"]["joint_ids"] is None

    def test_push_interval(self, spot_ir: EnvIR):
        event = next(e for e in spot_ir.events if e.name == "push_robot")
        assert event.interval_range_s == (10.0, 15.0)
        assert event.params["velocity_range"]["x"] == [-0.5, 0.5]
        assert event.params["velocity_range"]["y"] == [-0.5, 0.5]


class TestSpotTerminations:
    def test_terms(self, spot_ir: EnvIR):
        by_name = {t.name: t for t in spot_ir.terminations}
        assert by_name["time_out"].time_out is True
        assert by_name["time_out"].func == "isaaclab.envs.mdp.terminations:time_out"
        body_contact = by_name["body_contact"]
        assert body_contact.time_out is False
        assert body_contact.func == "isaaclab.envs.mdp.terminations:illegal_contact"
        assert body_contact.params["threshold"] == 1.0
        assert body_contact.params["sensor_cfg"]["body_names"] == ["body", ".*leg"]
        assert by_name["terrain_out_of_bounds"].time_out is True


class TestG1Fixture:
    """The real Isaac-Velocity-Flat-G1-v0 dump: dict-valued cfg fields, actuator
    coalescing, and null-entry dropping that the Spot fixture doesn't exercise."""

    def test_smoke(self, g1_ir: EnvIR):
        assert g1_ir.timing.physics_dt == 0.005
        assert g1_ir.timing.decimation == 4
        assert g1_ir.timing.episode_length_s == 20.0
        assert g1_ir.usd_path is not None and g1_ir.usd_path.endswith("Unitree/G1/g1_minimal.usd")
        assert g1_ir.terrain is not None
        assert g1_ir.terrain.terrain_type == "plane"
        assert g1_ir.terrain.generator is None
        assert g1_ir.terrain.num_envs == 4096
        assert g1_ir.terrain.env_spacing == 2.5
        assert g1_ir.terrain.max_init_terrain_level == 5
        init = g1_ir.robot_init
        assert init.root_pos_env == (0.0, 0.0, 0.74)
        assert init.joint_pos[".*_knee_joint"] == 0.42
        assert len(init.joint_pos) == 12
        assert init.joint_vel == {".*": 0.0}
        (cmd,) = g1_ir.commands
        assert cmd.type == "UniformVelocityCommand"
        assert cmd.params["heading_command"] is True
        assert cmd.params["ranges"]["lin_vel_x"] == [0.0, 1.0]
        by_name = {t.name: t for t in g1_ir.terminations}
        assert by_name["time_out"].time_out is True
        assert by_name["base_contact"].params["sensor_cfg"]["body_names"] == "torso_link"
        (sensor,) = g1_ir.contact_sensors
        assert sensor.history_length == 3
        assert sensor.track_air_time is True

    def test_obs_terms(self, g1_ir: EnvIR):
        policy = g1_ir.obs_group("policy")
        # height_scan is nulled out in the flat cfg and must be dropped.
        assert [t.name for t in policy.terms] == [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        assert policy.history_length is None
        by_name = {t.name: t for t in policy.terms}
        ang_noise = by_name["base_ang_vel"].noise
        assert ang_noise is not None
        assert (ang_noise.params["n_min"], ang_noise.params["n_max"]) == (-0.2, 0.2)
        vel_noise = by_name["joint_vel"].noise
        assert vel_noise is not None
        assert (vel_noise.params["n_min"], vel_noise.params["n_max"]) == (-1.5, 1.5)
        assert by_name["joint_pos"].history_length == 0

    def test_per_joint_action_scale(self, g1_ir: EnvIR):
        assert class_name(g1_ir.action.func) == "JointPositionAction"
        assert g1_ir.action.scale == 0.5
        assert g1_ir.action.use_default_offset is True
        assert g1_ir.action.offset == 0.0
        # The stock cfg's scale is scalar; rewrite it as a per-joint regex dict to
        # exercise the dict-scale parser path.
        env = load_env_yaml(G1_ENV_YAML)
        env["actions"]["joint_pos"]["scale"] = {".*_knee_joint": 0.4, ".*": 0.5}
        action = parse_env_dict(env).action
        assert isinstance(action.scale, dict)
        assert action.scale[".*_knee_joint"] == 0.4
        assert action.scale[".*"] == 0.5

    def test_implicit_actuator_groups(self, g1_ir: EnvIR):
        assert [a.name for a in g1_ir.actuators] == ["legs", "feet", "arms"]
        assert all(a.model == "implicit_pd" for a in g1_ir.actuators)
        legs, feet, arms = g1_ir.actuators
        assert isinstance(legs.stiffness, dict) and legs.stiffness[".*_knee_joint"] == 200.0
        assert isinstance(legs.armature, dict) and legs.armature[".*_hip_.*"] == 0.01
        # effort_limit falls back to effort_limit_sim for implicit actuators.
        assert legs.effort_limit == 300
        assert feet.effort_limit == 20
        assert feet.stiffness == 20.0 and feet.damping == 2.0 and feet.armature == 0.01
        assert arms.effort_limit == 300
        assert isinstance(arms.armature, dict) and arms.armature[".*_five_joint"] == 0.001
        assert legs.min_delay is None and legs.max_delay is None
        assert legs.joint_parameter_lookup is None
        expr_count = sum(len(a.joint_names_expr) for a in g1_ir.actuators)
        assert expr_count == 19  # 5 + 2 + 12 regex groups covering all 37 joints

    def test_events_null_entries_dropped(self, g1_ir: EnvIR):
        names = [e.name for e in g1_ir.events]
        assert names == ["physics_material", "base_external_force_torque", "reset_base", "reset_robot_joints"]
        reset_joints = next(e for e in g1_ir.events if e.name == "reset_robot_joints")
        assert reset_joints.func == "isaaclab.envs.mdp.events:reset_joints_by_scale"
        assert reset_joints.params["position_range"] == [1.0, 1.0]
        assert reset_joints.params["velocity_range"] == [0.0, 0.0]
        reset_base = next(e for e in g1_ir.events if e.name == "reset_base")
        assert reset_base.params["velocity_range"]["yaw"] == [0.0, 0.0]


class TestNavPreTrainedPolicyAction:
    """PreTrainedPolicyAction parsing on the nav-shaped ANYmal-C fixture."""

    def test_wrapper_term(self, nav_ir: EnvIR):
        action = nav_ir.action
        assert class_name(action.func) == "PreTrainedPolicyAction"
        assert action.policy_path is not None
        assert action.policy_path.endswith("Policies/ANYmal-C/Blind/policy.pt")
        assert action.low_level_decimation == 4
        assert action.policy_bundle_path is None  # set by convert, not by parsing

    def test_embedded_low_level_action(self, nav_ir: EnvIR):
        low = nav_ir.action.low_level_action
        assert low is not None
        assert class_name(low.func) == "JointPositionAction"
        assert low.joint_names_expr == [".*"]
        assert low.scale == 0.5
        assert low.use_default_offset is True
        assert low.low_level_action is None  # no recursive wrapping

    def test_embedded_low_level_obs_group(self, nav_ir: EnvIR):
        group = nav_ir.action.low_level_obs
        assert group is not None
        assert group.name == "ll_policy"
        assert [t.name for t in group.terms] == [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        assert group.enable_corruption is True
        by_name = {t.name: t for t in group.terms}
        # The remapped terms carry the lambda source strings dump_yaml records
        # post-construction; params are cleared by the remap.
        assert by_name["velocity_commands"].func.startswith("lambda ")
        assert by_name["velocity_commands"].params == {}
        assert by_name["actions"].func.startswith("lambda ")
        noise = by_name["joint_vel"].noise
        assert noise is not None and (noise.params["n_min"], noise.params["n_max"]) == (-1.5, 1.5)

    def test_round_trip(self, nav_ir: EnvIR):
        as_dict = nav_ir.to_dict()
        rebuilt = EnvIR.from_dict(as_dict)
        assert rebuilt.to_dict() == as_dict
        assert rebuilt.action.low_level_obs is not None
        assert rebuilt.action.low_level_action is not None

    _WRAPPER_CLASS = "isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action:PreTrainedPolicyAction"

    def test_missing_policy_path_raises(self):
        from lab2mj.env_yaml import _parse_pre_trained_policy_action

        term = {
            "class_type": self._WRAPPER_CLASS,
            "low_level_actions": {"class_type": "x:JointPositionAction"},
            "low_level_observations": {},
        }
        with pytest.raises(ValueError, match="policy_path"):
            _parse_pre_trained_policy_action(term)


class TestUnsupportedObsFeatures:
    def test_obs_term_modifiers_raise(self):
        groups = {
            "policy": {
                "joint_pos": {
                    "func": "isaaclab.envs.mdp.observations:joint_pos_rel",
                    "modifiers": [{"func": "isaaclab.utils.modifiers:clip"}],
                }
            }
        }
        with pytest.raises(ValueError, match="modifiers"):
            _parse_obs_groups(groups)

    def test_noise_model_cfg_raises(self):
        noise = {"class_type": "isaaclab.utils.noise:NoiseModel", "noise_cfg": {"n_min": -0.1, "n_max": 0.1}}
        with pytest.raises(ValueError, match="NoiseModelCfg"):
            _parse_noise(noise)


class TestTerrainSceneFallback:
    def test_scene_values_fill_missing_terrain_layout(self):
        terrain_cfg = {"terrain_type": "plane", "terrain_generator": None, "physics_material": {}}
        scene = {"num_envs": 64, "env_spacing": 1.5}
        ir = _parse_terrain(terrain_cfg, scene)
        assert ir is not None
        assert ir.num_envs == 64
        assert ir.env_spacing == 1.5
        assert ir.max_init_terrain_level is None

    def test_terrain_block_wins_over_scene(self):
        terrain_cfg = {
            "terrain_type": "plane",
            "num_envs": 32,
            "env_spacing": 4.0,
            "max_init_terrain_level": 2,
        }
        ir = _parse_terrain(terrain_cfg, {"num_envs": 64, "env_spacing": 1.5})
        assert ir is not None
        assert ir.num_envs == 32
        assert ir.env_spacing == 4.0
        assert ir.max_init_terrain_level == 2


class TestRoundTrip:
    def test_g1_round_trip(self, g1_ir: EnvIR):
        as_dict = g1_ir.to_dict()
        rebuilt = EnvIR.from_dict(as_dict)
        assert rebuilt.to_dict() == as_dict

    def test_spot_round_trip(self, spot_ir: EnvIR):
        as_dict = spot_ir.to_dict()
        rebuilt = EnvIR.from_dict(as_dict)
        assert rebuilt.to_dict() == as_dict
        lut = rebuilt.actuators[1].joint_parameter_lookup
        assert isinstance(lut, np.ndarray) and lut.shape == (101, 3)


class TestRequiredKeys:
    def test_missing_sim_dt_raises_value_error(self):
        from lab2mj.env_yaml import load_env_yaml, parse_env_dict

        data = load_env_yaml(SPOT_ENV_YAML)
        del data["sim"]["dt"]
        with pytest.raises(ValueError, match="sim.dt"):
            parse_env_dict(data)


class TestPhysicsMaterials:
    def test_sim_default_material_parsed(self, spot_ir: EnvIR):
        assert spot_ir.sim_physics_material is not None
        assert spot_ir.sim_physics_material["static_friction"] == 1.0
        assert spot_ir.sim_physics_material["friction_combine_mode"] == "multiply"
        assert "func" not in spot_ir.sim_physics_material


class TestSolverPositionIterations:
    def test_spot_articulation_tgs_iterations(self, spot_ir: EnvIR):
        assert spot_ir.solver_position_iterations == 4
