"""End-to-end tests for the MuJoCo runtime env (offline; heavy cases skip without assets)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from lab2mj.env import ActionProcessor, free_joint_qvel

from .shared import FIXTURES, G1_USD, HAS_PXR, HAS_TORCH, REPO_ROOT, SPOT_USD, needs_g1, needs_spot  # noqa: F401


def _find_g1_policy() -> Path:
    """A jit-exported G1 velocity policy: the cached copy, else any local training run."""
    cached = REPO_ROOT / "data" / "policy_cache" / "g1_flat_policy.pt"
    if cached.exists():
        return cached
    runs = sorted((REPO_ROOT / "logs" / "g1_flat").glob("*/exported/policy.pt"))
    return runs[-1] if runs else cached


G1_POLICY = _find_g1_policy()

needs_g1_policy = pytest.mark.skipif(
    not (HAS_PXR and G1_USD.exists() and G1_POLICY.exists() and HAS_TORCH),
    reason="requires the G1 assets, torch, and the exported G1 policy.pt under logs/",
)

GO1_BUNDLE = REPO_ROOT / "data" / "mj_bundles" / "go1_flat"
GO1_DUMP = REPO_ROOT / "data" / "sim2sim_dumps" / "unitree_go1_fwd_settled_phys.npz"
needs_go1_bundle = pytest.mark.skipif(
    not ((GO1_BUNDLE / "manifest.json").exists() and GO1_DUMP.exists() and HAS_TORCH),
    reason="requires torch, the converted go1_flat bundle, and the go1 reference dump under data/",
)


@pytest.fixture(scope="module")
def g1_bundle(tmp_path_factory):
    from lab2mj.convert import convert_run

    out = tmp_path_factory.mktemp("g1_bundle")
    return convert_run(FIXTURES / "g1_flat_env.yaml", usd=G1_USD, out=out)


@pytest.fixture(scope="module")
def spot_bundle(tmp_path_factory):
    from lab2mj.convert import convert_run

    out = tmp_path_factory.mktemp("spot_bundle")
    return convert_run(FIXTURES / "spot_velocity_env.yaml", usd=SPOT_USD, out=out)


@needs_g1
class TestG1Bundle:
    def test_bundle_contents(self, g1_bundle):
        from lab2mj.bundle import read_manifest

        assert (g1_bundle / "scene.xml").is_file()
        assert (g1_bundle / "robot.xml").is_file()
        manifest = read_manifest(g1_bundle)
        assert len(manifest["robot"]["isaac_joint_order"]) == 37
        assert manifest["obs"]["obs_dims"]["policy"] == 123
        assert manifest["action"]["dim"] == 37
        layout = manifest["obs"]["layouts"]["policy"]
        assert layout[-1]["stop"] == 123

    def test_strict_reset_and_zero_action_steps(self, g1_bundle):
        from lab2mj.env import MjEnv

        env = MjEnv(g1_bundle, strict=True, seed=0)
        obs = env.reset()
        assert obs.shape == (123,)
        assert np.all(np.isfinite(obs))
        # Default init: joint_pos_rel and last_action terms are exactly zero.
        layout = {e["name"]: (e["start"], e["stop"]) for e in env.manifest["obs"]["layouts"]["policy"]}
        start, stop = layout["joint_pos"]
        np.testing.assert_allclose(obs[start:stop], 0.0, atol=1e-6)
        for _ in range(10):
            obs, terminated, truncated, _ = env.step(np.zeros(env.action_dim))
            assert obs.shape == (123,)
            assert np.all(np.isfinite(obs))
        assert not truncated

    def test_strict_determinism(self, g1_bundle):
        from lab2mj.env import MjEnv

        traces = []
        for _ in range(2):
            env = MjEnv(g1_bundle, strict=True, seed=0, command=[0.5, 0.0, 0.0])
            env.reset()
            trace = [env.step(np.full(env.action_dim, 0.05))[0] for _ in range(5)]
            traces.append(np.stack(trace))
        np.testing.assert_array_equal(traces[0], traces[1])

    def test_command_pinned_in_obs(self, g1_bundle):
        from lab2mj.env import MjEnv

        command = np.array([0.7, -0.1, 0.3])
        env = MjEnv(g1_bundle, strict=True, seed=0, command=command)
        obs = env.reset()
        layout = {e["name"]: (e["start"], e["stop"]) for e in env.manifest["obs"]["layouts"]["policy"]}
        start, stop = layout["velocity_commands"]
        np.testing.assert_allclose(obs[start:stop], command, atol=1e-6)

    def test_manifest_env_origin_and_sim_fields(self, g1_bundle):
        from lab2mj.bundle import read_manifest
        from lab2mj.terrain import _env_origins_grid_w

        manifest = read_manifest(g1_bundle)
        assert manifest["sim"]["gravity_w"] == [0.0, 0.0, -9.81]
        # Grid-plane env 0 origin from the training run's num_envs/env_spacing.
        terrain = manifest["terrain"]
        expected = _env_origins_grid_w(int(terrain["num_envs"]), float(terrain["env_spacing"]))[0]
        np.testing.assert_allclose(manifest["init"]["env_origin_w"], expected, atol=1e-9)
        # Default root spawn sits on the env origin.
        np.testing.assert_allclose(np.asarray(manifest["robot"]["default_qpos"][:2]), expected[:2], atol=1e-9)
        # Source provenance hashes match the actual inputs.
        from lab2mj.bundle import sha256_file

        assert manifest["sources"]["usd_sha256"] == sha256_file(G1_USD)
        assert manifest["sources"]["env_yaml_sha256"] == sha256_file(FIXTURES / "g1_flat_env.yaml")

    def test_gravity_mismatch_raises(self, g1_bundle, tmp_path):
        import json
        import shutil

        from lab2mj.env import MjEnv

        tampered = tmp_path / "tampered_bundle"
        shutil.copytree(g1_bundle, tampered)
        manifest_path = tampered / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["sim"]["gravity_w"] = [0.0, 0.0, -3.71]
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="gravity"):
            MjEnv(tampered, strict=True, seed=0)

    def test_reset_preserves_reset_event_wrenches(self, g1_bundle):
        """reset() and _auto_reset() must leave identical apply_external_force_torque
        effects: the wrench persists until the next reset (Isaac semantics)."""
        from lab2mj.env import MjEnv
        from lab2mj.events import EventSet
        from lab2mj.ir import EventIR

        env = MjEnv(g1_bundle, strict=False, seed=0)
        push = EventIR(
            name="push_torso",
            func="isaaclab.envs.mdp.events:apply_external_force_torque",
            mode="reset",
            params={
                "force_range": (50.0, 50.0),
                "torque_range": (0.0, 0.0),
                # A surviving body: a nonzero wrench on a welded-away link raises.
                "asset_cfg": {"body_names": "torso_link"},
            },
        )
        env._events = EventSet([push])
        env._startup_done = True

        env.reset()
        xfrc_reset = float(np.abs(env.data.xfrc_applied).sum())
        env._auto_reset()
        xfrc_auto = float(np.abs(env.data.xfrc_applied).sum())
        assert xfrc_reset > 0.0, "reset() wiped the reset-event wrench"
        assert xfrc_reset == pytest.approx(xfrc_auto)


@needs_g1
class TestPhysicsSubsteps:
    def test_default_bundle_authors_substep_timestep_and_solref(self, g1_bundle):
        from lab2mj.bundle import read_manifest
        from lab2mj.env import MjEnv

        manifest = read_manifest(g1_bundle)
        physics_dt = float(manifest["timing"]["physics_dt"])
        substeps = int(manifest["timing"]["physics_substeps"])
        # Default derives from the articulation's TGS solver_position_iteration_count (G1: 8).
        assert substeps == 8
        model = mujoco.MjModel.from_xml_path(str(g1_bundle / "scene.xml"))
        assert model.opt.timestep == pytest.approx(physics_dt / substeps)
        # Contact solref is authored against the substep dt (stiffest stable reference).
        np.testing.assert_allclose(model.geom_solref[:, 0], 2.0 * physics_dt / substeps)

        env = MjEnv(g1_bundle, strict=True, seed=0)
        assert env.physics_substeps == substeps
        assert env.substep_dt == pytest.approx(physics_dt / substeps)
        env.reset()
        env.step(np.zeros(env.action_dim))
        # One policy step still advances exactly decimation * physics_dt of sim time.
        assert env.data.time == pytest.approx(env.policy_dt)

    def test_substeps_one_and_legacy_manifest(self, tmp_path):
        import json

        from lab2mj.bundle import read_manifest
        from lab2mj.convert import convert_run
        from lab2mj.env import MjEnv

        bundle = convert_run(FIXTURES / "g1_flat_env.yaml", usd=G1_USD, out=tmp_path / "n1", substeps=1)
        manifest = read_manifest(bundle)
        physics_dt = float(manifest["timing"]["physics_dt"])
        assert manifest["timing"]["physics_substeps"] == 1
        model = mujoco.MjModel.from_xml_path(str(bundle / "scene.xml"))
        assert model.opt.timestep == pytest.approx(physics_dt)
        np.testing.assert_allclose(model.geom_solref[:, 0], 2.0 * physics_dt)

        # Pre-substep manifests carry no physics_substeps key; that must read as 1.
        manifest_path = bundle / "manifest.json"
        legacy = json.loads(manifest_path.read_text())
        del legacy["timing"]["physics_substeps"]
        manifest_path.write_text(json.dumps(legacy))
        env = MjEnv(bundle, strict=True, seed=0)
        assert env.physics_substeps == 1
        assert np.all(np.isfinite(env.reset()))

    def test_manifest_scene_substep_mismatch_raises(self, g1_bundle, tmp_path):
        import json
        import shutil

        from lab2mj.env import MjEnv

        tampered = tmp_path / "tampered_bundle"
        shutil.copytree(g1_bundle, tampered)
        manifest_path = tampered / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["timing"]["physics_substeps"]  # scene.xml keeps the substep timestep
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="physics_substeps"):
            MjEnv(tampered, strict=True, seed=0)

    def test_implicit_pd_ctrl_recomputed_per_substep(self, g1_bundle, monkeypatch):
        """implicit_pd groups get fresh ctrl each substep: clip(kp*(q_des-q) - kd*qd)
        + kd*qd against the live substep state (PhysX re-evaluates the implicit drive
        per solver iteration; only explicit groups hold the boundary torque)."""
        from lab2mj.env import MjEnv

        env = MjEnv(g1_bundle, strict=True, seed=0)
        env.reset()
        assert env.physics_substeps > 1
        assert env._implicit_joint_ids.size == env.num_joints  # G1: every group is implicit_pd

        recorded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        real_step = mujoco.mj_step

        def spy(model, data, *args, **kwargs):
            recorded.append((data.ctrl.copy(), data.qpos.copy(), data.qvel.copy()))
            return real_step(model, data, *args, **kwargs)

        monkeypatch.setattr(mujoco, "mj_step", spy)
        action = np.full(env.action_dim, 0.1)
        env.step(action)
        assert len(recorded) == env.decimation * env.physics_substeps

        ids = env._implicit_joint_ids
        assert env._joint_action is not None
        q_des = env._joint_action.targets_isaac(action)[ids]
        kp, kd, effort = env._implicit_kp, env._implicit_kd, env._implicit_effort
        changed_within_step = False
        for k, (ctrl, qpos, qvel) in enumerate(recorded):
            q = qpos[env.robot_map.qpos_adr[ids]]
            qd = qvel[env.robot_map.dof_adr[ids]]
            tau = np.clip(kp * (q_des - q) - kd * qd, -effort, effort)
            # Every substep's ctrl is the clamped total PD at the CURRENT state,
            # with +kd*qd added back for the dof_damping split.
            np.testing.assert_allclose(ctrl[env._implicit_ctrl_adr], tau + kd * qd, atol=1e-12)
            boundary = k - k % env.physics_substeps
            if k != boundary and not np.array_equal(ctrl, recorded[boundary][0]):
                changed_within_step = True
        assert changed_within_step, "ctrl never changed within a physics step; recompute is not happening"


@needs_g1_policy
class TestG1Policy:
    def test_policy_rollout_stays_upright(self, g1_bundle):
        from lab2mj.env import MjEnv

        env = MjEnv(g1_bundle, policy_path=G1_POLICY, strict=True, seed=0, command=[0.5, 0.0, 0.0])
        env.reset()
        rec = env.run_policy(50)
        assert rec["obs"].shape == (50, 123)
        assert rec["action_raw"].shape == (50, 37)
        assert np.all(np.isfinite(rec["joint_pos"]))
        # Informative stability bar; the strict trajectory gates run against Isaac dumps.
        assert float(rec["root_pos_w"][:, 2].min()) > 0.5


@needs_spot
class TestSpotBundle:
    def test_bundle_and_strict_smoke(self, spot_bundle):
        from lab2mj.bundle import read_manifest
        from lab2mj.env import MjEnv

        manifest = read_manifest(spot_bundle)
        assert len(manifest["robot"]["isaac_joint_order"]) == 12
        assert manifest["obs"]["obs_dims"]["policy"] == 48
        # Welded Isaac foot bodies must resolve onto the lower legs.
        assert manifest["robot"]["body_map"]["fl_foot"] == "fl_lleg"

        env = MjEnv(spot_bundle, strict=True, seed=0)
        obs = env.reset()
        assert obs.shape == (48,)
        for _ in range(10):
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
            assert obs.shape == (48,)
            assert np.all(np.isfinite(obs))

    def test_explicit_ctrl_held_across_substeps(self, spot_bundle, monkeypatch):
        """Explicit actuator groups (Spot: delayed_pd + remotized_pd) keep the ctrl
        computed at the physics-step boundary for all substeps — Isaac computes their
        torque once per physics step, so per-substep recompute would be unfaithful."""
        from lab2mj.env import MjEnv

        env = MjEnv(spot_bundle, strict=True, seed=0)
        env.reset()
        assert env._implicit_joint_ids.size == 0  # Spot has no implicit_pd group

        recorded: list[np.ndarray] = []
        real_step = mujoco.mj_step

        def spy(model, data, *args, **kwargs):
            recorded.append(data.ctrl.copy())
            return real_step(model, data, *args, **kwargs)

        monkeypatch.setattr(mujoco, "mj_step", spy)
        env.step(np.full(env.action_dim, 0.1))
        assert len(recorded) == env.decimation * env.physics_substeps
        assert env.physics_substeps > 1

        blocks = [recorded[i * env.physics_substeps : (i + 1) * env.physics_substeps] for i in range(env.decimation)]
        for block in blocks:
            for ctrl in block[1:]:
                np.testing.assert_array_equal(ctrl, block[0])
        # Sanity: the held ctrl does evolve across physics-step boundaries.
        assert any(not np.array_equal(blocks[0][0], block[0]) for block in blocks[1:])


class TestResolveActionArrays:
    @pytest.mark.parametrize(
        "ref",
        [
            "isaaclab.envs.mdp.actions.joint_actions:JointVelocityAction",
            # The Cfg-suffixed ref additionally exercises the class_name suffix strip.
            "isaaclab.envs.mdp.actions.joint_actions_to_limits:EMAJointPositionToLimitsActionCfg",
        ],
    )
    def test_non_position_action_classes_rejected(self, ref):
        from lab2mj.env import resolve_action_arrays

        ir = {"func": ref, "joint_names_expr": [".*"]}
        with pytest.raises(ValueError, match="JointPositionAction"):
            resolve_action_arrays(ir, ["a", "b"], np.zeros(2))


class TestActionProcessor:
    def _processor(self, clip=None):
        return ActionProcessor(
            num_joints=4,
            joint_ids_isaac=np.array([0, 1, 2, 3]),
            scale=np.array([0.5, 0.5, 0.25, 0.25]),
            offset=np.array([0.1, -0.2, 0.3, 0.0]),
            clip=clip,
            default_joint_pos_isaac=np.array([0.1, -0.2, 0.3, 0.0]),
        )

    def test_affine_transform(self):
        proc = self._processor()
        raw = np.array([1.0, -1.0, 2.0, 0.0])
        expected = raw * np.array([0.5, 0.5, 0.25, 0.25]) + np.array([0.1, -0.2, 0.3, 0.0])
        np.testing.assert_allclose(proc.processed(raw), expected)
        np.testing.assert_allclose(proc.targets_isaac(raw), expected)

    def test_subset_action_holds_default_targets(self):
        proc = ActionProcessor(
            num_joints=3,
            joint_ids_isaac=np.array([0, 2]),
            scale=np.array([1.0, 1.0]),
            offset=np.array([0.0, 0.0]),
            clip=None,
            default_joint_pos_isaac=np.array([0.5, -0.7, 0.9]),
        )
        targets = proc.targets_isaac(np.array([0.2, -0.3]))
        np.testing.assert_allclose(targets, [0.2, -0.7, -0.3])

    def test_from_manifest(self):
        action = {
            "joint_ids_isaac": [0, 1],
            "scale": [0.2, 0.2],
            "offset": [0.0, 0.1],
            "clip": [[-1.0, 1.0], [-1.0, 1.0]],
        }
        proc = ActionProcessor.from_manifest(action, np.zeros(2))
        np.testing.assert_allclose(proc.processed(np.array([2.0, -20.0])), [0.4, -1.0])


class TestFreeJointQvel:
    """The dump-init qvel conversion, verified against MuJoCo itself."""

    _XML = """
    <mujoco>
      <worldbody>
        <body name="base" pos="0 0 1">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.1" mass="0"/>
          <inertial pos="0.2 0.05 -0.1" mass="3.0" diaginertia="0.02 0.02 0.02"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def test_matches_mujoco_object_velocity(self):
        rng = np.random.default_rng(7)
        quat_wxyz = rng.normal(size=4)
        quat_wxyz /= np.linalg.norm(quat_wxyz)
        root_link_lin_vel_w = np.array([0.4, -0.2, 0.1])
        root_ang_vel_w = np.array([0.3, 0.5, -0.7])

        model = mujoco.MjModel.from_xml_string(self._XML)
        data = mujoco.MjData(model)
        data.qpos[0:3] = (0.0, 0.0, 1.0)
        data.qpos[3:7] = quat_wxyz
        data.qvel[0:6] = free_joint_qvel(quat_wxyz, root_link_lin_vel_w, root_ang_vel_w)
        mujoco.mj_forward(model, data)

        # mj_objectVelocity(XBODY, flg_local=0): world-frame [ang, lin] at the body origin.
        vel_buf = np.zeros(6)
        body_id = model.body("base").id
        mujoco.mj_objectVelocity(model, data, int(mujoco.mjtObj.mjOBJ_XBODY), body_id, vel_buf, 0)
        np.testing.assert_allclose(vel_buf[0:3], root_ang_vel_w, atol=1e-10)
        np.testing.assert_allclose(vel_buf[3:6], root_link_lin_vel_w, atol=1e-10)


@needs_go1_bundle
class TestActuatorNetWarmStartOnReset:
    """Warm-start wiring on the dump-init path (real go1 bundle: actuator_net_mlp)."""

    def _reset_from_dump(self, env, dump, with_last_action: bool):
        return env.reset_from_state(
            root_pos_w=dump["init_root_pos_w"],
            root_quat_wxyz=dump["init_root_quat_w"],
            root_link_lin_vel_w=dump["init_root_link_lin_vel_w"],
            root_ang_vel_w=dump["init_root_ang_vel_w"],
            joint_pos_isaac=dump["init_joint_pos"],
            joint_vel_isaac=dump["init_joint_vel"],
            last_action_raw=dump["init_last_action"] if with_last_action else None,
        )

    def test_dump_init_warm_starts_and_plain_reset_is_cold(self):
        from lab2mj.env import MjEnv

        env = MjEnv(GO1_BUNDLE, strict=True, seed=0, command=[0.5, 0.0, 0.0])
        dump = np.load(GO1_DUMP)
        self._reset_from_dump(env, dump, with_last_action=True)
        group = next(g for g in env.actuators.groups if g.has_net)
        # Every history row carries the t0-consistent input: pos error against the
        # targets the recorded pre-recording action produces, velocity from the dump.
        assert env._joint_action is not None
        assert group._pos_error_history is not None and group._vel_history is not None
        q_des = env._joint_action.targets_isaac(np.asarray(dump["init_last_action"], dtype=np.float64))
        pos_error = (q_des - np.asarray(dump["init_joint_pos"], dtype=np.float64))[group.joint_ids]
        vel = np.asarray(dump["init_joint_vel"], dtype=np.float64)[group.joint_ids]
        for row in range(group._pos_error_history.shape[0]):
            np.testing.assert_allclose(group._pos_error_history[row], pos_error.astype(np.float32), rtol=0, atol=0)
            np.testing.assert_allclose(group._vel_history[row], vel.astype(np.float32), rtol=0, atol=0)
        assert np.any(group._pos_error_history != 0.0)

        env.reset()
        np.testing.assert_array_equal(group._pos_error_history, 0.0)
        np.testing.assert_array_equal(group._vel_history, 0.0)

    def test_dump_init_without_last_action_stays_cold(self):
        from lab2mj.env import MjEnv

        env = MjEnv(GO1_BUNDLE, strict=True, seed=0, command=[0.5, 0.0, 0.0])
        dump = np.load(GO1_DUMP)
        self._reset_from_dump(env, dump, with_last_action=False)
        group = next(g for g in env.actuators.groups if g.has_net)
        assert group._pos_error_history is not None and group._vel_history is not None
        np.testing.assert_array_equal(group._pos_error_history, 0.0)
        np.testing.assert_array_equal(group._vel_history, 0.0)


@needs_go1_bundle
class TestExactInertiaStamp:
    def test_manifest_exact_inertia_is_stamped(self):
        import json
        import shutil
        import tempfile

        from lab2mj.env import MjEnv

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(GO1_BUNDLE, bundle)
            manifest_path = bundle / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            # A tensor MuJoCo's compiler would reject (I0 > I1 + I2) must survive
            # into the loaded model via the load-time stamp.
            exact = [0.5, 0.01, 0.005]
            manifest["robot"]["body_inertia_exact"] = {"FL_calf": exact}
            manifest_path.write_text(json.dumps(manifest))
            env = MjEnv(bundle, strict=True, seed=0)
            np.testing.assert_allclose(env.model.body_inertia[env.model.body("FL_calf").id], exact)
