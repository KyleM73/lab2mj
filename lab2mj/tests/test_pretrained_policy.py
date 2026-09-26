"""PreTrainedPolicyAction runtime tests (navigation tasks).

The action-path tests run on a hand-built minimal bundle (2-joint robot, no
external assets) with a scripted low-level "policy" that records the
observations it receives, so the held-command slot, the low-level cadence, and
the target transform are asserted exactly. The convert round-trip test uses
the nav-shaped ANYmal-C fixture against the locally cached ANYmal-D USD and
actuator net (joint structure and names are identical) and skips without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lab2mj import bundle as bundle_mod
from lab2mj.env import (
    HELD_ACTION_COMMAND,
    PRE_TRAINED_POLICY_ACTION_DIM,
    PreTrainedPolicyRuntime,
    remap_low_level_obs_group,
    resolve_action_arrays,
)
from lab2mj.ir import NoiseIR, ObsGroupIR, ObsTermIR

from .shared import FIXTURES, HAS_PXR, HAS_TORCH, REPO_ROOT, minimal_manifest

ANYMAL_USD = REPO_ROOT / "data" / "usd_cache" / "ANYmal-D" / "anymal_d.usd"
ANYDRIVE_NET = REPO_ROOT / "data" / "actuator_nets" / "anydrive_3_lstm_jit.pt"

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
needs_convert_assets = pytest.mark.skipif(
    not (HAS_PXR and HAS_TORCH and ANYMAL_USD.exists() and ANYDRIVE_NET.exists()),
    reason="requires pxr, torch, data/usd_cache/ANYmal-D/anymal_d.usd, and the cached anydrive net",
)

# Minimal 2-joint floating-base robot; timestep/gravity must match the manifest.
_SCENE_XML = """
<mujoco>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom name="base_geom" type="box" size="0.1 0.1 0.05" mass="1.0"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="j1" axis="0 1 0"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1.0"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="j2" axis="0 1 0"/>
          <geom name="link2_geom" type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1.0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1"/>
    <motor joint="j2"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0.5 1 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""

JOINTS = ["j1", "j2"]
LL_DIM = 2
LL_DECIMATION = 2
DECIMATION = 4
LL_SCALE = np.array([0.25, 0.5])
LL_OFFSET = np.array([0.1, -0.1])

# Low-level obs layout (dims 3, 3, 3, 3, 2, 2, 2): the held-command and
# low-level-action slices asserted below.
LL_COMMAND_SLICE = slice(9, 12)
LL_ACTIONS_SLICE = slice(16, 18)


def _term(name: str, func: str, params: dict | None = None, noise: NoiseIR | None = None) -> ObsTermIR:
    return ObsTermIR(name=name, func=func, params=params or {}, noise=noise)


def _low_level_obs_group() -> ObsGroupIR:
    # Shaped like a dumped locomotion policy group after PreTrainedPolicyAction's
    # remap: the two remapped terms carry the recorded lambda strings.
    return ObsGroupIR(
        name="ll_policy",
        enable_corruption=True,
        terms=[
            _term(
                "base_lin_vel",
                "isaaclab.envs.mdp.observations:base_lin_vel",
                noise=NoiseIR(kind="uniform", params={"n_min": -0.1, "n_max": 0.1}),
            ),
            _term("base_ang_vel", "isaaclab.envs.mdp.observations:base_ang_vel"),
            _term("projected_gravity", "isaaclab.envs.mdp.observations:projected_gravity"),
            _term("velocity_commands", "lambda dummy_env: self._raw_actions"),
            _term("joint_pos", "isaaclab.envs.mdp.observations:joint_pos_rel"),
            _term("joint_vel", "isaaclab.envs.mdp.observations:joint_vel_rel"),
            _term("actions", "lambda dummy_env: last_action()"),
        ],
    )


def _policy_obs_group() -> ObsGroupIR:
    return ObsGroupIR(
        name="policy",
        terms=[
            _term("base_lin_vel", "isaaclab.envs.mdp.observations:base_lin_vel"),
            _term("projected_gravity", "isaaclab.envs.mdp.observations:projected_gravity"),
            _term("actions", "isaaclab.envs.mdp.observations:last_action"),
        ],
    )


def _manifest() -> dict:
    low_level_ir = {
        "func": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
        "joint_names_expr": [".*"],
        "scale": 0.5,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": None,
        "preserve_order": False,
    }
    action = {
        "ir": {
            "func": ("isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action:PreTrainedPolicyAction"),
            "policy_path": "https://example.com/policy.pt",
            "low_level_decimation": LL_DECIMATION,
            "low_level_action": low_level_ir,
            "low_level_obs": _low_level_obs_group().to_dict(),
            "policy_bundle_path": "assets/policies/policy.pt",
        },
        "dim": PRE_TRAINED_POLICY_ACTION_DIM,
        "low_level": {
            "ir": low_level_ir,
            "dim": LL_DIM,
            "joint_ids_isaac": [0, 1],
            "scale": LL_SCALE.tolist(),
            "offset": LL_OFFSET.tolist(),
            "clip": None,
        },
        "low_level_decimation": LL_DECIMATION,
    }
    return minimal_manifest(
        robot={
            "isaac_joint_order": JOINTS,
            "mj_joint_order": JOINTS,
            "isaac_to_mj": [0, 1],
            "mj_to_isaac": [0, 1],
            "isaac_body_names": ["base", "link1", "link2"],
            "body_map": {"base": "base", "link1": "link1", "link2": "link2"},
            "geom_map": {"base": ["base_geom"], "link1": ["link1_geom"], "link2": ["link2_geom"]},
            "default_qpos": [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "default_qvel": [0.0] * 8,
            "default_joint_pos_isaac": [0.0, 0.0],
            "default_joint_vel_isaac": [0.0, 0.0],
            "joint_pos_limits_isaac": [[-3.14, 3.14], [-3.14, 3.14]],
            "joint_vel_limits_isaac": [30.0, 30.0],
            "soft_joint_pos_limit_factor": 1.0,
            "total_mass": 3.0,
        },
        timing={"decimation": DECIMATION},
        obs={"groups": {"policy": _policy_obs_group().to_dict()}},
        action=action,
        actuators=[
            {
                "name": "all",
                "joint_names_expr": [".*"],
                "model": "ideal_pd",
                "stiffness": 20.0,
                "damping": 0.5,
                "effort_limit": 100.0,
            }
        ],
        commands=[],
        terminations=[{"name": "time_out", "func": "isaaclab.envs.mdp.terminations:time_out", "time_out": True}],
        terrain=None,
        init={"seed": 0},
        sources={"env_yaml_sha256": "0" * 64, "usd_sha256": "0" * 64},
    )


def _write_bundle(bundle_dir: Path, *, with_policy_file: bool = True) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "scene.xml").write_text(_SCENE_XML)
    (bundle_dir / "robot.xml").write_text(_SCENE_XML)
    if with_policy_file:
        policy = bundle_dir / "assets" / "policies" / "policy.pt"
        policy.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder bytes: the tests below install a scripted policy object
        # directly, so the file only has to exist.
        policy.write_bytes(b"placeholder")
    bundle_mod.write_manifest(bundle_dir, _manifest())
    return bundle_dir


class SpyPolicy:
    """Scripted low-level policy: records received observations, returns a
    call-count-dependent constant so consecutive fires are distinguishable."""

    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def output(self, call_index: int) -> np.ndarray:
        return np.array([0.1 * (call_index + 1), -0.1 * (call_index + 1)])

    def __call__(self, obs_t):
        import torch

        self.received.append(obs_t[0].numpy().copy())
        return torch.from_numpy(self.output(len(self.received) - 1).reshape(1, LL_DIM).astype(np.float32))


@pytest.fixture()
def env_and_spy(tmp_path):
    from lab2mj.env import MjEnv

    env = MjEnv(_write_bundle(tmp_path / "bundle"), strict=True, seed=0)
    assert isinstance(env.action, PreTrainedPolicyRuntime)
    spy = SpyPolicy()
    env.action._policy = spy
    env.reset()
    return env, spy


class TestRemapLowLevelObsGroup:
    def test_remap_rewrites_the_two_named_terms(self):
        group = remap_low_level_obs_group(_low_level_obs_group())
        by_name = {t.name: t for t in group.terms}
        assert by_name["actions"].func.endswith("last_action")
        assert by_name["actions"].params == {}
        assert by_name["velocity_commands"].func.endswith("generated_commands")
        assert by_name["velocity_commands"].params == {"command_name": HELD_ACTION_COMMAND}
        # Other terms and per-term processing survive untouched.
        assert by_name["base_lin_vel"].noise is not None
        assert by_name["joint_pos"].func.endswith("joint_pos_rel")

    @pytest.mark.parametrize("missing", ["actions", "velocity_commands"])
    def test_missing_required_term_raises(self, missing):
        group = _low_level_obs_group()
        group.terms = [t for t in group.terms if t.name != missing]
        with pytest.raises(ValueError, match=missing):
            remap_low_level_obs_group(group)


class TestResolveActionArrays:
    def test_wrapper_resolution(self):
        manifest = _manifest()
        resolved = resolve_action_arrays(manifest["action"]["ir"], JOINTS, np.zeros(2))
        assert resolved["dim"] == PRE_TRAINED_POLICY_ACTION_DIM
        assert resolved["low_level_decimation"] == LL_DECIMATION
        low = resolved["low_level"]
        assert low["dim"] == LL_DIM
        np.testing.assert_allclose(low["scale"], [0.5, 0.5])

    def test_missing_embedded_cfg_raises(self):
        ir = dict(_manifest()["action"]["ir"])
        ir["low_level_action"] = None
        with pytest.raises(ValueError, match="low_level_action"):
            resolve_action_arrays(ir, JOINTS, np.zeros(2))


class TestManifestValidation:
    def test_valid_wrapper_manifest_passes(self):
        bundle_mod.validate_manifest(bundle_mod.jsonable(_manifest()))

    def test_missing_low_level_arrays_rejected(self):
        manifest = bundle_mod.jsonable(_manifest())
        del manifest["action"]["low_level"]["scale"]
        with pytest.raises(ValueError, match="low_level"):
            bundle_mod.validate_manifest(manifest)

    def test_missing_policy_bundle_path_rejected(self):
        manifest = bundle_mod.jsonable(_manifest())
        manifest["action"]["ir"]["policy_bundle_path"] = None
        with pytest.raises(ValueError, match="policy_bundle_path"):
            bundle_mod.validate_manifest(manifest)


@needs_torch
class TestActionPath:
    def test_missing_policy_file_fails_fast(self, tmp_path):
        from lab2mj.env import MjEnv

        with pytest.raises(FileNotFoundError, match="low-level policy"):
            MjEnv(_write_bundle(tmp_path / "bundle", with_policy_file=False), strict=True, seed=0)

    def test_command_slot_carries_held_high_level_action(self, env_and_spy):
        env, spy = env_and_spy
        high = np.array([0.4, -0.2, 0.1])
        env.step(high)
        assert len(spy.received) == DECIMATION // LL_DECIMATION
        for obs in spy.received:
            np.testing.assert_allclose(obs[LL_COMMAND_SLICE], high, atol=1e-6)
        # A new high-level action replaces the held one on the next policy step.
        high2 = np.array([-0.3, 0.5, 0.0])
        env.step(high2)
        for obs in spy.received[2:]:
            np.testing.assert_allclose(obs[LL_COMMAND_SLICE], high2, atol=1e-6)

    def test_low_level_cadence_and_target_hold(self, env_and_spy):
        env, spy = env_and_spy
        seen_targets: list[np.ndarray] = []
        original = env.actuators.step_delay_buffers

        def recording(q_target):
            seen_targets.append(np.asarray(q_target, dtype=np.float64).copy())
            return original(q_target)

        env.actuators.step_delay_buffers = recording
        env.step(np.array([0.4, -0.2, 0.1]))
        assert len(spy.received) == 2  # fires at physics steps 0 and LL_DECIMATION
        assert len(seen_targets) == DECIMATION
        expected_fire0 = LL_SCALE * spy.output(0) + LL_OFFSET
        expected_fire1 = LL_SCALE * spy.output(1) + LL_OFFSET
        np.testing.assert_allclose(seen_targets[0], expected_fire0, atol=1e-12)
        np.testing.assert_allclose(seen_targets[1], expected_fire0, atol=1e-12)  # held between fires
        np.testing.assert_allclose(seen_targets[2], expected_fire1, atol=1e-12)
        np.testing.assert_allclose(seen_targets[3], expected_fire1, atol=1e-12)

    def test_low_level_actions_slot_observes_previous_raw_output(self, env_and_spy):
        env, spy = env_and_spy
        env.step(np.zeros(3))
        env.step(np.zeros(3))
        # Every fire of the episode's first policy step observes zeros in the
        # low-level actions slot (Isaac's episode_length_buf == 0 remap; the
        # buffer increments only after the full decimation loop). From the
        # second policy step on, each fire observes the previous fire's RAW
        # policy output (pre scale/offset).
        np.testing.assert_allclose(spy.received[0][LL_ACTIONS_SLICE], 0.0, atol=1e-12)
        np.testing.assert_allclose(spy.received[1][LL_ACTIONS_SLICE], 0.0, atol=1e-12)
        np.testing.assert_allclose(spy.received[2][LL_ACTIONS_SLICE], spy.output(1), atol=1e-6)
        np.testing.assert_allclose(spy.received[3][LL_ACTIONS_SLICE], spy.output(2), atol=1e-6)

    def test_reset_zeroes_low_level_actions_for_whole_first_policy_step(self, env_and_spy):
        env, spy = env_and_spy
        env.step(np.zeros(3))
        env.reset()
        env.step(np.zeros(3))
        env.step(np.zeros(3))
        # Fires 2 and 3 span the first policy step after the reset: both observe
        # zeros even though fires 0-2 produced nonzero raw outputs. Normal
        # previous-raw-output semantics resume on the next policy step.
        np.testing.assert_allclose(spy.received[2][LL_ACTIONS_SLICE], 0.0, atol=1e-12)
        np.testing.assert_allclose(spy.received[3][LL_ACTIONS_SLICE], 0.0, atol=1e-12)
        np.testing.assert_allclose(spy.received[4][LL_ACTIONS_SLICE], spy.output(3), atol=1e-6)

    def test_high_level_obs_observes_high_level_action(self, env_and_spy):
        env, _ = env_and_spy
        high = np.array([0.4, -0.2, 0.1])
        obs, _, _, _ = env.step(high)
        np.testing.assert_allclose(obs[6:9], high, atol=1e-6)  # policy group last_action slice
        assert np.all(np.isfinite(obs))

    def test_strict_determinism(self, tmp_path):
        from lab2mj.env import MjEnv

        traces = []
        for k in range(2):
            env = MjEnv(_write_bundle(tmp_path / f"bundle{k}"), strict=True, seed=0)
            assert isinstance(env.action, PreTrainedPolicyRuntime)
            env.action._policy = SpyPolicy()
            env.reset()
            trace = [env.step(np.array([0.2, 0.0, -0.1]))[0] for _ in range(3)]
            traces.append(np.stack(trace))
        assert np.all(np.isfinite(traces[0]))
        np.testing.assert_array_equal(traces[0], traces[1])


@needs_convert_assets
class TestConvertRoundTrip:
    """Full convert -> bundle -> MjEnv round trip on the nav fixture.

    The fixture's policy URL is rewritten to a local scripted policy so the
    conversion runs offline; the ANYmal-C robot cfg is converted against the
    locally cached ANYmal-D USD (identical joint names and structure).
    """

    POLICY_URL = (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/"
        "Policies/ANYmal-C/Blind/policy.pt"
    )

    @pytest.fixture(scope="class")
    def nav_bundle(self, tmp_path_factory):
        import torch

        from lab2mj.convert import convert_run

        cfg_dir = tmp_path_factory.mktemp("nav_cfg")

        class TinyPolicy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(0)
                self.lin = torch.nn.Linear(48, 12)

            def forward(self, x):
                return 0.1 * torch.tanh(self.lin(x))

        policy_path = cfg_dir / "nav_low_level_policy.pt"
        torch.jit.save(torch.jit.script(TinyPolicy()), str(policy_path))

        text = (FIXTURES / "anymal_c_nav_env_like.yaml").read_text()
        assert text.count(self.POLICY_URL) == 1
        yaml_path = cfg_dir / "nav_env.yaml"
        yaml_path.write_text(text.replace(self.POLICY_URL, str(policy_path)))
        return convert_run(yaml_path, usd=ANYMAL_USD, out=tmp_path_factory.mktemp("nav_bundle"))

    def test_policy_file_copied_and_manifest_recorded(self, nav_bundle):
        manifest = bundle_mod.read_manifest(nav_bundle)
        action = manifest["action"]
        assert action["dim"] == 3
        assert action["low_level_decimation"] == 4
        assert action["low_level"]["dim"] == 12
        assert action["low_level_obs_dim"] == 48
        rel = action["ir"]["policy_bundle_path"]
        assert rel == "assets/policies/nav_low_level_policy.pt"
        assert (nav_bundle / rel).is_file()
        # The actuator net rides along like any actuator-net bundle.
        (net,) = [a["network_bundle_path"] for a in manifest["actuators"]]
        assert (nav_bundle / net).is_file()

    def test_high_level_obs_layout(self, nav_bundle):
        manifest = bundle_mod.read_manifest(nav_bundle)
        assert manifest["obs"]["obs_dims"]["policy"] == 10  # 3 + 3 + 4 (pose command)
        names = [entry["name"] for entry in manifest["obs"]["layouts"]["policy"]]
        assert names == ["base_lin_vel", "projected_gravity", "pose_command"]

    def test_strict_rollout_runs_the_low_level_policy(self, nav_bundle):
        from lab2mj.env import MjEnv

        env = MjEnv(nav_bundle, strict=True, seed=0)
        assert isinstance(env.action, PreTrainedPolicyRuntime)
        assert env.action_dim == 3
        obs = env.reset()
        assert obs.shape == (10,)

        import torch

        class CountingPolicy:
            def __init__(self, inner) -> None:
                self.inner = inner
                self.calls = 0

            def __call__(self, obs_t):
                self.calls += 1
                assert obs_t.shape == (1, 48)
                return self.inner(obs_t)

        counting = CountingPolicy(torch.jit.load(str(env.action._policy_path), map_location="cpu").eval())
        env.action._policy = counting
        for _ in range(3):
            obs, _, _, _ = env.step(np.array([0.5, 0.0, 0.0]))
            assert np.all(np.isfinite(obs))
        assert counting.calls == 3 * (env.decimation // env.action.low_level_decimation)
        assert env.data.time == pytest.approx(3 * env.policy_dt)
