"""Tests for the height-scan provider (`lab2mj.heightscan`)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from lab2mj.env_yaml import parse_env_dict
from lab2mj.heightscan import (
    HeightScanProvider,
    grid_pattern,
    height_scan_extras_dims,
    pattern_num_rays,
)
from lab2mj.ir import EnvIR, HeightScannerIR
from lab2mj.obs import ObsContext, ObsPipeline

from .shared import G1_USD, HAS_PXR, HAS_TORCH, g1_rough_env_dict, needs_g1  # noqa: F401

HEIGHT_SCAN_FUNC = "isaaclab.envs.mdp.observations:height_scan"

# The stock rough-task grid: size [1.6, 1.0] at 0.1 m resolution -> 17 x 11 = 187 rays.
G1_PATTERN = {
    "func": "isaaclab.sensors.ray_caster.patterns.patterns:grid_pattern",
    "resolution": 0.1,
    "size": [1.6, 1.0],
    "direction": [0.0, 0.0, -1.0],
    "ordering": "xy",
}


def make_scanner(**overrides) -> HeightScannerIR:
    fields: dict[str, Any] = dict(
        name="height_scanner",
        prim_path="/World/envs/env_.*/Robot/torso_link",
        attach_body_name="torso_link",
        pattern=dict(G1_PATTERN),
        offset_pos=(0.0, 0.0, 20.0),
        offset_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ray_alignment="yaw",
        max_distance=1.0e6,
        drift_range=(0.0, 0.0),
        ray_cast_drift_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        update_period=0.02,
    )
    fields.update(overrides)
    return HeightScannerIR(**fields)


def flat_height_at(x_w: np.ndarray, y_w: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(x_w, dtype=np.float64))


def plane_height_at(x_w: np.ndarray, y_w: np.ndarray) -> np.ndarray:
    return 0.1 * np.asarray(x_w, dtype=np.float64) + 0.2 * np.asarray(y_w, dtype=np.float64)


"""
Grid pattern.
"""


def _load_isaaclab_grid_pattern():
    """Load isaaclab's patterns.py by file path (skips the heavy package __init__)."""
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("isaaclab package not locatable")
    path = Path(list(spec.submodule_search_locations)[0]) / "sensors" / "ray_caster" / "patterns" / "patterns.py"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    module_spec = importlib.util.spec_from_file_location("_isaaclab_raycaster_patterns", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.grid_pattern


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch for the isaaclab reference pattern")
@pytest.mark.parametrize("ordering", ["xy", "yx"])
@pytest.mark.parametrize("size,resolution", [((1.6, 1.0), 0.1), ((2.0, 1.5), 0.25), ((0.3, 0.3), 0.1)])
def test_grid_pattern_matches_isaaclab_source(ordering, size, resolution):
    isaac_grid_pattern = _load_isaaclab_grid_pattern()
    cfg = SimpleNamespace(resolution=resolution, size=list(size), direction=(0.0, 0.0, -1.0), ordering=ordering)
    starts_ref, dirs_ref = isaac_grid_pattern(cfg, "cpu")
    starts, dirs = grid_pattern(
        {"resolution": resolution, "size": list(size), "direction": (0.0, 0.0, -1.0), "ordering": ordering}
    )
    assert starts.shape == tuple(starts_ref.shape)
    # isaaclab builds the pattern in float32; the numpy port in float64.
    np.testing.assert_allclose(starts, starts_ref.numpy(), atol=1e-6)
    np.testing.assert_allclose(dirs, dirs_ref.numpy(), atol=1e-6)


def test_grid_pattern_g1_cfg_hand_derived():
    starts, dirs = grid_pattern(G1_PATTERN)
    assert starts.shape == (187, 3)  # 17 x 11
    np.testing.assert_allclose(dirs, np.tile([0.0, 0.0, -1.0], (187, 1)))
    # ordering "xy": inner loop over x (17 values -0.8..0.8), outer over y (11 values -0.5..0.5).
    np.testing.assert_allclose(starts[0], [-0.8, -0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[1], [-0.7, -0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[16], [0.8, -0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[17], [-0.8, -0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[186], [0.8, 0.5, 0.0], atol=1e-12)
    assert pattern_num_rays(G1_PATTERN) == 187


def test_grid_pattern_yx_ordering():
    starts, _ = grid_pattern({**G1_PATTERN, "ordering": "yx"})
    # "yx": inner loop over y (11 values), outer over x (17 values).
    np.testing.assert_allclose(starts[0], [-0.8, -0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[1], [-0.8, -0.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[10], [-0.8, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(starts[11], [-0.7, -0.5, 0.0], atol=1e-12)


def test_grid_pattern_rejects_bad_cfg():
    with pytest.raises(ValueError, match="Ordering"):
        grid_pattern({**G1_PATTERN, "ordering": "diagonal"})
    with pytest.raises(ValueError, match="Resolution"):
        grid_pattern({**G1_PATTERN, "resolution": 0.0})
    with pytest.raises(NotImplementedError, match="pinhole_camera_pattern"):
        grid_pattern({"func": "isaaclab.sensors.ray_caster.patterns.patterns:pinhole_camera_pattern"})


"""
Provider math.
"""


def test_flat_terrain_identity_pose():
    provider = HeightScanProvider(make_scanner(), flat_height_at)
    assert provider.num_rays == 187
    values = provider.height_scan(np.array([1.0, -2.0, 0.72]), np.array([1.0, 0.0, 0.0, 0.0]))
    # value = sensor_z - hit_z - offset; the 20 m ray-start offset must NOT leak
    # into the sensor height (height_scan reads data.pos_w, not the ray starts).
    np.testing.assert_allclose(values, np.full(187, 0.72 - 0.5), atol=1e-12)


def test_rotated_translated_base_hand_computed():
    provider = HeightScanProvider(make_scanner(), plane_height_at)
    yaw, roll = 0.7, 0.3
    # q = q_yaw (x) q_roll: heading (and yaw_quat's yaw) is exactly `yaw`, so the
    # roll must not tilt the scan grid under ray_alignment="yaw".
    q_yaw = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    q_roll = np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0])
    w1, x1, y1, z1 = q_yaw
    w2, x2, y2, z2 = q_roll
    quat = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    pos_w = np.array([2.0, -1.0, 0.9])
    values = provider.height_scan(pos_w, quat)

    starts, _ = grid_pattern(G1_PATTERN)
    c, s = math.cos(yaw), math.sin(yaw)
    x_w = c * starts[:, 0] - s * starts[:, 1] + pos_w[0]
    y_w = s * starts[:, 0] + c * starts[:, 1] + pos_w[1]
    expected = pos_w[2] - (0.1 * x_w + 0.2 * y_w) - 0.5
    np.testing.assert_allclose(values, expected, atol=1e-9)


def test_missed_rays_are_minus_inf():
    def hole_height_at(x_w, y_w):
        heights = np.zeros_like(np.asarray(x_w, dtype=np.float64))
        return np.where(np.asarray(x_w) > 0.0, np.nan, heights)

    provider = HeightScanProvider(make_scanner(), hole_height_at)
    values = provider.height_scan(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    starts, _ = grid_pattern(G1_PATTERN)
    assert np.all(values[starts[:, 0] > 0.0] == -np.inf)
    np.testing.assert_allclose(values[starts[:, 0] <= 0.0], -0.5, atol=1e-12)


def test_surface_above_ray_start_misses():
    # Ray starts sit at attach_z + 20; a surface above them cannot be hit.
    def tall_height_at(x_w, y_w):
        return np.full_like(np.asarray(x_w, dtype=np.float64), 25.0)

    provider = HeightScanProvider(make_scanner(), tall_height_at)
    values = provider.height_scan(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(values == -np.inf)


def test_max_distance_limits_hits():
    provider = HeightScanProvider(make_scanner(max_distance=5.0), flat_height_at)
    # Start z = 1 + 20 = 21 > max_distance above the ground: every ray misses.
    values = provider.height_scan(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(values == -np.inf)
    provider = HeightScanProvider(make_scanner(max_distance=25.0), flat_height_at)
    values = provider.height_scan(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(values, 0.5, atol=1e-12)


def test_drift_resampling_and_strict_reset():
    scanner = make_scanner(
        drift_range=(0.01, 0.02),
        ray_cast_drift_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.03, 0.03)},
    )
    provider = HeightScanProvider(scanner, flat_height_at)
    provider.reset(np.random.default_rng(7))
    values = provider.height_scan(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    # Mirror the provider's seeded draws: 3 world-drift components, then the
    # per-axis ray_cast_drift vector.
    rng = np.random.default_rng(7)
    drift_w = rng.uniform(0.01, 0.02, size=3)
    rng.uniform(np.array([0.0, 0.0, 0.03]), np.array([0.0, 0.0, 0.03]))
    # World drift shifts the sensor height; ray_cast_drift z shifts the hit points.
    np.testing.assert_allclose(values, (1.0 + drift_w[2]) - 0.03 - 0.5, atol=1e-12)

    # Strict reset pins both drifts back to zero.
    provider.reset(np.random.default_rng(7), strict=True)
    values = provider.height_scan(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(values, 0.5, atol=1e-12)


def test_ray_cast_drift_xy_rotates_with_yaw():
    scanner = make_scanner(ray_cast_drift_range={"x": (0.5, 0.5), "y": (0.0, 0.0), "z": (0.0, 0.0)})
    provider = HeightScanProvider(scanner, plane_height_at)
    provider.reset(np.random.default_rng(0))
    yaw = math.pi / 2
    quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    values = provider.height_scan(np.zeros(3), quat)
    starts, _ = grid_pattern(G1_PATTERN)
    # At yaw = pi/2 the +x projection-frame drift points along world +y.
    x_w = -starts[:, 1]
    y_w = starts[:, 0] + 0.5
    np.testing.assert_allclose(values, -(0.1 * x_w + 0.2 * y_w) - 0.5, atol=1e-9)


def test_world_alignment_ignores_orientation():
    provider = HeightScanProvider(make_scanner(ray_alignment="world"), plane_height_at)
    yaw = 1.1
    quat = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    values = provider.height_scan(np.array([1.0, 2.0, 0.8]), quat)
    starts, _ = grid_pattern(G1_PATTERN)
    expected = 0.8 - (0.1 * (starts[:, 0] + 1.0) + 0.2 * (starts[:, 1] + 2.0)) - 0.5
    np.testing.assert_allclose(values, expected, atol=1e-9)


def test_unsupported_configurations_raise():
    with pytest.raises(NotImplementedError, match="ray_alignment"):
        HeightScanProvider(make_scanner(ray_alignment="base"), flat_height_at)
    with pytest.raises(NotImplementedError, match="straight down"):
        HeightScanProvider(make_scanner(pattern={**G1_PATTERN, "direction": [0.0, 0.1, -1.0]}), flat_height_at)
    # An offset rotation that tilts the (straight-down) rays is equally unservable.
    tilt = np.array([math.cos(0.2), math.sin(0.2), 0.0, 0.0])
    with pytest.raises(NotImplementedError, match="straight down"):
        HeightScanProvider(make_scanner(offset_quat_wxyz=tuple(tilt)), flat_height_at)
    not_a_lookup: Any = None
    with pytest.raises(TypeError, match="callable"):
        HeightScanProvider(make_scanner(), not_a_lookup)


"""
Fixture parsing + obs pipeline integration.
"""


@pytest.fixture(scope="module")
def rough_ir() -> EnvIR:
    return parse_env_dict(g1_rough_env_dict())


def test_fixture_scanner_ir(rough_ir: EnvIR):
    (scanner,) = rough_ir.height_scanners
    assert scanner.name == "height_scanner"
    assert scanner.attach_body_name == "torso_link"
    assert scanner.ray_alignment == "yaw"
    assert scanner.offset_pos == (0.0, 0.0, 20.0)
    assert scanner.offset_quat_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert scanner.update_period == 0.02
    assert scanner.mesh_prim_paths == ["/World/ground"]
    assert scanner.drift_range == (0.0, 0.0)
    assert scanner.ray_cast_drift_range["z"] == (0.0, 0.0)
    assert pattern_num_rays(scanner.pattern) == 187


def test_fixture_round_trip(rough_ir: EnvIR):
    as_dict = rough_ir.to_dict()
    rebuilt = EnvIR.from_dict(as_dict)
    assert rebuilt.to_dict() == as_dict
    assert rebuilt.height_scanners[0].pattern["ordering"] == "xy"


def test_attach_yaw_only_overrides_ray_alignment():
    from lab2mj.env_yaml import _parse_height_scanners

    scene = {
        "scanner": {
            "class_type": "isaaclab.sensors.ray_caster.ray_caster:RayCaster",
            "prim_path": "/World/envs/env_.*/Robot/base",
            "ray_alignment": "base",
            "attach_yaw_only": True,
            "pattern_cfg": dict(G1_PATTERN),
        }
    }
    (scanner,) = _parse_height_scanners(scene)
    assert scanner.ray_alignment == "yaw"
    assert scanner.attach_body_name == "base"


def test_extras_dims_helper(rough_ir: EnvIR):
    policy = rough_ir.obs_group("policy")
    assert height_scan_extras_dims(policy, rough_ir.height_scanners) == {"height_scan": 187}
    # A term referencing an unrecorded scanner is left out (caller decides).
    assert height_scan_extras_dims(policy, []) == {}


def test_obs_pipeline_dim_and_clip(rough_ir: EnvIR):
    policy = rough_ir.obs_group("policy")
    pipeline = ObsPipeline(
        policy,
        num_joints=37,
        action_dim=37,
        command_dims={"base_velocity": 3},
        extras_dims=height_scan_extras_dims(policy, rough_ir.height_scanners),
        enable_noise=False,
    )
    assert pipeline.obs_dim == 3 + 3 + 3 + 3 + 37 + 37 + 37 + 187
    names_dims = [(name, dim) for name, _, dim in pipeline.layout()]
    assert names_dims[-1] == ("height_scan", 187)

    rng = np.random.default_rng(0)
    scan = np.linspace(-4.0, 4.0, 187)
    scan[0] = -np.inf  # missed ray
    ctx = ObsContext(
        root_lin_vel_b=rng.normal(size=3),
        root_ang_vel_b=rng.normal(size=3),
        projected_gravity_b=np.array([0.0, 0.0, -1.0]),
        joint_pos_isaac=rng.normal(size=37),
        joint_vel_isaac=rng.normal(size=37),
        default_joint_pos_isaac=np.zeros(37),
        default_joint_vel_isaac=np.zeros(37),
        last_action_raw=np.zeros(37),
        commands={"base_velocity": np.zeros(3)},
        extras={"height_scan": scan},
    )
    obs = pipeline.compute(ctx)
    (_, sl, _) = pipeline.layout()[-1]
    np.testing.assert_array_equal(obs[sl], np.clip(scan, -1.0, 1.0).astype(np.float32))
    assert obs[sl.start] == -1.0


def test_convert_attach_body_check(rough_ir: EnvIR):
    from lab2mj.convert import _check_height_scanner_bodies

    _check_height_scanner_bodies(rough_ir, {"torso_link": "torso_link"})
    with pytest.raises(ValueError, match="not an articulation body"):
        _check_height_scanner_bodies(rough_ir, {"base": "base"})
    with pytest.raises(ValueError, match="welded"):
        _check_height_scanner_bodies(rough_ir, {"torso_link": "pelvis"})
    # Scanners no obs term consumes are recorded but never checked.
    unused = EnvIR.from_dict(rough_ir.to_dict())
    unused.obs_groups = []
    _check_height_scanner_bodies(unused, {})


"""
End-to-end: convert the rough fixture and run the MuJoCo env (needs G1 USD).
"""


@pytest.fixture(scope="module")
def rough_bundle(tmp_path_factory):
    from lab2mj.convert import convert_run

    # Shrink the terrain grid so the test compiles a handful of tiles instead of
    # the training run's 10 x 20; everything else is the fixture verbatim.
    env = g1_rough_env_dict()
    generator = env["scene"]["terrain"]["terrain_generator"]
    assert (generator["num_rows"], generator["num_cols"]) == (10, 20)
    generator["num_rows"] = generator["num_cols"] = 2
    small_yaml = tmp_path_factory.mktemp("rough_yaml") / "g1_rough_env.yaml"
    small_yaml.write_text(yaml.safe_dump(env, sort_keys=False))
    out = tmp_path_factory.mktemp("rough_bundle")
    return convert_run(small_yaml, usd=G1_USD, out=out)


@needs_g1
class TestRoughBundle:
    def test_manifest_records_scanner_and_layout(self, rough_bundle):
        from lab2mj.bundle import read_manifest

        manifest = read_manifest(rough_bundle)
        (scanner,) = manifest["height_scanners"]
        assert scanner["name"] == "height_scanner"
        assert scanner["attach_body_name"] == "torso_link"
        assert manifest["obs"]["obs_dims"]["policy"] == 310
        layout = {entry["name"]: entry for entry in manifest["obs"]["layouts"]["policy"]}
        assert layout["height_scan"]["dim"] == 187
        assert layout["height_scan"]["stop"] == 310
        assert layout["height_scan"]["clip"] == [-1.0, 1.0]

    def test_strict_obs_matches_terrain_lookup(self, rough_bundle):
        from lab2mj.env import MjEnv

        env = MjEnv(rough_bundle, strict=True, seed=0)
        obs = env.reset()
        assert obs.shape == (310,)
        layout = {e["name"]: (e["start"], e["stop"]) for e in env.manifest["obs"]["layouts"]["policy"]}
        start, stop = layout["height_scan"]

        # Recompute the scan from the attach body pose and the terrain lookup.
        body_id = int(env.model.body("torso_link").id)
        pos_w = env.data.xpos[body_id]
        w, x, y, z = env.data.xquat[body_id]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        starts, _ = grid_pattern(G1_PATTERN)
        c, s = math.cos(yaw), math.sin(yaw)
        x_w = c * starts[:, 0] - s * starts[:, 1] + pos_w[0]
        y_w = s * starts[:, 0] + c * starts[:, 1] + pos_w[1]
        hit_z_w = env._terrain_build().height_at(x_w, y_w)
        expected = np.clip(pos_w[2] - hit_z_w - 0.5, -1.0, 1.0)
        np.testing.assert_allclose(obs[start:stop], expected, atol=1e-6)
