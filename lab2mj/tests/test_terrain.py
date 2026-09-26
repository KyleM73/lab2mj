"""Tests for lab2mj.terrain (offline, no Isaac Sim)."""

import mujoco
import numpy as np
import pytest

from lab2mj.ir import TerrainIR
from lab2mj.terrain import (
    FlatPatchSampler,
    MeasuredSurface,
    MeasuredTerrainData,
    TerrainBuild,
    TileBuild,
    _build_measured,
    _curriculum_terrain_types,
    _Obstacle,
    add_to_spec,
    build_terrain,
    pair_friction,
    plan_measured_exact_window,
    platform_clear_region_w,
    repeated_object_params,
)

from .shared import parse_fixture

_REPEATED_OBJECTS_FUNC = "isaaclab.terrains.trimesh.mesh_terrains:repeated_objects_terrain"


def _cylinder_sub_terrain(
    proportion: float = 1.0,
    num_objects: tuple[int, int] = (5, 25),
    radius: float = 0.3,
    height: tuple[float, float] = (2.0, 2.0),
    flat_patches: dict | None = None,
) -> dict:
    return {
        "function": _REPEATED_OBJECTS_FUNC,
        "proportion": proportion,
        "size": [20.0, 20.0],
        "object_type": "isaaclab.terrains.trimesh.utils:make_cylinder",
        "object_params_start": {
            "num_objects": num_objects[0],
            "height": height[0],
            "radius": radius,
            "max_yx_angle": 0.0,
            "degrees": True,
        },
        "object_params_end": {
            "num_objects": num_objects[1],
            "height": height[1],
            "radius": radius,
            "max_yx_angle": 0.0,
            "degrees": True,
        },
        "abs_height_noise": [0.0, 0.0],
        "rel_height_noise": [1.0, 1.0],
        "platform_width": 1.0,
        "platform_height": 0.0,
        "flat_patch_sampling": flat_patches or {},
    }


def _box_sub_terrain(proportion: float = 1.0, size_xy: tuple[float, float] = (0.3, 0.3)) -> dict:
    return {
        "function": _REPEATED_OBJECTS_FUNC,
        "proportion": proportion,
        "size": [20.0, 20.0],
        "object_type": "isaaclab.terrains.trimesh.utils:make_box",
        "object_params_start": {
            "num_objects": 5,
            "height": 2.0,
            "size": list(size_xy),
            "max_yx_angle": 0.0,
            "degrees": True,
        },
        "object_params_end": {
            "num_objects": 100,
            "height": 2.0,
            "size": list(size_xy),
            "max_yx_angle": 0.0,
            "degrees": True,
        },
        "abs_height_noise": [0.0, 0.0],
        "rel_height_noise": [1.0, 1.0],
        "platform_width": 1.0,
        "platform_height": 0.0,
        "flat_patch_sampling": {},
    }


def _generator_cfg(
    sub_terrains: dict,
    num_rows: int = 2,
    num_cols: int = 2,
    seed: int = 0,
    curriculum: bool = False,
    size: tuple[float, float] = (20.0, 20.0),
) -> dict:
    return {
        "seed": seed,
        "curriculum": curriculum,
        "size": list(size),
        "border_width": 1.0,
        "border_height": 1.0,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "difficulty_range": [0.0, 1.0],
        "sub_terrains": sub_terrains,
    }


def _generator_ir(generator: dict) -> TerrainIR:
    return TerrainIR(
        terrain_type="generator",
        generator=generator,
        physics_material={
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "friction_combine_mode": "multiply",
        },
    )


def _robot_spec(foot_friction: float = 0.8) -> mujoco.MjSpec:
    """Minimal robot-like spec: one collision foot geom, one visual geom."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="base", pos=[0, 0, 1.0])
    body.add_freejoint()
    foot = body.add_geom(name="foot", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0])
    foot.friction = [foot_friction, 0.005, 0.0001]
    foot.contype = 1
    foot.conaffinity = 1
    visual = body.add_geom(name="shell", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0])
    visual.contype = 0
    visual.conaffinity = 0
    return spec


# ---------------------------------------------------------------------------
# plane (both committed fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["spot_velocity_env.yaml", "g1_flat_env.yaml"])
def test_plane_fixture_builds_and_compiles(fixture):
    ir = parse_fixture(fixture)
    assert ir.terrain is not None and ir.terrain.terrain_type == "plane"

    spec = _robot_spec(foot_friction=0.8)
    build = build_terrain(ir.terrain, spec, np.random.default_rng(0))
    model = spec.compile()

    assert build.terrain_type == "plane"
    assert build.ground_geom_names == ["ground"]
    np.testing.assert_allclose(build.env_origins_w, np.zeros((1, 3)))

    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    assert model.geom_type[ground_id] == mujoco.mjtGeom.mjGEOM_PLANE
    # Fixture material: static_friction = 1.0, combine mode "multiply".
    assert model.geom_friction[ground_id, 0] == pytest.approx(1.0)
    # Friction mapping: PhysX multiply-combine gives mu_pair = mu_foot * mu_ground.
    # MuJoCo pairwise rule is elementwise max unless priorities differ, so the foot
    # gets priority 1 and carries the pre-combined pair value; with mu_ground = 1.0
    # this is exactly mu_foot.
    assert model.geom_priority[foot_id] == 1
    assert model.geom_priority[ground_id] == 0
    assert model.geom_friction[foot_id, 0] == pytest.approx(0.8 * 1.0)
    # Visual geoms are left untouched.
    shell_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shell")
    assert model.geom_priority[shell_id] == 0


def test_pair_friction_combine_modes():
    assert pair_friction(0.8, 0.5, "multiply") == pytest.approx(0.4)
    assert pair_friction(0.8, 0.5, "average") == pytest.approx(0.65)
    assert pair_friction(0.8, 0.5, "min") == pytest.approx(0.5)
    assert pair_friction(0.8, 0.5, "max") == pytest.approx(0.8)
    with pytest.raises(ValueError):
        pair_friction(1.0, 1.0, "geometric")


def test_add_to_spec_applies_pair_friction_for_non_unit_ground():
    terrain = TerrainIR(
        terrain_type="plane",
        physics_material={"static_friction": 0.5, "friction_combine_mode": "multiply"},
    )
    spec = _robot_spec(foot_friction=0.8)
    build_terrain(terrain, spec, np.random.default_rng(0))
    model = spec.compile()
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    assert model.geom_friction[foot_id, 0] == pytest.approx(0.8 * 0.5)
    assert model.geom_priority[foot_id] == 1


def test_add_to_spec_self_pair_scheme_resolves_both_pairings():
    # Self-collision robots: terrain carries the robot-ground pair value at priority 2,
    # robot geoms the self-pair value at priority 1 — both PhysX pairings exact.
    # mu_r=0.8, mu_g=1.0, multiply: ground pair 0.8, self pair 0.64.
    spec = mujoco.MjSpec()
    for k, pos in enumerate(([0.0, 0.0, 0.05], [0.08, 0.0, 0.08])):
        body = spec.worldbody.add_body(name=f"link{k}", pos=pos)
        body.add_freejoint()
        geom = body.add_geom(name=f"geom{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.06, 0, 0])
        geom.friction = [0.8, 0.005, 0.0001]
        geom.contype = 1
        geom.conaffinity = 1
        geom.mass = 1.0
    terrain = TerrainIR(
        terrain_type="plane",
        physics_material={"static_friction": 1.0, "friction_combine_mode": "multiply"},
    )
    build = build_terrain(terrain, rng=np.random.default_rng(0))
    add_to_spec(spec, build, robot_self_pair=True)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert model.geom_priority[ground_id] == 2
    contact_mu = {}
    for contact in data.contact:
        pair = tuple(sorted(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in contact.geom))
        contact_mu[pair] = float(contact.friction[0])
    assert contact_mu[("geom0", "geom1")] == pytest.approx(0.8 * 0.8)  # PhysX self pair
    assert contact_mu[("geom0", "ground")] == pytest.approx(0.8 * 1.0)  # PhysX ground pair


def test_add_to_spec_self_pair_falls_back_on_heterogeneous_robot_friction(capsys):
    spec = _robot_spec(foot_friction=0.8)
    extra = spec.body("base").add_geom(name="toe", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0])
    extra.friction = [0.4, 0.005, 0.0001]
    extra.contype = 1
    extra.conaffinity = 1
    terrain = TerrainIR(
        terrain_type="plane",
        physics_material={"static_friction": 0.5, "friction_combine_mode": "multiply"},
    )
    add_to_spec(spec, build_terrain(terrain, rng=np.random.default_rng(0)), robot_self_pair=True)
    assert "heterogeneous" in capsys.readouterr().out
    model = spec.compile()
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot")
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    assert model.geom_friction[foot_id, 0] == pytest.approx(0.8 * 0.5)  # ground-pair stamping kept
    assert model.geom_priority[ground_id] == 0


def test_plane_grid_env_origins_match_isaaclab_formula():
    build = build_terrain(
        TerrainIR(terrain_type="plane", physics_material={"static_friction": 1.0}),
        num_envs=4,
        env_spacing=2.5,
    )
    # TerrainImporter._compute_env_origins_grid with num_envs=4: 2x2 grid.
    expected = np.array(
        [
            [1.25, -1.25, 0.0],
            [1.25, 1.25, 0.0],
            [-1.25, -1.25, 0.0],
            [-1.25, 1.25, 0.0],
        ]
    )
    np.testing.assert_allclose(build.env_origins_w, expected)


# ---------------------------------------------------------------------------
# generator terrain
# ---------------------------------------------------------------------------


def _build_obstacle_terrain(seed: int = 7, curriculum: bool = False, rng_seed: int = 1) -> TerrainBuild:
    generator = _generator_cfg(
        {"cylinders": _cylinder_sub_terrain(proportion=0.5), "boxes": _box_sub_terrain(proportion=0.5)},
        num_rows=3,
        num_cols=2,
        seed=seed,
        curriculum=curriculum,
    )
    return build_terrain(_generator_ir(generator), rng=np.random.default_rng(rng_seed))


def test_generator_same_seed_identical_layout():
    build_a = _build_obstacle_terrain(seed=7)
    build_b = _build_obstacle_terrain(seed=7)
    assert len(build_a.geoms) == len(build_b.geoms)
    for geom_a, geom_b in zip(build_a.geoms, build_b.geoms):
        assert geom_a == geom_b
    build_c = _build_obstacle_terrain(seed=8)
    assert [g.pos_w for g in build_a.geoms] != [g.pos_w for g in build_c.geoms]


def test_generator_compiles_in_mjspec():
    spec = _robot_spec()
    build = build_terrain(_generator_ir(_generator_cfg({"cyl": _cylinder_sub_terrain()})), spec)
    model = spec.compile()
    assert model.ngeom == 2 + len(build.geoms)  # foot + shell + terrain
    # Obstacle geoms carry the ground friction at priority 0.
    for name in [g.name for g in build.geoms]:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_priority[geom_id] == 0
        assert model.geom_friction[geom_id, 0] == pytest.approx(1.0)


def test_difficulty_interpolation_matches_isaaclab_formulas():
    sub = _cylinder_sub_terrain(num_objects=(5, 100), radius=0.15, height=(0.10, 0.30))
    p0 = repeated_object_params(sub, 0.0)
    assert p0["num_objects"] == 5
    assert p0["height"] == pytest.approx(0.10)
    assert p0["radius"] == pytest.approx(0.15)
    p1 = repeated_object_params(sub, 1.0)
    assert p1["num_objects"] == 100
    assert p1["height"] == pytest.approx(0.30)
    # num_objects truncates: n0 + int(d * (n1 - n0)), exactly as repeated_objects_terrain.
    p_mid = repeated_object_params(sub, 0.37)
    assert p_mid["num_objects"] == 5 + int(0.37 * 95)
    assert p_mid["height"] == pytest.approx(0.10 + 0.37 * 0.20)

    box = _box_sub_terrain(size_xy=(0.1, 0.1))
    box["object_params_end"]["size"] = [0.3, 0.3]
    b_mid = repeated_object_params(box, 0.5)
    assert b_mid["length"] == pytest.approx(0.2)
    assert b_mid["width"] == pytest.approx(0.2)
    # platform_height < 0 falls back to the interpolated object height.
    box["platform_height"] = -1.0
    assert repeated_object_params(box, 1.0)["platform_height"] == pytest.approx(2.0)


def test_difficulty_schedule_matches_generator_rng():
    """Random-mode type/difficulty schedule must consume default_rng(seed) like IsaacLab."""
    seed = 11
    build = _build_obstacle_terrain(seed=seed, curriculum=False)
    np_rng = np.random.default_rng(seed)
    proportions = np.array([0.5, 0.5])
    names = ["cylinders", "boxes"]
    for index, tile in enumerate(build.tiles):
        sub_row, sub_col = np.unravel_index(index, (3, 2))
        assert (tile.row, tile.col) == (sub_row, sub_col)
        expected_name = names[int(np_rng.choice(len(proportions), p=proportions))]
        expected_difficulty = float(np_rng.uniform(0.0, 1.0))
        assert tile.sub_terrain == expected_name
        assert tile.difficulty == pytest.approx(expected_difficulty)


def test_curriculum_schedule_matches_generator_rng():
    seed = 3
    build = _build_obstacle_terrain(seed=seed, curriculum=True)
    np_rng = np.random.default_rng(seed)
    # proportions 0.5/0.5 over 2 cols -> col 0 = cylinders, col 1 = boxes.
    expected = {}
    for sub_col in range(2):
        for sub_row in range(3):
            difficulty = (sub_row + np_rng.uniform()) / 3
            expected[(sub_row, sub_col)] = ("cylinders" if sub_col == 0 else "boxes", difficulty)
    for tile in build.tiles:
        name, difficulty = expected[(tile.row, tile.col)]
        assert tile.sub_terrain == name
        assert tile.difficulty == pytest.approx(difficulty)


def test_tile_grid_origins_match_rows_cols_size():
    generator = _generator_cfg({"cyl": _cylinder_sub_terrain()}, num_rows=3, num_cols=2, size=(8.0, 6.0))
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    assert build.terrain_origins_w is not None
    assert build.terrain_origins_w.shape == (3, 2, 3)
    for row in range(3):
        for col in range(2):
            expected_x = (row + 0.5) * 8.0 - 3 * 8.0 / 2
            expected_y = (col + 0.5) * 6.0 - 2 * 6.0 / 2
            np.testing.assert_allclose(build.terrain_origins_w[row, col], [expected_x, expected_y, 0.0], atol=1e-12)


def test_platform_center_clear():
    build = _build_obstacle_terrain(seed=5)
    for tile in build.tiles:
        assert tile.obstacles, f"tile ({tile.row}, {tile.col}) has no obstacles"
        (low_x, low_y), (high_x, high_y) = platform_clear_region_w(tile)
        for obstacle in tile.obstacles:
            inside = low_x <= obstacle.x_w <= high_x and low_y <= obstacle.y_w <= high_y
            assert not inside, f"obstacle at ({obstacle.x_w}, {obstacle.y_w}) inside platform clear region"
        # Sanity: the platform center itself has ground-level height (platform_height = 0).
        assert tile.height_w(tile.origin_w[0], tile.origin_w[1]) == pytest.approx(0.0)


def test_object_counts_match_difficulty():
    build = _build_obstacle_terrain(seed=13)
    for tile in build.tiles:
        sub = _cylinder_sub_terrain(proportion=0.5) if tile.sub_terrain == "cylinders" else _box_sub_terrain(0.5)
        expected = repeated_object_params(sub, tile.difficulty)
        assert len(tile.obstacles) == expected["num_objects"]
        for obstacle in tile.obstacles:
            assert obstacle.top_z_w == pytest.approx(0.5 * expected["height"])
            if obstacle.kind == "cylinder":
                assert obstacle.radius == pytest.approx(expected["radius"])
            else:
                assert 2 * obstacle.half_x == pytest.approx(expected["length"])
                assert 2 * obstacle.half_y == pytest.approx(expected["width"])


def test_flat_patches_avoid_obstacles():
    patch_cfg = {
        "num_patches": 100,
        "patch_radius": [0.1, 0.25],
        "x_range": [-18.0, 18.0],
        "y_range": [-18.0, 18.0],
        "z_range": [-0.1, 0.1],
        "max_height_diff": 0.001,
    }
    sub = _cylinder_sub_terrain(
        num_objects=(30, 30), radius=0.3, flat_patches={"target": patch_cfg, "init_pos": patch_cfg}
    )
    build = build_terrain(
        _generator_ir(_generator_cfg({"big_cylinders": sub}, num_rows=1, num_cols=1, seed=21)),
        rng=np.random.default_rng(2),
    )
    assert set(build.flat_patches) == {"target", "init_pos"}
    sampler = build.flat_patches["target"]
    assert isinstance(sampler, FlatPatchSampler)
    patches_w = sampler(np.random.default_rng(42))
    assert patches_w.shape == (100, 3)
    np.testing.assert_allclose(patches_w[:, 2], 0.0, atol=1e-12)

    tile = build.tiles[0]
    min_patch_radius = 0.1
    for x_w, y_w, _ in patches_w:
        # Inside the tile with room for the largest validity ring.
        assert abs(x_w - tile.origin_w[0]) <= 10.0
        assert abs(y_w - tile.origin_w[1]) <= 10.0
        for obstacle in tile.obstacles:
            dist = np.hypot(x_w - obstacle.x_w, y_w - obstacle.y_w)
            assert dist > obstacle.radius + min_patch_radius, (
                f"patch ({x_w:.3f}, {y_w:.3f}) within footprint + radius of obstacle at "
                f"({obstacle.x_w:.3f}, {obstacle.y_w:.3f})"
            )

    # Same rng seed -> identical patches (seeded rejection sampling).
    np.testing.assert_array_equal(patches_w, sampler(np.random.default_rng(42)))


def test_flat_patch_sampler_rejects_impossible_cfg():
    patch_cfg = {
        "num_patches": 5,
        "patch_radius": [0.1],
        "x_range": [-18.0, 18.0],
        "y_range": [-18.0, 18.0],
        "z_range": [0.5, 0.6],  # nothing sits at this height
        "max_height_diff": 0.001,
    }
    sub = _cylinder_sub_terrain(num_objects=(3, 3), flat_patches={"target": patch_cfg})
    build = build_terrain(
        _generator_ir(_generator_cfg({"cyl": sub}, num_rows=1, num_cols=1, seed=0)),
        rng=np.random.default_rng(0),
    )
    with pytest.raises(RuntimeError, match="flat-patch sampling failed"):
        build.flat_patches["target"](np.random.default_rng(0))


class TestCurriculumTerrainTypes:
    """Column assignment must be bit-exact with TerrainImporter's float32 floor-div."""

    def test_num_envs_off_by_one(self):
        # floor(i / (7/3)) and floor(i / (5/3)): exact column-boundary arithmetic.
        np.testing.assert_array_equal(_curriculum_terrain_types(7, 3), [0, 0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(_curriculum_terrain_types(5, 3), [0, 0, 1, 1, 2])
        np.testing.assert_array_equal(_curriculum_terrain_types(3, 2), [0, 0, 1])

    def test_inexact_boundary_regression(self):
        # num_envs/num_cols = 409.6 is not float-representable; torch's fmod-based
        # kernel assigns env 2048 to column 4, where a naive floor(2048/409.6) gives 5.
        assert _curriculum_terrain_types(4096, 10)[2048] == 4

    @pytest.mark.parametrize(
        ("num_envs", "num_cols"),
        [(6, 3), (7, 3), (5, 3), (4095, 10), (4096, 10), (4097, 10), (8192, 20), (100, 7), (101, 7), (99, 7)],
    )
    def test_matches_torch_reference(self, num_envs, num_cols):
        torch = pytest.importorskip("torch")
        expected = torch.div(torch.arange(num_envs), num_envs / num_cols, rounding_mode="floor").to(torch.long).numpy()
        np.testing.assert_array_equal(_curriculum_terrain_types(num_envs, num_cols), expected)


def test_generator_env_origins_use_curriculum_layout():
    build = _build_obstacle_terrain(seed=7, rng_seed=123)
    assert build.terrain_origins_w is not None
    rng = np.random.default_rng(123)
    # build_terrain consumes nothing from rng before the terrain levels (seeded generator cfg).
    levels = rng.integers(0, 3, size=1)  # max_init_level = num_rows - 1 = 2
    types = (np.arange(1) / (1 / 2)).astype(np.int64)
    np.testing.assert_allclose(build.env_origins_w, build.terrain_origins_w[levels, types])


# ---------------------------------------------------------------------------
# NotImplementedError paths
# ---------------------------------------------------------------------------


def test_heightfield_stairs_not_implemented():
    """The height-field stairs variant shares the mesh function's bare name; only the
    trimesh implementation has a box mapping."""
    sub = {
        "function": "isaaclab.terrains.height_field.hf_terrains:pyramid_stairs_terrain",
        "proportion": 1.0,
        "step_height_range": [0.05, 0.23],
        "step_width": 0.3,
        "platform_width": 3.0,
    }
    generator = _generator_cfg({"hf_stairs": sub}, num_rows=1, num_cols=1)
    with pytest.raises(NotImplementedError, match="pyramid_stairs_terrain"):
        build_terrain(_generator_ir(generator))


def test_tilted_objects_not_implemented():
    sub = _cylinder_sub_terrain()
    sub["object_params_start"]["max_yx_angle"] = 30.0
    sub["object_params_end"]["max_yx_angle"] = 30.0
    generator = _generator_cfg({"tilted": sub}, num_rows=1, num_cols=1)
    with pytest.raises(NotImplementedError, match="max_yx_angle"):
        build_terrain(_generator_ir(generator))


# ---------------------------------------------------------------------------
# ROUGH_TERRAINS_CFG sub-terrains (heightfield + mesh)
# ---------------------------------------------------------------------------

_HF_RANDOM_UNIFORM_FUNC = "isaaclab.terrains.height_field.hf_terrains:random_uniform_terrain"
_HF_PYRAMID_SLOPED_FUNC = "isaaclab.terrains.height_field.hf_terrains:pyramid_sloped_terrain"
_MESH_STAIRS_FUNC = "isaaclab.terrains.trimesh.mesh_terrains:pyramid_stairs_terrain"
_MESH_STAIRS_INV_FUNC = "isaaclab.terrains.trimesh.mesh_terrains:inverted_pyramid_stairs_terrain"
_MESH_RANDOM_GRID_FUNC = "isaaclab.terrains.trimesh.mesh_terrains:random_grid_terrain"


def _random_uniform_sub(proportion: float = 1.0) -> dict:
    return {
        "function": _HF_RANDOM_UNIFORM_FUNC,
        "proportion": proportion,
        "noise_range": [0.02, 0.10],
        "noise_step": 0.02,
        "border_width": 0.25,
        "downsampled_scale": None,
    }


def _pyramid_sloped_sub(proportion: float = 1.0, inverted: bool = False) -> dict:
    return {
        "function": _HF_PYRAMID_SLOPED_FUNC,
        "proportion": proportion,
        "slope_range": [0.0, 0.4],
        "platform_width": 2.0,
        "border_width": 0.25,
        "inverted": inverted,
    }


def _stairs_sub(proportion: float = 1.0, inverted: bool = False) -> dict:
    return {
        "function": _MESH_STAIRS_INV_FUNC if inverted else _MESH_STAIRS_FUNC,
        "proportion": proportion,
        "step_height_range": [0.05, 0.23],
        "step_width": 0.3,
        "platform_width": 3.0,
        "border_width": 1.0,
        "holes": False,
    }


def _random_grid_sub(proportion: float = 1.0) -> dict:
    return {
        "function": _MESH_RANDOM_GRID_FUNC,
        "proportion": proportion,
        "grid_width": 0.45,
        "grid_height_range": [0.05, 0.2],
        "platform_width": 2.0,
        "holes": False,
    }


_ROUGH_SUBS = {
    "pyramid_stairs": _stairs_sub(0.2),
    "pyramid_stairs_inv": _stairs_sub(0.2, inverted=True),
    "boxes": _random_grid_sub(0.2),
    "random_rough": _random_uniform_sub(0.2),
    "hf_pyramid_slope": _pyramid_sloped_sub(0.1),
    "hf_pyramid_slope_inv": _pyramid_sloped_sub(0.1, inverted=True),
}


def _rough_generator(
    sub_terrains: dict,
    num_rows: int = 1,
    num_cols: int = 1,
    seed: int = 1,
    curriculum: bool = True,
    difficulty_range: tuple[float, float] = (0.5, 0.5),
    border_width: float = 0.0,
    border_height: float = 1.0,
) -> dict:
    return {
        "seed": seed,
        "curriculum": curriculum,
        "size": [8.0, 8.0],
        "border_width": border_width,
        "border_height": border_height,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "horizontal_scale": 0.1,
        "vertical_scale": 0.005,
        "slope_threshold": 0.75,
        "difficulty_range": list(difficulty_range),
        "sub_terrains": sub_terrains,
    }


def _one_tile_build(sub: dict, difficulty: float = 0.5, seed: int = 1) -> TerrainBuild:
    generator = _rough_generator({"t": sub}, difficulty_range=(difficulty, difficulty), seed=seed)
    return build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))


@pytest.mark.parametrize("name", sorted(_ROUGH_SUBS))
def test_height_at_matches_mujoco_raycast(name):
    """Each rough sub-terrain compiles via add_to_spec, and height_at is the
    collision surface: a MuJoCo downward raycast agrees."""
    build = _one_tile_build(_ROUGH_SUBS[name])
    spec = mujoco.MjSpec()
    build_names = add_to_spec(spec, build)
    model = spec.compile()
    assert build_names == [g.name for g in build.geoms]
    assert model.ngeom == len(build.geoms)
    assert model.nhfield == len(build.hfields)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    rng = np.random.default_rng(9)
    x = rng.uniform(-3.95, 3.95, 200)
    y = rng.uniform(-3.95, 3.95, 200)
    expected = build.height_at(x, y)
    geomid = np.zeros(1, dtype=np.int32)
    for k in range(len(x)):
        point = np.array([x[k], y[k], 50.0])
        dist = mujoco.mj_ray(model, data, point, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert dist >= 0
        # hfield_data is float32; everything else is exact.
        assert 50.0 - dist == pytest.approx(expected[k], abs=1e-6)


def test_height_at_sees_obstacles_overhanging_tile_borders():
    # Repeated-objects centers sample the full tile extent, so a footprint can reach
    # past the border: a box 0.1 m inside tile (0, 0)'s +x edge with half_x = 0.5
    # overhangs 0.4 m into tile (1, 0). Isaac raycasts the merged mesh (and MuJoCo
    # collides the geom), so height_at must report the box top from the neighbor too.
    box = _Obstacle(kind="box", x_w=-0.1, y_w=0.0, top_z_w=0.5, half_x=0.5, half_y=0.5)
    tiles = [
        TileBuild(
            row=0, col=0, sub_terrain="t", difficulty=0.0, origin_w=(-2.0, 0.0, 0.0), size=(4.0, 4.0), obstacles=[box]
        ),
        TileBuild(row=1, col=0, sub_terrain="t", difficulty=0.0, origin_w=(2.0, 0.0, 0.0), size=(4.0, 4.0)),
    ]
    build = TerrainBuild(
        terrain_type="generator",
        geoms=[],
        ground_geom_names=[],
        env_origins_w=np.zeros((1, 3)),
        flat_patches={},
        friction_combine_mode="multiply",
        ground_friction=1.0,
        tiles=tiles,
        tile_size=(4.0, 4.0),
        grid_shape=(2, 1),
    )
    assert build.height_at(0.2, 0.0) == pytest.approx(0.5)  # neighbor query over the overhang
    assert build.height_at(0.5, 0.0) == pytest.approx(0.0)  # neighbor query past the overhang
    assert build.height_at(-0.3, 0.0) == pytest.approx(0.5)  # owning-tile query unchanged


@pytest.mark.parametrize("inverted", [False, True])
def test_pyramid_stairs_height_anchors(inverted):
    """Hand-derived stair heights at difficulty 0.5: step_height = 0.05 + 0.5 * 0.18
    = 0.14, num_steps = (8 - 2*1 - 3) // (2*0.3) + 1 = 6 (float floordiv gives 5.0),
    ring k (Chebyshev band (3-(k+1)*0.3, 3-k*0.3) from the tile center) tops out at
    (k+1)*0.14, platform at (6+1)*0.14. Inverted stairs mirror every anchor to
    -(k+1)*0.14 with the tile border staying at 0; at ring edges the higher (outer)
    surface wins the raycast, which is the same signed pattern in both cases."""
    sign = -1.0 if inverted else 1.0
    build = _one_tile_build(_stairs_sub(inverted=inverted))
    tile = build.tiles[0]
    step_height = 0.14
    assert tile.origin_w == pytest.approx((0.0, 0.0, sign * 7 * step_height))
    assert build.height_at(0.0, 0.0) == pytest.approx(sign * 7 * step_height)  # platform
    assert build.height_at(2.9, 0.0) == pytest.approx(sign * 1 * step_height)  # ring 0
    assert build.height_at(0.0, 2.5) == pytest.approx(sign * 2 * step_height)  # ring 1
    assert build.height_at(2.05, 0.0) == pytest.approx(sign * 4 * step_height)  # ring 3
    assert build.height_at(2.9, 2.9) == pytest.approx(sign * 1 * step_height)  # ring corner
    assert build.height_at(-3.5, 0.0) == pytest.approx(0.0)  # tile border
    # Step edge between ring 1 and ring 0 at 2.7 m: heights jump by one step.
    assert build.height_at(2.7 - 1e-6, 0.0) == pytest.approx(sign * 2 * step_height)
    assert build.height_at(2.7 + 1e-6, 0.0) == pytest.approx(sign * 1 * step_height)
    assert np.isnan(build.height_at(4.5, 0.0))  # off the terrain (no generator border)


@pytest.mark.parametrize("inverted", [False, True])
def test_pyramid_sloped_height_anchors(inverted):
    """Hand-derived pyramid-slope heights at difficulty 0.5 (slope 0.2): the bordered
    inner grid is 75x75 nodes (center 37), height_max = int(0.2 * 7.5 / 2 / 0.005) =
    150 grid units, platform clip z_pf = 150 * (27/37)^2 -> rint 80 units = 0.40 m,
    and the node at world x = -2.0 (inner i = 17) is rint(150 * 17/37) = 69 units."""
    sign = -1.0 if inverted else 1.0
    build = _one_tile_build(_pyramid_sloped_sub(inverted=inverted))
    tile = build.tiles[0]
    assert tile.origin_w == pytest.approx((0.0, 0.0, sign * 0.40) if not inverted else (0.0, 0.0, -0.40))
    assert build.height_at(0.0, 0.0) == pytest.approx(sign * 0.40)
    assert build.height_at(-2.0, 0.0) == pytest.approx(sign * 69 * 0.005)
    assert build.height_at(-3.85, 0.0) == pytest.approx(0.0)  # border pixels
    # The platform is flat: both platform points agree.
    assert build.height_at(0.3, -0.4) == pytest.approx(sign * 0.40)


def test_pyramid_sloped_zero_difficulty_is_flat():
    build = _one_tile_build(_pyramid_sloped_sub(inverted=True), difficulty=0.0)
    x = np.linspace(-3.9, 3.9, 25)
    np.testing.assert_allclose(build.height_at(x, x), np.zeros(25), atol=1e-12)
    assert build.tiles[0].origin_w[2] == pytest.approx(0.0)


def test_random_uniform_bounds_and_lattice():
    build = _one_tile_build(_random_uniform_sub(), seed=3)
    tile = build.tiles[0]
    assert tile.grid_z_w is not None and tile.grid_z_w.shape == (81, 81)
    # Interior nodes (3 border pixels on each side) sit on the noise_step lattice
    # within noise_range; border nodes are zero.
    interior = tile.grid_z_w[3:-3, 3:-3]
    np.testing.assert_allclose(np.round(interior / 0.02) * 0.02, interior, atol=1e-12)
    assert interior.min() >= 0.02 - 1e-12 and interior.max() <= 0.10 + 1e-12
    np.testing.assert_allclose(tile.grid_z_w[0, :], 0.0)
    np.testing.assert_allclose(tile.grid_z_w[:, -1], 0.0)
    # height_at at exact node coordinates returns the node heights.
    i, j = 20, 45
    node_x = -4.0 + i * 0.1
    node_y = -4.0 + j * 0.1
    assert build.height_at(node_x, node_y) == pytest.approx(tile.grid_z_w[i, j])
    # Interpolated interior heights stay within the sampled node range.
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.6, 3.6, 300)
    y = rng.uniform(-3.6, 3.6, 300)
    heights = build.height_at(x, y)
    assert heights.min() >= 0.02 - 1e-12 and heights.max() <= 0.10 + 1e-12
    # Tile origin: max grid height over the central 2 m window.
    x1, x2 = int((8.0 * 0.5 - 1) / 0.1), int((8.0 * 0.5 + 1) / 0.1)
    assert tile.origin_w[2] == pytest.approx(tile.grid_z_w[x1:x2, x1:x2].max())


def test_random_grid_layout():
    """8 m / 0.45 m -> 17x17 cells, border = 8 - 17*0.45 = 0.35 m (top at 0),
    difficulty 0.5 -> grid_height = 0.125: cell tops in [-0.125, 0.125], platform
    top exactly at grid_height."""
    build = _one_tile_build(_random_grid_sub())
    tile = build.tiles[0]
    grid_height = 0.125
    assert tile.origin_w == pytest.approx((0.0, 0.0, grid_height))
    assert tile.platform_top_z_w == pytest.approx(grid_height)
    assert build.height_at(0.0, 0.0) == pytest.approx(grid_height)
    cell_names = [g.name for g in build.geoms if "_cell_" in g.name]
    assert len(cell_names) == 17 * 17
    # Cell tops: uniform offsets in (-grid_height, grid_height); constant per cell.
    x0 = -4.0 + 0.35 / 2  # first cell edge
    for i, j in [(0, 0), (5, 11), (16, 16)]:
        cx = x0 + (i + 0.5) * 0.45
        cy = x0 + (j + 0.5) * 0.45
        top = build.height_at(cx, cy)
        assert abs(top) <= grid_height
        assert build.height_at(cx + 0.1, cy - 0.1) == pytest.approx(top)
    # Border ring around the cells is at ground level.
    assert build.height_at(-3.95, 0.0) == pytest.approx(0.0)
    assert build.height_at(0.0, 3.95) == pytest.approx(0.0)


def test_random_grid_requires_positive_border():
    sub = _random_grid_sub()
    sub["grid_width"] = 2.0  # 8 / 2 = 4 cells exactly -> zero border
    with pytest.raises(RuntimeError, match="Border width"):
        _one_tile_build(sub)


def test_rough_curriculum_column_assignment_matches_proportions():
    """ROUGH proportions (0.2, 0.2, 0.2, 0.2, 0.1, 0.1) over 10 columns map columns
    to sub-terrains via the cumulative-proportion rule of TerrainGenerator."""
    generator = _rough_generator(
        _ROUGH_SUBS, num_rows=2, num_cols=10, curriculum=True, difficulty_range=(0.0, 1.0), seed=42
    )
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    names = list(_ROUGH_SUBS)
    expected_cols = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5]
    for tile in build.tiles:
        assert tile.sub_terrain == names[expected_cols[tile.col]]
    # Difficulty grows along rows within each column.
    for col in range(10):
        col_tiles = sorted((t for t in build.tiles if t.col == col), key=lambda t: t.row)
        assert col_tiles[0].difficulty < col_tiles[1].difficulty
    # Platform-centered tiles spawn exactly on the surface; random_rough spawns at
    # the max grid height of the central 2 m window (height_field_to_mesh origin),
    # which sits on or above the surface under the origin.
    for tile in build.tiles:
        surface_z = build.height_at(*tile.origin_w[:2])
        if tile.sub_terrain == "random_rough":
            assert tile.origin_w[2] >= surface_z - 1e-9
        else:
            assert surface_z == pytest.approx(tile.origin_w[2], abs=1e-9)


def test_rough_same_seed_identical_layout():
    def build_once(seed):
        generator = _rough_generator(
            _ROUGH_SUBS, num_rows=1, num_cols=6, curriculum=True, difficulty_range=(0.0, 1.0), seed=seed
        )
        return build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))

    build_a, build_b, build_c = build_once(7), build_once(7), build_once(8)
    assert [g for g in build_a.geoms] == [g for g in build_b.geoms]
    assert len(build_a.hfields) == len(build_b.hfields) > 0
    for hf_a, hf_b in zip(build_a.hfields, build_b.hfields):
        assert (hf_a.name, hf_a.nrow, hf_a.ncol, hf_a.size) == (hf_b.name, hf_b.nrow, hf_b.ncol, hf_b.size)
        np.testing.assert_array_equal(hf_a.data_w, hf_b.data_w)
    assert [g.pos_w for g in build_a.geoms] != [g.pos_w for g in build_c.geoms]


def test_generator_border_ring_and_off_terrain():
    generator = _rough_generator({"t": _stairs_sub()}, border_width=1.5, border_height=1.0)
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    border_names = [g.name for g in build.geoms if g.name.startswith("border_")]
    assert sorted(border_names) == ["border_bottom", "border_left", "border_right", "border_top"]
    # Positive border_height digs below ground: the ring top is at z = 0.
    assert build.height_at(4.5, 0.0) == pytest.approx(0.0)
    assert np.isnan(build.height_at(6.0, 0.0))  # beyond the ring
    # Negative border_height raises the ring above ground (TerrainGeneratorCfg note).
    generator = _rough_generator({"t": _stairs_sub()}, border_width=1.5, border_height=-0.3)
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    assert build.height_at(4.5, 0.0) == pytest.approx(0.3)


def test_generator_border_geoms_form_centered_ring():
    """Border boxes are world geoms surrounding the centered grid: for a 2x3 grid
    of 8 m tiles with border_width 1.5, inner = (16, 24), outer = (19, 27), and
    the left box center sits at (-(16 + 1.5) / 2, 0) — the ring's own center is
    the world origin, matching TerrainGenerator's post-centering make_border."""
    generator = _rough_generator({"t": _stairs_sub()}, num_rows=2, num_cols=3, border_width=1.5, border_height=1.0)
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    borders = {g.name: g for g in build.geoms if g.name.startswith("border_")}
    np.testing.assert_allclose(borders["border_left"].pos_w, (-8.75, 0.0, -0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_right"].pos_w, (8.75, 0.0, -0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_top"].pos_w, (0.0, 12.75, -0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_bottom"].pos_w, (0.0, -12.75, -0.5), atol=1e-12)
    # make_border box dims: left/right span thickness x inner_y, top/bottom span
    # outer_x x thickness (sizes are half-extents).
    np.testing.assert_allclose(borders["border_left"].size, (0.75, 12.0, 0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_right"].size, (0.75, 12.0, 0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_top"].size, (9.5, 0.75, 0.5), atol=1e-12)
    np.testing.assert_allclose(borders["border_bottom"].size, (9.5, 0.75, 0.5), atol=1e-12)


def test_generator_border_ring_matches_mujoco_raycast():
    """With a border ring present, the compiled collision surface agrees with
    height_at over the ring region AND the tile interior (border boxes must not
    bury the tile surfaces)."""
    generator = _rough_generator({"t": _stairs_sub()}, border_width=1.5, border_height=1.0)
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    spec = mujoco.MjSpec()
    add_to_spec(spec, build)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def raycast_z(x: float, y: float) -> float:
        geomid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(model, data, np.array([x, y, 50.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert dist >= 0, f"no surface under ({x}, {y})"
        return 50.0 - dist

    # Interior + ring sweep: everything inside the outer boundary has a surface
    # that matches height_at (the tile spans |x|,|y| <= 4; the ring 4 <= . <= 5.5).
    rng = np.random.default_rng(9)
    x = rng.uniform(-5.45, 5.45, 300)
    y = rng.uniform(-5.45, 5.45, 300)
    expected = build.height_at(x, y)
    assert np.all(np.isfinite(expected))
    for k in range(len(x)):
        assert raycast_z(x[k], y[k]) == pytest.approx(expected[k], abs=1e-9)
    # Pure ring probes on all four sides and a corner: top at z = 0.
    for px, py in [(4.7, 0.0), (-4.7, 0.0), (0.0, 4.7), (0.0, -4.7), (4.7, 4.7), (-5.4, -5.4)]:
        assert raycast_z(px, py) == pytest.approx(0.0, abs=1e-12)

    # Negative border_height raises the ring: its top compiles at z = 0.3.
    generator = _rough_generator({"t": _stairs_sub()}, border_width=1.5, border_height=-0.3)
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(0))
    spec = mujoco.MjSpec()
    add_to_spec(spec, build)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for px, py in [(4.7, 0.0), (0.0, -4.7)]:
        assert raycast_z(px, py) == pytest.approx(0.3, abs=1e-12)


@pytest.mark.parametrize(
    ("sub", "platform_z", "max_center_dist"),
    # The stairs platform is a 3 m square (centers within 1.5 m). The sloped
    # terrain's flat plateau is the np.clip region {xx * yy >= z_pf / height_max}, a
    # hyperbola-bounded star reaching ~1.74 m along the axes.
    [(_pyramid_sloped_sub(), 0.40, 1.8), (_stairs_sub(), 0.98, 1.5)],
    ids=["pyramid_sloped", "pyramid_stairs"],
)
def test_flat_patches_only_on_platform(sub, platform_z, max_center_dist):
    """With a z_range pinned to the platform height, every sampled patch lands on the
    flat central platform (steps/slopes fail the ring z_range / height-diff checks).
    z_range is relative to the tile origin (find_flat_patches shifts it by the origin
    z, which equals the platform height for these terrains)."""
    patch_cfg = {
        "num_patches": 30,
        "patch_radius": [0.1, 0.3],
        "x_range": [-3.0, 3.0],
        "y_range": [-3.0, 3.0],
        "z_range": [-0.31, 0.01],
        "max_height_diff": 0.005,
    }
    sub = dict(sub, flat_patch_sampling={"target": patch_cfg})
    build = _one_tile_build(sub)
    patches_w = build.flat_patches["target"](np.random.default_rng(5))
    assert patches_w.shape == (30, 3)
    # z_range shifts by the tile origin z (= platform height for these terrains).
    tile = build.tiles[0]
    assert tile.origin_w[2] == pytest.approx(platform_z)
    ring = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    for x_w, y_w, z_w in patches_w:
        assert z_w == pytest.approx(platform_z, abs=1e-9)
        assert abs(x_w) <= max_center_dist and abs(y_w) <= max_center_dist
        # The neighborhood is flat at the platform height up to max_height_diff
        # (edge patches may overhang the plateau rim by up to that tolerance).
        neighborhood = build.height_at(x_w + 0.29 * np.cos(ring), y_w + 0.29 * np.sin(ring))
        np.testing.assert_allclose(neighborhood, platform_z, atol=0.006)


def test_env_origins_land_on_rough_tile_origins():
    generator = _rough_generator(
        _ROUGH_SUBS, num_rows=3, num_cols=6, curriculum=True, difficulty_range=(0.0, 1.0), seed=11
    )
    build = build_terrain(_generator_ir(generator), rng=np.random.default_rng(4), num_envs=1, max_init_terrain_level=0)
    # max_init_terrain_level = 0 pins the level; a single env maps to column 0.
    assert build.terrain_origins_w is not None
    np.testing.assert_allclose(build.env_origins_w[0], build.terrain_origins_w[0, 0])
    tile = build.tile_at(0, 0)
    np.testing.assert_allclose(build.env_origins_w[0], np.asarray(tile.origin_w))
    assert tile.origin_w[2] != 0.0  # stairs column: spawn height follows the platform


# ---------------------------------------------------------------------------
# measured terrain (grid + mesh captured from a reference dump)
# ---------------------------------------------------------------------------


def _grid_mesh(grid_z: np.ndarray, res: float, x0: float, y0: float) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated lattice surface with the hfield cell split: upper triangle
    (v00, v01, v11) above the (i, j) -> (i+1, j+1) diagonal, lower (v00, v10, v11)."""
    nx, ny = grid_z.shape
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    vertices = np.stack([x0 + res * ii, y0 + res * jj, grid_z], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00, v10 = i * ny + j, (i + 1) * ny + j
            v01, v11 = i * ny + j + 1, (i + 1) * ny + j + 1
            faces.append((v00, v01, v11))
            faces.append((v00, v10, v11))
    return vertices, np.asarray(faces, dtype=np.int32)


def _step_mesh(step_z: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Two horizontal rectangles joined by a vertical wall at x = 1: z = 0 over
    [0, 1] x [0, 1] and z = step_z over [1, 2] x [0, 1]."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, step_z],
            [2.0, 0.0, step_z],
            [2.0, 1.0, step_z],
            [1.0, 1.0, step_z],
        ]
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [1, 4, 7], [1, 7, 2]],  # last two: vertical wall
        dtype=np.int32,
    )
    return vertices, faces


def _interp_tile(grid_z: np.ndarray, res: float, x0: float, y0: float) -> TileBuild:
    """TileBuild carrying the grid, used as the hfield-interpolation reference."""
    nx, ny = grid_z.shape
    span_x, span_y = (nx - 1) * res, (ny - 1) * res
    return TileBuild(
        row=0,
        col=0,
        sub_terrain="measured",
        difficulty=0.0,
        origin_w=(x0 + 0.5 * span_x, y0 + 0.5 * span_y, 0.0),
        size=(span_x, span_y),
        surface_kind="hfield",
        grid_z_w=np.asarray(grid_z, dtype=np.float64),
        grid_x0_w=x0,
        grid_y0_w=y0,
        grid_step=res,
    )


def _step_measured_data(step_z: float = 0.5, res: float = 0.25) -> "MeasuredTerrainData":
    vertices, faces = _step_mesh(step_z)
    surface = MeasuredSurface(vertices, faces, cell=2.0 * res)
    nx, ny = int(round(2.0 / res)) + 1, int(round(1.0 / res)) + 1
    gx, gy = np.meshgrid(res * np.arange(nx), res * np.arange(ny), indexing="ij")
    grid = surface.heights_w(gx, gy)
    assert not np.isnan(grid).any()
    return MeasuredTerrainData(
        height_grid=grid.astype(np.float32),
        grid_origin_xy=(0.0, 0.0),
        grid_resolution=res,
        mesh_vertices_w=vertices.astype(np.float32),
        mesh_faces=faces,
        env_origin_w=np.array([0.5, 0.5, 0.0]),
        tile_origins_w=np.zeros((1, 1, 3)),
    )


def test_measured_surface_matches_hfield_interp_on_continuous_mesh():
    """On a continuous lattice mesh (the hfield triangulation itself), the exact
    mesh raycast and the grid interpolation are the same surface."""
    rng = np.random.default_rng(3)
    grid_z = rng.uniform(-0.2, 0.3, size=(6, 9))
    res, x0, y0 = 0.1, -0.25, 0.4
    vertices, faces = _grid_mesh(grid_z, res, x0, y0)
    surface = MeasuredSurface(vertices, faces, cell=2.0 * res)
    tile = _interp_tile(grid_z, res, x0, y0)

    x = rng.uniform(x0, x0 + 0.5, 300)
    y = rng.uniform(y0, y0 + 0.8, 300)
    np.testing.assert_allclose(surface.heights_w(x, y), tile.heights_w(x, y), atol=1e-12)
    # Exact at the nodes.
    ii, jj = np.meshgrid(np.arange(6), np.arange(9), indexing="ij")
    np.testing.assert_allclose(surface.heights_w(x0 + res * ii, y0 + res * jj), grid_z, atol=1e-12)


def test_measured_surface_step_discontinuity():
    """Vertical walls survive the mesh raycast exactly: no interpolation ramp,
    on-edge queries return the top surface, off-mesh queries return NaN."""
    surface = MeasuredSurface(*_step_mesh(step_z=0.5))
    assert surface.heights_w(0.999, 0.5)[0] == pytest.approx(0.0, abs=1e-12)
    assert surface.heights_w(1.001, 0.5)[0] == pytest.approx(0.5, abs=1e-12)
    assert surface.heights_w(1.0, 0.5)[0] == pytest.approx(0.5, abs=1e-12)  # edge: first (top) hit
    assert np.isnan(surface.heights_w(2.5, 0.5)[0])
    assert np.isnan(surface.heights_w(0.5, -0.5)[0])


def test_build_measured_terrain_hfield_and_height_at():
    """The bundle hfield is the measured grid (MuJoCo raycast == grid interpolation)
    while height_at raycasts the measured mesh (exact across the step wall)."""
    measured = _step_measured_data(step_z=0.5, res=0.25)
    terrain_ir = TerrainIR(
        terrain_type="generator",
        generator=None,
        physics_material={"static_friction": 0.9, "friction_combine_mode": "multiply"},
    )
    build = build_terrain(terrain_ir, measured=measured)
    assert build.terrain_type == "measured"
    assert build.ground_friction == pytest.approx(0.9)
    assert build.robot_pair_friction(0.8) == pytest.approx(0.72)
    np.testing.assert_allclose(build.env_origins_w, [[0.5, 0.5, 0.0]])
    assert build.terrain_origins_w is not None and build.terrain_origins_w.shape == (1, 1, 3)

    # height_at: the exact mesh surface, including inside the wall-crossing grid cell
    # where the interpolated grid would ramp.
    assert build.height_at(0.85, 0.5) == pytest.approx(0.0, abs=1e-7)
    assert build.height_at(1.05, 0.5) == pytest.approx(0.5, abs=1e-7)
    tile = _interp_tile(measured.height_grid.astype(np.float64), 0.25, 0.0, 0.0)
    assert tile.heights_w(np.array([0.85]), np.array([0.5]))[0] > 0.05  # the ramp the mesh avoids
    assert np.isnan(build.height_at(2.5, 0.5))

    # MuJoCo collision surface: the compiled hfield equals the grid interpolation.
    spec = mujoco.MjSpec()
    names = add_to_spec(spec, build)
    assert names == ["terrain_measured"]
    model = spec.compile()
    assert model.nhfield == 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(11)
    x = rng.uniform(0.01, 1.99, 200)
    y = rng.uniform(0.01, 0.99, 200)
    expected = tile.heights_w(x, y)
    geomid = np.zeros(1, dtype=np.int32)
    for k in range(len(x)):
        dist = mujoco.mj_ray(model, data, np.array([x[k], y[k], 5.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        assert dist >= 0
        assert 5.0 - dist == pytest.approx(expected[k], abs=1e-6)


def test_measured_npz_roundtrip(tmp_path):
    measured = _step_measured_data()
    path = tmp_path / "terrain_measured.npz"
    measured.save_npz(path)
    loaded = MeasuredTerrainData.from_npz(np.load(path))
    assert loaded is not None
    np.testing.assert_array_equal(loaded.height_grid, measured.height_grid)
    assert loaded.grid_origin_xy == measured.grid_origin_xy
    assert loaded.grid_resolution == measured.grid_resolution
    np.testing.assert_array_equal(loaded.mesh_vertices_w, measured.mesh_vertices_w)
    np.testing.assert_array_equal(loaded.mesh_faces, measured.mesh_faces)
    np.testing.assert_allclose(loaded.env_origin_w, measured.env_origin_w)
    assert loaded.tile_origins_w is not None

    # A record-free npz (plane dump / pre-capture dump) parses to None.
    plain = tmp_path / "plain.npz"
    np.savez(plain, obs=np.zeros(3))
    assert MeasuredTerrainData.from_npz(np.load(plain)) is None


def test_measured_requires_generator_terrain():
    measured = _step_measured_data()
    plane = TerrainIR(terrain_type="plane", physics_material={"static_friction": 1.0})
    with pytest.raises(ValueError, match="generator terrain only"):
        build_terrain(plane, measured=measured)


def test_measured_rejects_nan_grid():
    measured = _step_measured_data()
    grid = measured.height_grid.copy()
    grid[0, 0] = np.nan
    bad = MeasuredTerrainData(
        height_grid=grid,
        grid_origin_xy=measured.grid_origin_xy,
        grid_resolution=measured.grid_resolution,
        mesh_vertices_w=measured.mesh_vertices_w,
        mesh_faces=measured.mesh_faces,
        env_origin_w=measured.env_origin_w,
    )
    terrain_ir = TerrainIR(terrain_type="generator", generator=None, physics_material={})
    with pytest.raises(ValueError, match="NaN"):
        build_terrain(terrain_ir, measured=bad)


# ---------------------------------------------------------------------------
# measured terrain: fine collision resampling
# ---------------------------------------------------------------------------

_MEASURED_IR = TerrainIR(terrain_type="generator", generator=None, physics_material={"static_friction": 1.0})


def _collision_heights(build: "TerrainBuild", x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Downward mj_ray heights of the compiled collision surface at (x, y)."""
    spec = mujoco.MjSpec()
    add_to_spec(spec, build)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geomid = np.zeros(1, dtype=np.int32)
    out = np.empty(len(x))
    for k in range(len(x)):
        dist = mujoco.mj_ray(model, data, np.array([x[k], y[k], 10.0]), np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
        out[k] = 10.0 - dist if dist >= 0 else np.nan
    return out


def test_measured_fine_grid_nodes_match_mesh_raycast():
    """Every node of the resampled fine collision grid carries the exact mesh-raycast
    height (agreement of the fine-grid surface with the recorded mesh at the nodes)."""
    measured = _step_measured_data(step_z=0.5, res=0.25)
    build = build_terrain(_MEASURED_IR, measured=measured, measured_collision_resolution=0.05)
    assert build.terrain_type == "measured"
    assert build.measured_collision_resolution == pytest.approx(0.05)
    assert build.measured_fine_window_w is None  # full extent: single hfield
    assert [g.name for g in build.geoms] == ["terrain_measured"]
    (asset,) = build.hfields
    nx, ny = asset.ncol, asset.nrow
    assert (nx, ny) == (41, 21)  # 2 m x 1 m extent at 0.05 m
    x_nodes = 0.05 * np.arange(nx)
    y_nodes = 0.05 * np.arange(ny)
    gx, gy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    assert build.measured_surface is not None
    expected = build.measured_surface.heights_w(gx, gy)
    assert not np.isnan(expected).any()
    np.testing.assert_allclose(asset.data_w.T, expected, atol=1e-12)
    # The compiled MuJoCo surface interpolates exactly those fine nodes (probed
    # off-node: an exact node hit is degenerate for mj_ray).
    tile = _interp_tile(expected, 0.05, 0.0, 0.0)
    rng = np.random.default_rng(7)
    x = rng.uniform(0.01, 1.99, 200)
    y = rng.uniform(0.01, 0.99, 200)
    zs = _collision_heights(build, x, y)
    np.testing.assert_allclose(zs, tile.heights_w(x, y), atol=1e-6)


def test_measured_fine_riser_sharpness_scales_with_resolution():
    """The step riser's collision ramp is exactly one cell wide, so the worst
    collision-vs-mesh deviation shrinks proportionally with the resolution."""
    measured = _step_measured_data(step_z=0.5, res=0.25)
    y = np.full(200, 0.5)
    x = np.linspace(0.01, 1.99, 200)
    mesh_z = np.where(x >= 1.0, 0.5, 0.0)  # exact surface (riser at x = 1)
    max_err = {}
    for res in (None, 0.05, 0.025):
        build = build_terrain(_MEASURED_IR, measured=measured, measured_collision_resolution=res)
        collision_z = _collision_heights(build, x, y)
        cell = 0.25 if res is None else res
        # The ramp lives in the one cell left of the riser; exact elsewhere.
        ramp = (x > 1.0 - cell) & (x < 1.0)
        np.testing.assert_allclose(collision_z[~ramp], mesh_z[~ramp], atol=1e-6)
        max_err[res] = float(np.abs(collision_z - mesh_z).max())
    assert max_err[0.05] < max_err[None]
    assert max_err[0.025] < max_err[0.05]


def test_measured_fine_window_pieces_cover_terrain():
    """A windowed fine build tiles the terrain: fine hfield inside the (snapped)
    window, recorded-resolution slabs outside, no gaps, shared edge nodes agree."""
    measured = _step_measured_data(step_z=0.5, res=0.25)
    build = build_terrain(
        _MEASURED_IR,
        measured=measured,
        measured_collision_resolution=0.05,
        measured_collision_window_w=(0.7, 1.3, 0.3, 0.7),
    )
    names = [g.name for g in build.geoms]
    assert names[0] == "terrain_measured_fine"
    assert set(names[1:]) == {
        "terrain_measured_xlo",
        "terrain_measured_xhi",
        "terrain_measured_ylo",
        "terrain_measured_yhi",
    }
    assert build.ground_geom_names == names
    # Window snapped outward to recorded nodes.
    assert build.measured_fine_window_w == pytest.approx((0.5, 1.5, 0.25, 0.75))
    # No gaps anywhere on the terrain; the riser cell is fine-resolution sharp
    # inside the window and record-resolution elsewhere.
    rng = np.random.default_rng(5)
    x = rng.uniform(0.01, 1.99, 400)
    y = rng.uniform(0.01, 0.99, 400)
    collision_z = _collision_heights(build, x, y)
    assert not np.isnan(collision_z).any()
    mesh_z = np.where(x >= 1.0, 0.5, 0.0)
    in_window = (x > 0.5) & (x < 1.5) & (y > 0.25) & (y < 0.75)
    ramp_fine = (x > 0.95) & (x < 1.0)
    ramp_coarse = (x > 0.75) & (x < 1.0)
    exact = (in_window & ~ramp_fine) | (~in_window & ~ramp_coarse)
    np.testing.assert_allclose(collision_z[exact], mesh_z[exact], atol=1e-6)
    # Shared window-edge nodes: fine and slab pieces carry identical heights.
    # HfieldAsset.data_w is (nrow=y, ncol=x); the fine grid's x = 0.5 edge column
    # holds y nodes at 0.05 spacing over [0.25, 0.75] (every 5th is a recorded
    # node), the xlo slab's last column holds the same edge at recorded y nodes.
    fine = build.hfields[0]
    xlo = next(h for h in build.hfields if h.name == "terrain_measured_xlo")
    np.testing.assert_allclose(fine.data_w[::5, 0], xlo.data_w[1:4, -1], atol=1e-12)


def test_measured_fine_validation_errors():
    measured = _step_measured_data()
    with pytest.raises(ValueError, match="positive"):
        build_terrain(_MEASURED_IR, measured=measured, measured_collision_resolution=-0.1)
    with pytest.raises(ValueError, match="requires collision_resolution"):
        build_terrain(_MEASURED_IR, measured=measured, measured_collision_window_w=(0.0, 1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="positive extent"):
        build_terrain(
            _MEASURED_IR,
            measured=measured,
            measured_collision_resolution=0.05,
            measured_collision_window_w=(1.0, 0.5, 0.0, 1.0),
        )
    with pytest.raises(ValueError, match="requires a measured terrain record"):
        build_terrain(_MEASURED_IR, measured_collision_resolution=0.05)


class TestMeasuredExactPrisms:
    """Exact-collision prisms: risers collide as true vertical walls."""

    @staticmethod
    def _stair_record():
        # Two treads (z=0 for x<0.5, z=0.3 beyond) plus the vertical riser quad.
        verts = np.array(
            [
                [-2, -2, 0],
                [0.5, -2, 0],
                [0.5, 2, 0],
                [-2, 2, 0],
                [0.5, -2, 0.3],
                [2, -2, 0.3],
                [2, 2, 0.3],
                [0.5, 2, 0.3],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [1, 4, 7], [1, 7, 2]], dtype=np.int32)
        res = 0.5
        xs = np.arange(-2, 2.01, res)
        grid = np.where(xs[:, None] >= 0.5, 0.3, 0.0) * np.ones((1, len(xs)))
        return MeasuredTerrainData(
            height_grid=grid.astype(np.float32),
            grid_origin_xy=(-2.0, -2.0),
            grid_resolution=res,
            mesh_vertices_w=verts,
            mesh_faces=faces,
            env_origin_w=np.zeros(3),
        )

    @staticmethod
    def _compile_with_sphere(build, pos):
        spec = mujoco.MjSpec()
        spec.option.timestep = 0.002
        add_to_spec(spec, build, set_robot_pair_friction=False)
        body = spec.worldbody.add_body()
        body.pos = list(pos)
        body.add_freejoint()
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
        geom.size = [0.04, 0, 0]
        geom.mass = 0.1
        return spec.compile()

    def test_vertical_faces_skipped_and_slabs_tile_the_rest(self):
        build = _build_measured(self._stair_record(), 1.0, "multiply", exact_window_w=(-1.5, 1.5, -1.5, 1.5))
        assert build.measured_prisms_w is not None
        assert build.measured_prisms_w.shape == (4, 6, 3)  # riser triangles carry no prism
        assert sorted(h.name.rsplit("_", 1)[-1] for h in build.hfields) == ["xhi", "xlo", "yhi", "ylo"]

    def test_sphere_rests_on_exact_tread_next_to_riser(self):
        # An hfield ramp would hold the sphere well above the low tread here.
        build = _build_measured(self._stair_record(), 1.0, "multiply", exact_window_w=(-1.5, 1.5, -1.5, 1.5))
        model = self._compile_with_sphere(build, (0.45, 0.0, 1.0))
        data = mujoco.MjData(model)
        for _ in range(3000):
            mujoco.mj_step(model, data)
        assert data.qpos[2] == pytest.approx(0.04, abs=5e-3)

    def test_rolling_sphere_is_blocked_by_the_riser_wall(self):
        # On the one-cell hfield ramp the sphere would climb to the high tread.
        build = _build_measured(self._stair_record(), 1.0, "multiply", exact_window_w=(-1.5, 1.5, -1.5, 1.5))
        model = self._compile_with_sphere(build, (0.2, 0.0, 0.045))
        data = mujoco.MjData(model)
        data.qvel[0] = 1.2
        for _ in range(3000):
            mujoco.mj_step(model, data)
        assert data.qpos[0] < 0.46  # rebounded, never crossed the wall
        assert data.qpos[2] < 0.1  # still on the low tread

    def test_plan_window_builds_and_counts_upward_triangles_only(self):
        # The stair mesh has 6 triangles but only 4 upward ones; the plan counts
        # what the build extrudes (an all-orientation count would hit the cap at
        # ~half the true prism count on closed-box generator meshes). The planned
        # window must round-trip through the build's own snapping unchanged.
        record = self._stair_record()
        win = plan_measured_exact_window(record, (-1.6, 1.6, -1.6, 1.6))
        assert win is not None
        build = _build_measured(record, 1.0, "multiply", exact_window_w=win)
        assert build.measured_prisms_w is not None
        assert build.measured_prisms_w.shape == (4, 6, 3)
        assert build.measured_fine_window_w == pytest.approx(win)

    def test_plan_over_cap_and_overhang_fall_back_instead_of_raising(self, monkeypatch, capsys):
        import lab2mj.terrain as terrain_module

        monkeypatch.setattr(terrain_module, "MEASURED_EXACT_PRISM_CAP", 3)
        assert plan_measured_exact_window(self._stair_record(), (-1.5, 1.5, -1.5, 1.5)) is None
        assert "falling back to hfield" in capsys.readouterr().out

        record = self._stair_record()
        ceiling = record.mesh_vertices_w[:4] + np.array([0, 0, 1.0], dtype=np.float32)
        record.mesh_vertices_w = np.concatenate([record.mesh_vertices_w, ceiling])
        record.mesh_faces = np.concatenate([record.mesh_faces, np.array([[8, 10, 9]], dtype=np.int32)])
        assert plan_measured_exact_window(record, (-1.5, 1.5, -1.5, 1.5)) is None
        assert "falling back to hfield" in capsys.readouterr().out

    def test_overhang_raises_but_occluded_undersides_pass(self):
        # A downward face coplanar with (or under) the surface is an occluded box
        # underside and is skipped; a downward face ABOVE the surface is a real
        # overhang and must raise.
        record = self._stair_record()
        underside = record.mesh_vertices_w[:4] - np.array([0, 0, 0.2], dtype=np.float32)
        record.mesh_vertices_w = np.concatenate([record.mesh_vertices_w, underside])
        record.mesh_faces = np.concatenate([record.mesh_faces, np.array([[8, 10, 9]], dtype=np.int32)])
        build = _build_measured(record, 1.0, "multiply", exact_window_w=(-1.5, 1.5, -1.5, 1.5))
        assert build.measured_prisms_w is not None
        assert build.measured_prisms_w.shape == (4, 6, 3)

        record = self._stair_record()
        ceiling = record.mesh_vertices_w[:4] + np.array([0, 0, 1.0], dtype=np.float32)
        record.mesh_vertices_w = np.concatenate([record.mesh_vertices_w, ceiling])
        record.mesh_faces = np.concatenate([record.mesh_faces, np.array([[8, 10, 9]], dtype=np.int32)])
        with pytest.raises(ValueError, match="overhanging"):
            _build_measured(record, 1.0, "multiply", exact_window_w=(-1.5, 1.5, -1.5, 1.5))
