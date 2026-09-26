"""TerrainIR -> MuJoCo scene pieces (ground plane, procedural terrain tiles, flat patches).

Mirrors IsaacLab v2.3.0 semantics:

* ``isaaclab.terrains.terrain_importer.TerrainImporter`` -- env-origin layout
  (grid origins for ``plane``, curriculum origins for ``generator``).
* ``isaaclab.terrains.terrain_generator.TerrainGenerator`` -- sub-terrain type +
  difficulty schedule (replicated exactly: IsaacLab draws these from
  ``np.random.default_rng(cfg.seed)`` in a fixed order), per-tile placement, the
  surrounding border ring, and the generator-level ``size``/``horizontal_scale``/
  ``vertical_scale`` overrides it stamps onto every sub-terrain cfg.
* ``isaaclab.terrains.height_field.hf_terrains`` -- ``random_uniform_terrain`` and
  ``pyramid_sloped_terrain`` (upright + inverted) reproduce the exact discrete height
  grid (same ``int()`` unit conversions, border pixels, spline upsampling, ``np.rint``
  rounding) and emit it as a MuJoCo ``hfield`` asset per tile.
* ``isaaclab.terrains.trimesh.mesh_terrains`` -- ``repeated_objects_terrain``,
  ``pyramid_stairs_terrain``, ``inverted_pyramid_stairs_terrain`` and
  ``random_grid_terrain`` reproduce the exact box layout (step rings, grid cells,
  borders, platforms) as MuJoCo box geoms.
* ``isaaclab.terrains.utils.find_flat_patches`` -- flat-patch validity semantics,
  computed analytically on the primitive/grid layout instead of by raycast.

RNG parity caveat (documented, verified against the v2.3.0 source): IsaacLab samples
*randomized content* from process-global RNG state, not from the generator's seeded
``np_rng``: object placements and height noise in ``repeated_objects_terrain`` and the
height grid of ``random_uniform_terrain`` use the global ``np.random`` module, and the
cell heights of ``random_grid_terrain`` use the global torch RNG. The global state at
terrain-build time depends on every earlier consumer in the Isaac process, so exact
draw-for-draw parity is impossible from the config alone. This module instead
guarantees:

* exact parity of the per-tile sub-terrain type and difficulty schedule (same
  ``default_rng(seed)`` consumption order as ``TerrainGenerator``),
* structural parity of each tile: same object/cell counts, interpolated dimensions
  and heights, same noise bounds and discrete height lattice, same platform-clear
  region, same tile origins,
* same-seed => identical layout across builds of this module (randomized per-tile
  content uses ``np.random.default_rng([seed, row, col])``).

Height-field tiles (hfield mapping)
-----------------------------------
The discrete grid ``heights[i, j]`` (node ``i`` along x, ``j`` along y, spacing
``horizontal_scale``, integer multiples of ``vertical_scale``) is IsaacLab's exact
array. MuJoCo normalizes an hfield's elevation data to [0, 1] at compile time
(min -> 0, max -> 1) and renders node z as ``geom_pos_z + normalized * size[2]``; the
asset therefore carries the raw metric heights with ``size[2] = max - min`` and the
geom z position at the grid minimum, so every node sits at its exact world height
(elevation zero reference: the tile's ground level z = 0 maps to raw height 0).
IsaacLab's ``slope_threshold`` correction only shifts mesh-vertex x/y positions to
turn steep cell faces vertical -- it never changes the height grid -- so the hfield
data is unaffected; near such cliffs the MuJoCo surface (and :meth:`height lookup
<TileBuild.heights_w>`) is a one-cell-wide steep ramp where Isaac's mesh is a
vertical wall (difference bounded by one ``horizontal_scale`` cell horizontally).
Height lookups interpolate over the same two-triangle cell split (diagonal from node
``(i, j)`` to ``(i+1, j+1)``) that both IsaacLab's height-field mesh and MuJoCo's
hfield collider/raycaster use.

Ground representation: generator terrain emits no infinite plane -- each tile carries
its own ground (base box, hfield, step/cell boxes) plus the generator's border ring,
exactly like IsaacLab's finite terrain mesh; only ``terrain_type == "plane"`` uses a
MuJoCo plane.

Measured terrain (geometric identity with a reference dump)
------------------------------------------------------------
The procedural generator path above gives *statistical* parity only: tile content
drawn from IsaacLab's global RNG streams produces a different terrain instance, so
strict trajectory gates against an Isaac reference dump cannot pass on it. When the
reference dump carries a measured-terrain record (``terrain_mesh_vertices`` /
``terrain_height_grid`` keys written by ``lab2mj/isaac/dump_reference.py``),
:func:`build_terrain` accepts it via ``measured=`` and builds the terrain the
reference robot actually walked on instead of regenerating procedurally:

* collision: one MuJoCo ``hfield`` spanning the full terrain (tiles + border) whose
  nodes are the measured height grid (vertical raycast of Isaac's terrain mesh at
  the generator's ``horizontal_scale``). Vertical faces of the Isaac mesh (stair
  risers, grid-cell walls) become one-cell-wide steep ramps, the same bounded
  deviation the procedural height-field tiles already document. Passing
  ``measured_collision_resolution`` re-raycasts the recorded mesh onto a finer
  collision grid, shrinking those ramps to one *fine* cell; with
  ``measured_collision_window_w`` the fine grid covers only that xy window (snapped
  outward to recorded-grid nodes) and four recorded-resolution hfield slabs tile
  the rest of the terrain, keeping the total node count (and the elevation string
  MuJoCo's ``MjSpec.to_xml`` embeds in ``scene.xml``) bounded.
  ``measured_exact_window_w`` instead collides the exact recorded mesh inside the
  window — one convex prism per upward triangle, so risers are true vertical
  walls (the converter's default, via ``--terrain_exact``);
* height lookups (:meth:`TerrainBuild.height_at`, feeding ``height_scan``): exact
  vertical raycast against the measured triangle mesh itself
  (:class:`MeasuredSurface`), reproducing IsaacLab's warp raycast including
  discontinuities the interpolated grid cannot represent;
* env origin 0 comes from the dump (``env_origin_w``), tile origins from the
  importer's ``terrain_origins``.

The measured build's flat-patch samplers are unavailable (the measured record
carries no sub-terrain cfg). Without ``measured=`` the procedural path is used
and strict geometric identity is NOT guaranteed.

Friction mapping (PhysX -> MuJoCo)
----------------------------------
PhysX combines the two materials' friction per its ``friction_combine_mode``
("multiply" in the contact_lab / stock velocity tasks): mu_pair = mu_ground * mu_foot.
MuJoCo combines contact friction as the elementwise **max** of the two geoms' friction
vectors -- unless one geom has a higher ``priority``, in which case that geom's
friction is used verbatim (MJCF reference, ``geom/priority``; ``mj_contactParam``).

The ground alone therefore cannot express PhysX's multiply rule. The mapping used here:

* the ground geoms carry ``mu_ground`` (``static_friction``) at priority 0;
* :func:`add_to_spec` stamps every pre-existing robot collision geom with priority 1
  and replaces its slide friction with the PhysX-combined pair value
  ``pair_friction(mu_foot, mu_ground, mode)`` -- so every robot-ground contact uses
  exactly the PhysX pair friction. With mu_ground = 1.0 and mode "multiply" (both
  committed fixtures) this reduces to a pure priority bump: pair = mu_foot.
* self-collision robots (``robot_self_pair``): terrain geoms move UP to priority 2
  carrying the precomputed robot-ground pair value (they win every robot-ground
  contact) and robot geoms carry the precomputed robot-ROBOT pair value at
  priority 1 (the equal-priority max resolves every self contact to it) — both
  PhysX pairings exact. This needs a uniform robot material and no material DR
  event (the DR event rewrites robot-geom friction per draw and relies on robot
  geoms winning); with material DR the converter keeps the ground-pair stamping
  and warns that self contacts resolve at the wrong value.

PhysX distinguishes static/dynamic friction; MuJoCo has a single Coulomb coefficient.
``static_friction`` is used: it reproduces the no-slip breakaway threshold (the
dominant regime for stance feet). The base terrain materials of both committed
fixtures set static == dynamic, but the material DR event samples the columns
independently — slip-phase forces then follow the static draw, which
``lab2mj.events`` warns about at construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np
import scipy.interpolate as interpolate

from lab2mj.env_yaml import class_name
from lab2mj.ir import TerrainIR

_GROUND_GEOM_NAME = "ground"
_PLATFORM_CLEARANCE = 0.1  # repeated_objects_terrain constant
_TILE_BASE_HALF_THICKNESS = 0.05  # flat/repeated-objects tile ground box (top at z = 0)
_RANDOM_GRID_TERRAIN_HEIGHT = 1.0  # random_grid_terrain constant (cell box depth)
# find_flat_patches queries a ring of linspace(0, 2*pi, 10) angles per radius.
_PATCH_RING_ANGLES = np.linspace(0.0, 2.0 * np.pi, 10)
_BORDER_SIDES = ("left", "right", "top", "bottom")  # make_border mesh order

_MJ_GEOM_TYPE = {
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "hfield": mujoco.mjtGeom.mjGEOM_HFIELD,
}


def map_size_from_generator(generator: dict[str, Any]) -> tuple[float, float]:
    """Full terrain map size (width, height) from a terrain-generator cfg dict."""
    grid_width, grid_length = generator["size"]
    border_width = float(generator.get("border_width", 0.0))
    map_width = generator["num_rows"] * float(grid_width) + 2.0 * border_width
    map_height = generator["num_cols"] * float(grid_length) + 2.0 * border_width
    return (map_width, map_height)


def pair_friction(mu_robot: float, mu_ground: float, mode: str) -> float:
    """PhysX pair friction for the given ``friction_combine_mode``."""
    if mode == "multiply":
        return mu_robot * mu_ground
    if mode == "average":
        return 0.5 * (mu_robot + mu_ground)
    if mode == "min":
        return min(mu_robot, mu_ground)
    if mode == "max":
        return max(mu_robot, mu_ground)
    raise ValueError(f"unknown friction_combine_mode '{mode}'")


@dataclass(frozen=True)
class TerrainGeom:
    """One MuJoCo primitive of the terrain (world frame)."""

    name: str
    kind: Literal["plane", "box", "cylinder", "hfield"]
    pos_w: tuple[float, float, float]
    # MuJoCo size convention: plane (0, 0, spacing), box half-sizes, cylinder (radius, half-height, 0).
    # hfield geoms take their extents from the referenced asset; size is unused.
    size: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    friction_slide: float = 1.0
    hfield: str = ""  # asset name, kind == "hfield" only


@dataclass(eq=False)
class HfieldAsset:
    """One MuJoCo hfield asset carrying a height-field tile's raw metric grid.

    ``data_w`` is (nrow, ncol) row-major with row = y (ascending) and col = x
    (ascending), in meters (world z; the tile ground level is 0). MuJoCo's compiler
    normalizes the data to [0, 1] (min -> 0, max -> 1); ``size[2] = max - min`` and
    the geom's z position at the grid minimum recover the exact metric heights.
    """

    name: str
    nrow: int
    ncol: int
    size: tuple[float, float, float, float]  # (radius_x, radius_y, elevation_z, base_z)
    data_w: np.ndarray


@dataclass(frozen=True)
class _Obstacle:
    """Analytic footprint of one placed object (world frame, z base at 0)."""

    kind: Literal["box", "cylinder"]
    x_w: float
    y_w: float
    top_z_w: float
    radius: float = 0.0  # cylinder
    half_x: float = 0.0  # box (local axes)
    half_y: float = 0.0
    yaw: float = 0.0

    def covers(self, x_w: np.ndarray, y_w: np.ndarray) -> np.ndarray:
        dx, dy = x_w - self.x_w, y_w - self.y_w
        if self.kind == "cylinder":
            return dx * dx + dy * dy <= self.radius * self.radius
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return np.logical_and(np.abs(lx) <= self.half_x, np.abs(ly) <= self.half_y)


@dataclass
class TileBuild:
    """One generated sub-terrain tile."""

    row: int
    col: int
    sub_terrain: str
    difficulty: float
    origin_w: tuple[float, float, float]
    size: tuple[float, float]
    platform_width: float = 0.0
    platform_top_z_w: float = 0.0
    obstacles: list[_Obstacle] = field(default_factory=list)
    # patch-set name -> flat-patch sampling cfg dict (num_patches, patch_radius, ...).
    flat_patch_cfgs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Surface model for height lookups. "objects": flat ground at z = 0 plus platform
    # and obstacles. "hfield": discrete height grid (node (i, j) at grid_x0_w + i *
    # grid_step, grid_y0_w + j * grid_step). "boxes": union of axis-aligned boxes
    # (surface_boxes_w rows: x_lo, x_hi, y_lo, y_hi, top_z).
    surface_kind: Literal["objects", "hfield", "boxes"] = "objects"
    grid_z_w: np.ndarray | None = None  # (nx, ny) node heights, world z
    grid_x0_w: float = 0.0
    grid_y0_w: float = 0.0
    grid_step: float = 0.0
    surface_boxes_w: np.ndarray | None = None  # (M, 5)

    def heights_w(self, x_w: np.ndarray, y_w: np.ndarray) -> np.ndarray:
        """Downward-raycast heights at (x, y): highest covering surface, NaN where none.

        Matches raycasting the tile from above: the first hit is the highest surface.
        Off-tile queries (and on-tile holes, e.g. ``holes=True`` stair tiles) return
        NaN — except where one of THIS tile's obstacles overhangs the border
        (repeated-objects centers sample the full tile extent, so footprints reach
        past it): the overhang is surface, exactly as in the merged Isaac mesh. For
        hfield tiles the lookup interpolates the discrete grid over the same
        two-triangle cell split MuJoCo's hfield uses (see module docstring); this is
        the collision surface, not Isaac's slope-corrected mesh.
        """
        x_w, y_w = np.broadcast_arrays(np.atleast_1d(np.asarray(x_w, np.float64)), np.asarray(y_w, np.float64))
        out = np.full(x_w.shape, np.nan)
        on_tile = np.logical_and(
            np.abs(x_w - self.origin_w[0]) <= 0.5 * self.size[0],
            np.abs(y_w - self.origin_w[1]) <= 0.5 * self.size[1],
        )
        if self.surface_kind == "objects":
            heights = np.zeros(x_w.shape)
            if self.platform_top_z_w > 0.0:
                on_platform = np.logical_and(
                    np.abs(x_w - self.origin_w[0]) <= 0.5 * self.platform_width,
                    np.abs(y_w - self.origin_w[1]) <= 0.5 * self.platform_width,
                )
                heights = np.where(on_platform, self.platform_top_z_w, heights)
            out[on_tile] = heights[on_tile]
            for obstacle in self.obstacles:
                # `~(out >= top)` keeps NaN (off-tile overhang) and lower surfaces.
                covered = obstacle.covers(x_w, y_w) & ~(out >= obstacle.top_z_w)
                out[covered] = obstacle.top_z_w
        elif self.surface_kind == "hfield":
            assert self.grid_z_w is not None
            nx, ny = self.grid_z_w.shape
            u = (x_w - self.grid_x0_w) / self.grid_step
            v = (y_w - self.grid_y0_w) / self.grid_step
            # The grid may span slightly less than the tile (int truncation of
            # size / horizontal_scale); off-grid slivers are uncovered, like Isaac's mesh.
            on_grid = (u >= 0.0) & (u <= nx - 1) & (v >= 0.0) & (v <= ny - 1) & on_tile
            i = np.clip(np.floor(u), 0, nx - 2).astype(np.intp)
            j = np.clip(np.floor(v), 0, ny - 2).astype(np.intp)
            fu = np.clip(u - i, 0.0, 1.0)
            fv = np.clip(v - j, 0.0, 1.0)
            z00 = self.grid_z_w[i, j]
            z10 = self.grid_z_w[i + 1, j]
            z01 = self.grid_z_w[i, j + 1]
            z11 = self.grid_z_w[i + 1, j + 1]
            upper = fv >= fu  # triangle above the (i, j) -> (i+1, j+1) diagonal
            heights = np.where(
                upper,
                z00 + fu * (z11 - z01) + fv * (z01 - z00),
                z00 + fu * (z10 - z00) + fv * (z11 - z10),
            )
            out[on_grid] = heights[on_grid]
        else:
            assert self.surface_boxes_w is not None
            boxes = self.surface_boxes_w
            covered = (
                (x_w[..., None] >= boxes[:, 0])
                & (x_w[..., None] <= boxes[:, 1])
                & (y_w[..., None] >= boxes[:, 2])
                & (y_w[..., None] <= boxes[:, 3])
            )
            tops = np.where(covered, boxes[:, 4], -np.inf).max(axis=-1)
            hit = np.isfinite(tops) & on_tile
            out[hit] = tops[hit]
        return out

    def height_w(self, x_w: float, y_w: float) -> float | None:
        """Scalar :meth:`heights_w`; None off-tile / over holes."""
        height = float(self.heights_w(np.asarray([x_w]), np.asarray([y_w]))[0])
        return None if math.isnan(height) else height

    def obstacle_overhang_w(self) -> float:
        """Greatest distance any obstacle footprint extends beyond the tile bounds."""
        reach = 0.0
        for ob in self.obstacles:
            r = ob.radius if ob.kind == "cylinder" else math.hypot(ob.half_x, ob.half_y)
            reach = max(
                reach,
                abs(ob.x_w - self.origin_w[0]) + r - 0.5 * self.size[0],
                abs(ob.y_w - self.origin_w[1]) + r - 0.5 * self.size[1],
            )
        return max(reach, 0.0)


class FlatPatchSampler:
    """Analytic equivalent of ``isaaclab.terrains.utils.find_flat_patches``.

    Rejection-samples patch centers per tile; a patch is valid iff every ring query
    point (10 angles per configured radius, matching ``linspace(0, 2*pi, 10)``) hits
    the tile and its height lies inside ``z_range`` (shifted by the tile-origin z)
    with max-min height spread <= ``max_height_diff``. Heights come from
    :meth:`TileBuild.heights_w` instead of a warp raycast -- exact for primitive
    layouts, and matching the MuJoCo collision surface (not Isaac's slope-corrected
    mesh) within one grid cell on height-field tiles. Returns world-frame positions
    with z at the surface height, matching IsaacLab's ``flat_patches`` after the
    terrain-origin shift.

    Only tiles whose sub-terrain configures this patch set contribute (IsaacLab
    leaves other tiles' rows as zeros in its dense tensor).
    """

    def __init__(self, name: str, tiles: list[TileBuild], default_seed: int) -> None:
        self.name = name
        self._tiles = [tile for tile in tiles if name in tile.flat_patch_cfgs]
        self._default_seed = default_seed

    def __call__(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample all configured patches; returns (K, 3) world-frame positions."""
        rng = np.random.default_rng(self._default_seed) if rng is None else rng
        patches_w: list[np.ndarray] = []
        for tile in self._tiles:
            patches_w.append(self._sample_tile(tile, tile.flat_patch_cfgs[self.name], rng))
        if not patches_w:
            return np.zeros((0, 3), dtype=np.float64)
        return np.concatenate(patches_w, axis=0)

    def _sample_tile(self, tile: TileBuild, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
        num_patches = int(cfg["num_patches"])
        radii = cfg["patch_radius"]
        radii = [float(radii)] if isinstance(radii, (int, float)) else [float(r) for r in radii]
        # Sampling ranges are relative to the tile origin, clamped to the tile bounds
        # (find_flat_patches clamps to the mesh bounding box; any center whose ring
        # leaves the tile is rejected anyway, so the tile footprint is equivalent).
        x_lo = max(float(cfg["x_range"][0]) + tile.origin_w[0], tile.origin_w[0] - 0.5 * tile.size[0])
        x_hi = min(float(cfg["x_range"][1]) + tile.origin_w[0], tile.origin_w[0] + 0.5 * tile.size[0])
        y_lo = max(float(cfg["y_range"][0]) + tile.origin_w[1], tile.origin_w[1] - 0.5 * tile.size[1])
        y_hi = min(float(cfg["y_range"][1]) + tile.origin_w[1], tile.origin_w[1] + 0.5 * tile.size[1])
        z_lo = float(cfg["z_range"][0]) + tile.origin_w[2]
        z_hi = float(cfg["z_range"][1]) + tile.origin_w[2]
        max_height_diff = float(cfg["max_height_diff"])

        ring_dx = np.concatenate([r * np.cos(_PATCH_RING_ANGLES) for r in radii])
        ring_dy = np.concatenate([r * np.sin(_PATCH_RING_ANGLES) for r in radii])

        patches_w = np.zeros((num_patches, 3), dtype=np.float64)
        found = 0
        for _ in range(10000 * num_patches):
            if found >= num_patches:
                break
            x_w = rng.uniform(x_lo, x_hi)
            y_w = rng.uniform(y_lo, y_hi)
            values = tile.heights_w(x_w + ring_dx, y_w + ring_dy)
            if np.isnan(values).any():
                continue
            if values.min() < z_lo or values.max() > z_hi:
                continue
            if values.max() - values.min() > max_height_diff:
                continue
            patches_w[found] = (x_w, y_w, values[-1])
            found += 1
        if found < num_patches:
            raise RuntimeError(
                f"flat-patch sampling failed for '{self.name}' on tile ({tile.row}, {tile.col}): "
                f"found {found}/{num_patches} valid patches"
            )
        return patches_w


@dataclass(eq=False)
class MeasuredTerrainData:
    """Measured terrain captured from a running Isaac env (world frame).

    ``height_grid[i, j]`` is the surface height at ``grid_origin_xy + (i, j) *
    grid_resolution`` (vertical raycast of the terrain mesh; no NaNs).
    ``mesh_vertices_w``/``mesh_faces`` are the exact triangle mesh IsaacLab's
    RayCaster casts against (float32, as warp stores it).
    """

    height_grid: np.ndarray  # (nx, ny) float32
    grid_origin_xy: tuple[float, float]
    grid_resolution: float
    mesh_vertices_w: np.ndarray  # (V, 3) float32
    mesh_faces: np.ndarray  # (F, 3) int32
    env_origin_w: np.ndarray  # (3,) env-0 origin, world frame
    tile_origins_w: np.ndarray | None = None  # (rows, cols, 3)

    _KEYS = (
        "terrain_height_grid",
        "terrain_grid_origin_xy",
        "terrain_grid_resolution",
        "terrain_mesh_vertices",
        "terrain_mesh_faces",
        "env_origin_w",
    )

    @classmethod
    def from_npz(cls, data: Any) -> MeasuredTerrainData | None:
        """Parse a measured-terrain record from a dump / bundle npz mapping.

        Returns None when the mapping carries no record (plane-terrain dumps,
        pre-measured-capture dumps); raises if the record is incomplete.
        """
        if "terrain_height_grid" not in data:
            return None
        missing = [key for key in cls._KEYS if key not in data]
        if missing:
            raise ValueError(f"measured-terrain record is incomplete: missing npz keys {missing}")
        origin = np.asarray(data["terrain_grid_origin_xy"], dtype=np.float64)
        tile_origins = (
            np.asarray(data["terrain_tile_origins"], dtype=np.float64) if "terrain_tile_origins" in data else None
        )
        return cls(
            height_grid=np.asarray(data["terrain_height_grid"], dtype=np.float32),
            grid_origin_xy=(float(origin[0]), float(origin[1])),
            grid_resolution=float(data["terrain_grid_resolution"]),
            mesh_vertices_w=np.asarray(data["terrain_mesh_vertices"], dtype=np.float32),
            mesh_faces=np.asarray(data["terrain_mesh_faces"], dtype=np.int32),
            env_origin_w=np.asarray(data["env_origin_w"], dtype=np.float64).reshape(3),
            tile_origins_w=tile_origins,
        )

    def save_npz(self, path: Any) -> None:
        arrays: dict[str, np.ndarray] = {
            "terrain_height_grid": np.asarray(self.height_grid, dtype=np.float32),
            "terrain_grid_origin_xy": np.asarray(self.grid_origin_xy, dtype=np.float64),
            "terrain_grid_resolution": np.asarray(self.grid_resolution, dtype=np.float64),
            "terrain_mesh_vertices": np.asarray(self.mesh_vertices_w, dtype=np.float32),
            "terrain_mesh_faces": np.asarray(self.mesh_faces, dtype=np.int32),
            "env_origin_w": np.asarray(self.env_origin_w, dtype=np.float64),
        }
        if self.tile_origins_w is not None:
            arrays["terrain_tile_origins"] = np.asarray(self.tile_origins_w, dtype=np.float32)
        np.savez_compressed(path, **arrays)


# Barycentric edge tolerance of MeasuredSurface: points on a shared edge of two
# coplanar triangles get the same height from either; on a discontinuity edge the
# max over hits picks the top surface, like a downward ray's first hit.
_BARY_EPS = 1.0e-9


class MeasuredSurface:
    """Exact vertical-ray heights over a captured triangle mesh (world frame).

    Triangles are bucketed on a uniform xy grid so each query only tests local
    candidates; a query returns the highest covering triangle's height at (x, y)
    (the first hit of a downward ray), NaN where no triangle covers the point.
    xy-degenerate triangles (vertical walls) are dropped: a vertical ray only
    grazes them on a measure-zero edge.
    """

    def __init__(self, vertices_w: np.ndarray, faces: np.ndarray, cell: float = 0.2) -> None:
        if cell <= 0.0:
            raise ValueError(f"bucket cell size must be positive, got {cell}")
        vertices_w = np.asarray(vertices_w, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.intp)
        p0 = vertices_w[faces[:, 0]]
        p1 = vertices_w[faces[:, 1]]
        p2 = vertices_w[faces[:, 2]]
        e1 = p1 - p0
        e2 = p2 - p0
        det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
        keep = np.abs(det) > 1.0e-12
        p0, e1, e2, det = p0[keep], e1[keep], e2[keep], det[keep]
        if p0.shape[0] == 0:
            raise ValueError("terrain mesh has no xy-covering (non-vertical) triangles")
        self._p0 = p0
        self._e1z = e1[:, 2]
        self._e2z = e2[:, 2]
        # Inverse 2x2 xy-edge matrix rows: bary a = m00*dx + m01*dy, b = m10*dx + m11*dy.
        self._m00 = e2[:, 1] / det
        self._m01 = -e2[:, 0] / det
        self._m10 = -e1[:, 1] / det
        self._m11 = e1[:, 0] / det

        corners_x = np.stack([p0[:, 0], p0[:, 0] + e1[:, 0], p0[:, 0] + e2[:, 0]])
        corners_y = np.stack([p0[:, 1], p0[:, 1] + e1[:, 1], p0[:, 1] + e2[:, 1]])
        lo_x, hi_x = corners_x.min(axis=0), corners_x.max(axis=0)
        lo_y, hi_y = corners_y.min(axis=0), corners_y.max(axis=0)
        self._cell = float(cell)
        self._x0 = float(lo_x.min())
        self._y0 = float(lo_y.min())
        self._nbx = int(np.floor((hi_x.max() - self._x0) / self._cell)) + 1
        self._nby = int(np.floor((hi_y.max() - self._y0) / self._cell)) + 1
        ix0 = np.clip(np.floor((lo_x - self._x0) / self._cell).astype(np.intp), 0, self._nbx - 1)
        iy0 = np.clip(np.floor((lo_y - self._y0) / self._cell).astype(np.intp), 0, self._nby - 1)
        ix1 = np.clip(np.floor((hi_x - self._x0) / self._cell).astype(np.intp), 0, self._nbx - 1)
        iy1 = np.clip(np.floor((hi_y - self._y0) / self._cell).astype(np.intp), 0, self._nby - 1)
        # One (triangle, bucket) entry per bucket the triangle's xy bbox overlaps.
        wx = ix1 - ix0 + 1
        wy = iy1 - iy0 + 1
        counts = wx * wy
        tri_of_entry = np.repeat(np.arange(counts.shape[0], dtype=np.intp), counts)
        local = np.arange(int(counts.sum()), dtype=np.intp) - np.repeat(np.cumsum(counts) - counts, counts)
        bucket = (
            (ix0[tri_of_entry] + local % wx[tri_of_entry]) * self._nby + iy0[tri_of_entry] + (local // wx[tri_of_entry])
        )
        order = np.argsort(bucket, kind="stable")
        self._bucket_tris = tri_of_entry[order].astype(np.int32)
        self._bucket_starts = np.searchsorted(bucket[order], np.arange(self._nbx * self._nby + 1))

    def heights_w(self, x_w: Any, y_w: Any) -> np.ndarray:
        """Highest covering triangle height at each (x, y); NaN where none covers."""
        x_arr, y_arr = np.broadcast_arrays(
            np.atleast_1d(np.asarray(x_w, dtype=np.float64)), np.asarray(y_w, dtype=np.float64)
        )
        xq = x_arr.reshape(-1)
        yq = y_arr.reshape(-1)
        out = np.full(xq.shape, -np.inf)
        bx = np.floor((xq - self._x0) / self._cell).astype(np.intp)
        by = np.floor((yq - self._y0) / self._cell).astype(np.intp)
        covered = (bx >= 0) & (bx < self._nbx) & (by >= 0) & (by < self._nby)
        bucket = np.where(covered, bx * self._nby + by, 0)
        start = np.where(covered, self._bucket_starts[bucket], 0)
        count = np.where(covered, self._bucket_starts[bucket + 1] - start, 0)
        total = int(count.sum())
        if total > 0:
            qi = np.repeat(np.arange(xq.shape[0], dtype=np.intp), count)
            pos = np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(count) - count, count) + np.repeat(start, count)
            tri = self._bucket_tris[pos]
            dx = xq[qi] - self._p0[tri, 0]
            dy = yq[qi] - self._p0[tri, 1]
            a = self._m00[tri] * dx + self._m01[tri] * dy
            b = self._m10[tri] * dx + self._m11[tri] * dy
            hit = (a >= -_BARY_EPS) & (b >= -_BARY_EPS) & (a + b <= 1.0 + _BARY_EPS)
            z = self._p0[tri, 2] + a * self._e1z[tri] + b * self._e2z[tri]
            np.maximum.at(out, qi[hit], z[hit])
        out[~np.isfinite(out)] = np.nan
        return out.reshape(x_arr.shape)


@dataclass
class TerrainBuild:
    """Everything the scene composer needs from the terrain.

    The friction/priority mapping of the module docstring is mandatory:
    :func:`add_to_spec` stamps ground geoms and robot collision geoms with it
    automatically.
    """

    terrain_type: str
    geoms: list[TerrainGeom]
    ground_geom_names: list[str]
    env_origins_w: np.ndarray  # (E, 3)
    flat_patches: dict[str, FlatPatchSampler]
    friction_combine_mode: str
    ground_friction: float
    terrain_origins_w: np.ndarray | None = None  # (num_rows, num_cols, 3) for generator terrain
    tiles: list[TileBuild] = field(default_factory=list)
    hfields: list[HfieldAsset] = field(default_factory=list)
    # Generator grid layout (height_at metadata; None/0 for plane terrain).
    tile_size: tuple[float, float] | None = None
    grid_shape: tuple[int, int] | None = None
    border_width_w: float = 0.0
    border_top_z_w: float = 0.0
    # Exact mesh-raycast surface of a measured build; when set, height_at uses it
    # (terrain_type == "measured").
    measured_surface: MeasuredSurface | None = None
    # Collision-grid resolution of a measured build (== the record's grid resolution
    # unless the build resampled the mesh onto a finer collision grid).
    measured_collision_resolution: float | None = None
    # Snapped world-frame (x_lo, x_hi, y_lo, y_hi) of the fine collision window of a
    # measured build; None when the collision grid has a single resolution.
    measured_fine_window_w: tuple[float, float, float, float] | None = None
    # Exact-collision prisms of a measured build: (P, 6, 3) world-frame vertices, one
    # upward mesh triangle extruded to a common base per prism (see _build_measured).
    # None when the build uses hfield collision inside the window.
    measured_prisms_w: np.ndarray | None = None
    # Lazy (row, col) -> tile lookup for height_at (built on first query).
    _tiles_by_rc: dict[tuple[int, int], TileBuild] | None = field(default=None, repr=False)
    # Lazy [(tile, overhang margin)] of tiles whose obstacles reach past their bounds.
    _tile_overhangs: list[tuple[TileBuild, float]] | None = field(default=None, repr=False)

    def robot_pair_friction(self, mu_robot: float) -> float:
        """PhysX pair (slide) friction a robot collision geom must carry at priority 1."""
        return pair_friction(mu_robot, self.ground_friction, self.friction_combine_mode)

    def tile_at(self, row: int, col: int) -> TileBuild:
        """The tile at grid position (row, col); generator terrain only."""
        for tile in self.tiles:
            if tile.row == row and tile.col == col:
                return tile
        raise KeyError(f"no tile at ({row}, {col})")

    def height_at(self, x_w: Any, y_w: Any) -> Any:
        """Terrain surface height under (x, y): highest surface a downward ray hits.

        Vectorized over broadcastable array inputs; scalar in -> float out. Returns
        0.0 everywhere for ``plane`` terrain. For ``generator`` terrain the query is
        routed to the covering tile (boundary points exactly between tiles resolve to
        the higher row/col index; all supported tile types meet at their shared border
        height there) plus any neighboring tile whose obstacles overhang the border
        (highest surface wins, like Isaac's raycast against the merged mesh), the
        generator border ring returns its top, and queries beyond the terrain return
        NaN (a ray there misses, as in IsaacLab). ``measured`` terrain raycasts the
        captured Isaac mesh exactly (see module docstring).
        """
        x_arr = np.asarray(x_w, dtype=np.float64)
        y_arr = np.asarray(y_w, dtype=np.float64)
        scalar = x_arr.ndim == 0 and y_arr.ndim == 0
        x_arr, y_arr = np.broadcast_arrays(np.atleast_1d(x_arr), np.atleast_1d(y_arr))
        if self.measured_surface is not None:
            out = self.measured_surface.heights_w(x_arr, y_arr)
            return float(out.flat[0]) if scalar else out
        out = np.full(x_arr.shape, np.nan)
        if self.terrain_type == "plane":
            out[:] = 0.0
            return float(out.flat[0]) if scalar else out

        assert self.tile_size is not None and self.grid_shape is not None
        size_x, size_y = self.tile_size
        num_rows, num_cols = self.grid_shape
        x_min, x_max = -0.5 * num_rows * size_x, 0.5 * num_rows * size_x
        y_min, y_max = -0.5 * num_cols * size_y, 0.5 * num_cols * size_y
        on_grid = (x_arr >= x_min) & (x_arr <= x_max) & (y_arr >= y_min) & (y_arr <= y_max)
        rows = np.clip(np.floor((x_arr - x_min) / size_x), 0, num_rows - 1).astype(np.intp)
        cols = np.clip(np.floor((y_arr - y_min) / size_y), 0, num_cols - 1).astype(np.intp)
        if self._tiles_by_rc is None:
            self._tiles_by_rc = {(tile.row, tile.col): tile for tile in self.tiles}
        tiles_by_rc = self._tiles_by_rc
        for row, col in {(int(r), int(c)) for r, c in zip(rows[on_grid], cols[on_grid])}:
            mask = on_grid & (rows == row) & (cols == col)
            out[mask] = tiles_by_rc[(row, col)].heights_w(x_arr[mask], y_arr[mask])
        if self.border_width_w > 0.0:
            on_border = (
                ~on_grid
                & (x_arr >= x_min - self.border_width_w)
                & (x_arr <= x_max + self.border_width_w)
                & (y_arr >= y_min - self.border_width_w)
                & (y_arr <= y_max + self.border_width_w)
            )
            out[on_border] = self.border_top_z_w
        # Obstacles overhang tile borders (repeated-objects centers sample the full
        # tile extent): consult every tile whose footprint-expanded bounds cover a
        # query it does not own and keep the highest surface.
        if self._tile_overhangs is None:
            self._tile_overhangs = [
                (tile, margin) for tile in self.tiles if (margin := tile.obstacle_overhang_w()) > 0.0
            ]
        for tile, margin in self._tile_overhangs:
            near = (
                (np.abs(x_arr - tile.origin_w[0]) <= 0.5 * tile.size[0] + margin)
                & (np.abs(y_arr - tile.origin_w[1]) <= 0.5 * tile.size[1] + margin)
                & ~(on_grid & (rows == tile.row) & (cols == tile.col))
            )
            if near.any():
                out[near] = np.fmax(out[near], tile.heights_w(x_arr[near], y_arr[near]))
        return float(out.flat[0]) if scalar else out


def build_terrain(
    terrain: TerrainIR | None,
    spec: mujoco.MjSpec | None = None,
    rng: np.random.Generator | None = None,
    *,
    num_envs: int = 1,
    env_spacing: float | None = None,
    max_init_terrain_level: int | None = None,
    measured: MeasuredTerrainData | None = None,
    measured_collision_resolution: float | None = None,
    measured_collision_window_w: tuple[float, float, float, float] | None = None,
    measured_exact_window_w: tuple[float, float, float, float] | None = None,
) -> TerrainBuild:
    """Build MuJoCo terrain pieces from a :class:`TerrainIR`.

    ``rng`` seeds everything the terrain generator's own seed does not cover: the
    env-origin terrain levels (IsaacLab draws these from the torch RNG, which is not
    reproducible from the config) and, when the generator cfg has no seed, the base
    layout seed. ``measured`` (generator terrain only) replaces the procedural
    regeneration with the reference dump's measured terrain instance — see the
    module docstring; ``num_envs``/``env_spacing``/``rng`` are then unused (the
    measured env origin is authoritative). ``measured_collision_resolution``
    re-raycasts the measured mesh onto a finer collision grid, optionally only
    inside ``measured_collision_window_w = (x_lo, x_hi, y_lo, y_hi)`` (recorded
    resolution elsewhere) — see the module docstring; ``measured_exact_window_w``
    instead collides the exact recorded mesh inside that window (convex prism per
    upward triangle — no riser-ramp deviation at all). If ``spec`` is given the
    terrain geoms are added to it and robot collision geoms get the PhysX pair
    friction + priority treatment (see :func:`add_to_spec`).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    terrain_type = terrain.terrain_type if terrain is not None else "plane"
    material = dict(terrain.physics_material) if terrain is not None else {}
    mu_ground = float(material.get("static_friction", 1.0))
    combine_mode = str(material.get("friction_combine_mode", "average"))

    if measured is not None and terrain_type != "generator":
        raise ValueError(f"measured terrain records apply to generator terrain only, not '{terrain_type}'")
    if measured is None and measured_collision_resolution is not None:
        raise ValueError("measured_collision_resolution requires a measured terrain record")
    if terrain_type == "plane":
        build = _build_plane(mu_ground, combine_mode, num_envs, env_spacing)
    elif terrain_type == "generator":
        if measured is not None:
            build = _build_measured(
                measured,
                mu_ground,
                combine_mode,
                collision_resolution=measured_collision_resolution,
                collision_window_w=measured_collision_window_w,
                exact_window_w=measured_exact_window_w,
            )
        else:
            if terrain is None or terrain.generator is None:
                raise ValueError("terrain_type 'generator' requires TerrainIR.generator")
            build = _build_generator(terrain.generator, mu_ground, combine_mode, rng, num_envs, max_init_terrain_level)
    else:
        raise NotImplementedError(f"terrain_type '{terrain_type}' is not supported (plane/generator only)")

    if spec is not None:
        add_to_spec(spec, build)
    return build


def add_to_spec(
    spec: mujoco.MjSpec,
    build: TerrainBuild,
    *,
    set_robot_pair_friction: bool = True,
    robot_self_pair: bool = False,
) -> list[str]:
    """Add the terrain hfield assets + geoms to ``spec``; returns the added geom names.

    When ``set_robot_pair_friction`` is set, every collision geom already present in
    the spec (the robot) is stamped with ``priority=1`` and its slide friction is
    replaced by the PhysX pair value against the ground material, so MuJoCo's
    priority rule reproduces PhysX's ``friction_combine_mode`` exactly for
    robot-ground contacts (see module docstring).

    With ``robot_self_pair`` (self-collision robots) BOTH PhysX pairings are
    precomputed: terrain geoms carry the robot-GROUND pair value at priority 2
    (they win every robot-ground contact) and robot geoms carry the robot-ROBOT
    pair value at priority 1 (the equal-priority max resolves every self contact
    to it). Requires a uniform robot material — heterogeneous self pairs cannot
    be expressed per geom — and is incompatible with the material DR event,
    which rewrites robot-geom friction per draw and needs robot geoms to win
    (the converter picks the scheme accordingly).
    """
    existing_collision_geoms = [geom for geom in spec.geoms if geom.contype != 0 or geom.conaffinity != 0]
    self_pair_mu: float | None = None
    mu_robot = float(existing_collision_geoms[0].friction[0]) if existing_collision_geoms else 1.0
    if set_robot_pair_friction and robot_self_pair and existing_collision_geoms:
        mu_values = [float(geom.friction[0]) for geom in existing_collision_geoms]
        if max(mu_values) - min(mu_values) <= 1e-12:
            self_pair_mu = pair_friction(mu_robot, mu_robot, build.friction_combine_mode)
        else:
            print(
                "[terrain] WARNING: robot collision geoms carry heterogeneous friction; per-pair "
                "self-collision values cannot be expressed per geom — keeping the robot-ground "
                "pair stamping (self contacts resolve at its max)"
            )
    for hfield_asset in build.hfields:
        mjs_hfield = spec.add_hfield()
        mjs_hfield.name = hfield_asset.name
        mjs_hfield.nrow = hfield_asset.nrow
        mjs_hfield.ncol = hfield_asset.ncol
        mjs_hfield.size = list(hfield_asset.size)
        # Raw metric heights; the compiler normalizes to [0, 1] (see HfieldAsset).
        mjs_hfield.userdata = hfield_asset.data_w.flatten()
    added = []
    for terrain_geom in build.geoms:
        mjs_geom = spec.worldbody.add_geom(
            name=terrain_geom.name,
            type=_MJ_GEOM_TYPE[terrain_geom.kind],
            pos=list(terrain_geom.pos_w),
            quat=list(terrain_geom.quat_wxyz),
        )
        if terrain_geom.kind == "hfield":
            mjs_geom.hfieldname = terrain_geom.hfield
        else:
            mjs_geom.size = list(terrain_geom.size)
        if self_pair_mu is None:
            mjs_geom.friction = [terrain_geom.friction_slide, 0.005, 0.0001]
            mjs_geom.priority = 0
        else:
            mjs_geom.friction = [
                pair_friction(mu_robot, terrain_geom.friction_slide, build.friction_combine_mode),
                0.005,
                0.0001,
            ]
            mjs_geom.priority = 2
        added.append(terrain_geom.name)
    if build.measured_prisms_w is not None:
        # Exact-collision prisms: 6-vertex point clouds; the compiler's convex hull
        # of each IS the prism, so collision reproduces the mesh surface exactly.
        for k, prism in enumerate(build.measured_prisms_w):
            mesh = spec.add_mesh()
            mesh.name = f"terrain_prism_{k}"
            mesh.uservert = np.asarray(prism, dtype=np.float64).ravel()
            mjs_geom = spec.worldbody.add_geom(name=f"terrain_prism_{k}", type=mujoco.mjtGeom.mjGEOM_MESH)
            mjs_geom.meshname = mesh.name
            if self_pair_mu is None:
                mjs_geom.friction = [build.ground_friction, 0.005, 0.0001]
                mjs_geom.priority = 0
            else:
                mjs_geom.friction = [
                    pair_friction(mu_robot, build.ground_friction, build.friction_combine_mode),
                    0.005,
                    0.0001,
                ]
                mjs_geom.priority = 2
            added.append(str(mjs_geom.name))
    if set_robot_pair_friction:
        for geom in existing_collision_geoms:
            geom.priority = 1
            if self_pair_mu is None:
                geom.friction[0] = build.robot_pair_friction(float(geom.friction[0]))
            else:
                geom.friction[0] = self_pair_mu
    return added


# ---------------------------------------------------------------------------
# plane
# ---------------------------------------------------------------------------


def _build_plane(mu_ground: float, combine_mode: str, num_envs: int, env_spacing: float | None) -> TerrainBuild:
    plane = TerrainGeom(
        name=_GROUND_GEOM_NAME,
        kind="plane",
        pos_w=(0.0, 0.0, 0.0),
        size=(0.0, 0.0, 0.05),
        friction_slide=mu_ground,
    )
    if num_envs == 1:
        env_origins_w = np.zeros((1, 3), dtype=np.float64)
    else:
        if env_spacing is None:
            raise ValueError("env_spacing is required for grid env origins with num_envs > 1")
        env_origins_w = _env_origins_grid_w(num_envs, env_spacing)
    return TerrainBuild(
        terrain_type="plane",
        geoms=[plane],
        ground_geom_names=[plane.name],
        env_origins_w=env_origins_w,
        flat_patches={},
        friction_combine_mode=combine_mode,
        ground_friction=mu_ground,
    )


def _env_origins_grid_w(num_envs: int, env_spacing: float) -> np.ndarray:
    """Grid origins, replicating ``TerrainImporter._compute_env_origins_grid``."""
    num_rows = int(np.ceil(num_envs / int(np.sqrt(num_envs))))
    num_cols = int(np.ceil(num_envs / num_rows))
    ii, jj = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing="ij")
    env_origins_w = np.zeros((num_envs, 3), dtype=np.float64)
    env_origins_w[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
    env_origins_w[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
    return env_origins_w


# ---------------------------------------------------------------------------
# measured (from a reference dump)
# ---------------------------------------------------------------------------

_MEASURED_HFIELD_NAME = "terrain_measured"
# Query rows per MeasuredSurface raycast chunk when resampling a fine collision
# grid: bounds the (queries x candidate-triangles) scratch arrays to a few hundred MB.
_RESAMPLE_CHUNK_QUERIES = 250_000


def _snap_window_to_grid(
    window_w: tuple[float, float, float, float],
    x0_w: float,
    y0_w: float,
    resolution: float,
    nx: int,
    ny: int,
    *,
    what: str,
) -> tuple[int, int, int, int]:
    """Snap a world window outward to recorded-grid node indices (i_lo, i_hi, j_lo, j_hi).

    Outward snapping puts the surrounding slabs' inner node lines exactly on the
    window edge.
    """
    x_lo_w, x_hi_w, y_lo_w, y_hi_w = (float(v) for v in window_w)
    if x_hi_w <= x_lo_w or y_hi_w <= y_lo_w:
        raise ValueError(f"{what} must have positive extent, got {window_w}")
    # The node-index epsilon makes snapping idempotent: a window already on grid
    # nodes (e.g. one returned by plan_measured_exact_window) maps back to the
    # same nodes instead of growing a cell on float round-off.
    eps = 1.0e-9
    i_lo = int(np.clip(math.floor((x_lo_w - x0_w) / resolution + eps), 0, nx - 2))
    i_hi = int(np.clip(math.ceil((x_hi_w - x0_w) / resolution - eps), i_lo + 1, nx - 1))
    j_lo = int(np.clip(math.floor((y_lo_w - y0_w) / resolution + eps), 0, ny - 2))
    j_hi = int(np.clip(math.ceil((y_hi_w - y0_w) / resolution - eps), j_lo + 1, ny - 1))
    return i_lo, i_hi, j_lo, j_hi


def _surrounding_slabs(
    grid_z_w: np.ndarray,
    x0_w: float,
    y0_w: float,
    resolution: float,
    snapped: tuple[int, int, int, int],
    mu_ground: float,
) -> list[tuple["HfieldAsset", "TerrainGeom"]]:
    """Four recorded-resolution slabs around a snapped window.

    x-low/x-high span the full y extent; y-low/y-high cover the window's x range.
    Each slab's inner node line coincides with the window edge; slabs flush with a
    grid edge are skipped.
    """
    nx, ny = grid_z_w.shape
    i_lo, i_hi, j_lo, j_hi = snapped
    pieces: list[tuple[HfieldAsset, TerrainGeom]] = []
    for suffix, si_lo, si_hi, sj_lo, sj_hi in (
        ("xlo", 0, i_lo, 0, ny - 1),
        ("xhi", i_hi, nx - 1, 0, ny - 1),
        ("ylo", i_lo, i_hi, 0, j_lo),
        ("yhi", i_lo, i_hi, j_hi, ny - 1),
    ):
        if si_hi <= si_lo or sj_hi <= sj_lo:
            continue  # window flush with this grid edge
        pieces.append(
            _hfield_piece(
                f"{_MEASURED_HFIELD_NAME}_{suffix}",
                grid_z_w[si_lo : si_hi + 1, sj_lo : sj_hi + 1],
                x0_w + si_lo * resolution,
                y0_w + sj_lo * resolution,
                resolution,
                resolution,
                mu_ground,
            )
        )
    return pieces


def _hfield_piece(
    name: str, grid_z_w: np.ndarray, x0_w: float, y0_w: float, step_x: float, step_y: float, mu_ground: float
) -> tuple[HfieldAsset, TerrainGeom]:
    """One hfield asset + geom pair carrying a metric height grid (see HfieldAsset)."""
    nx, ny = grid_z_w.shape
    span_x = (nx - 1) * step_x
    span_y = (ny - 1) * step_y
    z_min = float(grid_z_w.min())
    z_max = float(grid_z_w.max())
    asset = HfieldAsset(
        name=name,
        nrow=ny,  # hfield rows run along y, columns along x
        ncol=nx,
        # elevation_z must be positive; an all-flat grid normalizes to 0 everywhere,
        # so a tiny scale keeps the surface exactly at the geom's z position.
        size=(0.5 * span_x, 0.5 * span_y, max(z_max - z_min, 1e-6), 1.0),
        data_w=grid_z_w.T.copy(),
    )
    geom = TerrainGeom(
        name=name,
        kind="hfield",
        pos_w=(x0_w + 0.5 * span_x, y0_w + 0.5 * span_y, z_min),
        size=(0.0, 0.0, 0.0),
        friction_slide=mu_ground,
        hfield=name,
    )
    return asset, geom


def _resample_measured_grid(
    surface: MeasuredSurface,
    grid_z_w: np.ndarray,
    x0_w: float,
    y0_w: float,
    resolution: float,
    x_nodes_w: np.ndarray,
    y_nodes_w: np.ndarray,
) -> np.ndarray:
    """Raycast the measured mesh at the given node lattice; (len(x), len(y)) heights.

    Nodes the mesh does not cover (numerical misses on the outermost boundary edge)
    fall back to bilinear interpolation of the recorded grid, so the result carries
    no NaNs; genuine interior holes would be recorded-grid NaNs and are rejected
    upstream.
    """
    fine_z_w = np.empty((x_nodes_w.size, y_nodes_w.size), dtype=np.float64)
    coarse_interp = interpolate.RegularGridInterpolator(
        (
            x0_w + resolution * np.arange(grid_z_w.shape[0], dtype=np.float64),
            y0_w + resolution * np.arange(grid_z_w.shape[1], dtype=np.float64),
        ),
        grid_z_w,
        bounds_error=False,
        fill_value=None,  # nearest-edge extrapolation for nodes marginally outside
    )
    rows_per_chunk = max(1, _RESAMPLE_CHUNK_QUERIES // max(int(y_nodes_w.size), 1))
    for start in range(0, x_nodes_w.size, rows_per_chunk):
        x_chunk_w = x_nodes_w[start : start + rows_per_chunk]
        gx_w, gy_w = np.meshgrid(x_chunk_w, y_nodes_w, indexing="ij")
        z_w = surface.heights_w(gx_w, gy_w)
        missed = np.isnan(z_w)
        if missed.any():
            z_w[missed] = coarse_interp(np.stack([gx_w[missed], gy_w[missed]], axis=-1))
        fine_z_w[start : start + x_chunk_w.size] = z_w
    return fine_z_w


# Hard cap on exact-collision prisms: generator meshes are coarse per feature, so a
# trajectory window holds a few hundred triangles; hitting this means the window or
# the mesh is unexpectedly dense and hfield collision should be used instead.
MEASURED_EXACT_PRISM_CAP = 20_000

# Triangles with |unit normal z| below this are vertical faces (stair risers,
# grid-cell walls): their surface is exactly the shared side face of the adjacent
# floor prisms, so they carry no prism of their own.
_EXACT_VERTICAL_NZ = 1.0e-6


def _exact_prism_tris(
    measured: MeasuredTerrainData,
    window_w: tuple[float, float, float, float],
    surface: MeasuredSurface,
) -> tuple[np.ndarray | None, str | None]:
    """Upward mesh triangles for exact-prism collision inside ``window_w``.

    Returns ``(tris, None)`` on success, ``(None, reason)`` when exact collision
    is infeasible in the window: no (upward) triangles, exposed overhangs, or an
    upward-triangle count over the prism cap. Both the plan
    (:func:`plan_measured_exact_window`) and the build run this same
    classification, so they cannot disagree.
    """
    verts = np.asarray(measured.mesh_vertices_w, dtype=np.float64)
    tris = verts[np.asarray(measured.mesh_faces, dtype=np.int64)]  # (F, 3, 3)
    x_lo, x_hi, y_lo, y_hi = window_w
    overlap = (
        (tris[:, :, 0].min(axis=1) <= x_hi)
        & (tris[:, :, 0].max(axis=1) >= x_lo)
        & (tris[:, :, 1].min(axis=1) <= y_hi)
        & (tris[:, :, 1].max(axis=1) >= y_lo)
    )
    tris = tris[overlap]
    if tris.shape[0] == 0:
        return None, f"no mesh triangles intersect the exact collision window {window_w}"
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm = np.linalg.norm(normals, axis=1)
    valid = norm > 0.0
    nz = np.where(valid, normals[:, 2] / np.where(valid, norm, 1.0), 0.0)
    down = valid & (nz < -_EXACT_VERTICAL_NZ)
    if np.any(down):
        centroids = tris[down].mean(axis=1)
        surface_z = np.asarray(surface.heights_w(centroids[:, 0], centroids[:, 1]), dtype=np.float64)
        # Occluded only when the surface lies STRICTLY above the face: an overhang is
        # itself the top surface at its xy, so equality means exposed. NaN too.
        exposed = ~(surface_z >= centroids[:, 2] + 1.0e-6)
        if np.any(exposed):
            return None, (
                f"{int(exposed.sum())} downward-facing mesh triangles form the surface inside the exact "
                "collision window (overhanging terrain cannot be represented by extruded prisms)"
            )
    tris = tris[valid & (nz > _EXACT_VERTICAL_NZ)]
    if tris.shape[0] == 0:
        return None, f"no upward mesh triangles inside the exact collision window {window_w}"
    if tris.shape[0] > MEASURED_EXACT_PRISM_CAP:
        return None, (
            f"{tris.shape[0]} exact-collision prisms in window {window_w} exceed the cap {MEASURED_EXACT_PRISM_CAP}"
        )
    return tris, None


def plan_measured_exact_window(
    measured: MeasuredTerrainData,
    window_w: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """The snapped exact-collision window the build will use, or ``None`` when infeasible.

    Snaps ``window_w`` outward to recorded-grid nodes and classifies the mesh
    triangles inside exactly like ``_build_measured`` will, so a returned window
    is guaranteed to build. On infeasibility (no upward triangles, exposed
    overhangs, prism count over the cap) a fallback notice is printed and the
    conversion should use hfield collision instead.
    """
    nx, ny = np.asarray(measured.height_grid).shape
    resolution = float(measured.grid_resolution)
    x0_w, y0_w = float(measured.grid_origin_xy[0]), float(measured.grid_origin_xy[1])
    i_lo, i_hi, j_lo, j_hi = _snap_window_to_grid(
        window_w, x0_w, y0_w, resolution, nx, ny, what="exact collision window"
    )
    win = (x0_w + i_lo * resolution, x0_w + i_hi * resolution, y0_w + j_lo * resolution, y0_w + j_hi * resolution)
    surface = MeasuredSurface(measured.mesh_vertices_w, measured.mesh_faces, cell=2.0 * resolution)
    tris, reason = _exact_prism_tris(measured, win, surface)
    if tris is None:
        print(f"[convert] NOTICE: {reason}; falling back to hfield collision")
        return None
    return win


def _measured_exact_prisms(
    measured: MeasuredTerrainData,
    window_w: tuple[float, float, float, float],
    surface: MeasuredSurface,
) -> np.ndarray:
    """Convex collision prisms (P, 6, 3) for the mesh triangles intersecting ``window_w``.

    Each upward triangle becomes its own convex prism: the triangle on top, its xy
    copy at a common base plane below the window's lowest vertex. The union of the
    prisms' surfaces reproduces the mesh surface exactly (top faces are the
    triangles; risers are the prisms' vertical sides). Downward-facing triangles
    that lie under the surface (Isaac's generator builds terrain from closed boxes,
    so undersides are common) are occluded by the prisms above and are skipped; a
    downward face that IS the surface at its location is a genuine overhang, which
    downward extrusion cannot represent. Infeasible windows raise —
    :func:`plan_measured_exact_window` decides feasibility ahead of the build with
    the same classification.
    """
    tris, reason = _exact_prism_tris(measured, window_w, surface)
    if tris is None:
        raise ValueError(f"{reason}; convert with hfield collision instead")
    base_z = float(tris[:, :, 2].min()) - 1.0
    bottom = tris.copy()
    bottom[:, :, 2] = base_z
    return np.concatenate([tris, bottom], axis=1)  # (P, 6, 3)


def _build_measured(
    measured: MeasuredTerrainData,
    mu_ground: float,
    combine_mode: str,
    *,
    collision_resolution: float | None = None,
    collision_window_w: tuple[float, float, float, float] | None = None,
    exact_window_w: tuple[float, float, float, float] | None = None,
) -> TerrainBuild:
    """Terrain from a measured record: hfield collision + exact mesh surface.

    Default collision is one full-extent hfield carrying the recorded height grid.
    With ``collision_resolution`` the mesh is re-raycast at that node spacing —
    over the full extent, or (with ``collision_window_w``, snapped outward to
    recorded-grid nodes) only inside the window, with four recorded-resolution
    slabs covering the rest. Slabs share their window-edge node line with the fine
    grid, so the pieces meet at the recorded nodes exactly (between shared nodes
    the two interpolations may differ by the usual sub-cell deviation).

    ``exact_window_w`` replaces the windowed FINE hfield with the exact recorded
    mesh: every upward mesh triangle intersecting the window becomes a convex
    prism (the triangle extruded to a common base), so vertical faces (stair
    risers) collide as true walls instead of one-cell ramps — hfield collision
    inside the window carries up to half a riser of surface error exactly where
    a foot stands. Vertical triangles are skipped (their surface is the shared
    side face of the neighboring floor prisms); downward-facing triangles
    (overhangs) are unsupported and raise. Recorded-resolution slabs cover the
    rest of the terrain as in the fine-window mode.
    """
    grid_z_w = np.asarray(measured.height_grid, dtype=np.float64)
    if grid_z_w.ndim != 2 or grid_z_w.shape[0] < 2 or grid_z_w.shape[1] < 2:
        raise ValueError(f"measured height grid must be (nx >= 2, ny >= 2), got {grid_z_w.shape}")
    if np.isnan(grid_z_w).any():
        raise ValueError("measured height grid contains NaN nodes; re-capture the dump")
    resolution = float(measured.grid_resolution)
    if resolution <= 0.0:
        raise ValueError(f"measured grid resolution must be positive, got {resolution}")
    if collision_window_w is not None and collision_resolution is None:
        raise ValueError("collision_window_w requires collision_resolution")
    if exact_window_w is not None and collision_resolution is not None:
        raise ValueError("exact_window_w and collision_resolution are mutually exclusive")
    nx, ny = grid_z_w.shape
    x0_w, y0_w = float(measured.grid_origin_xy[0]), float(measured.grid_origin_xy[1])
    surface = MeasuredSurface(measured.mesh_vertices_w, measured.mesh_faces, cell=2.0 * resolution)

    hfields: list[HfieldAsset] = []
    geoms: list[TerrainGeom] = []
    fine_window_w: tuple[float, float, float, float] | None = None
    prisms_w: np.ndarray | None = None
    if exact_window_w is not None:
        snapped = _snap_window_to_grid(exact_window_w, x0_w, y0_w, resolution, nx, ny, what="exact collision window")
        i_lo, i_hi, j_lo, j_hi = snapped
        win = (x0_w + i_lo * resolution, x0_w + i_hi * resolution, y0_w + j_lo * resolution, y0_w + j_hi * resolution)
        prisms_w = _measured_exact_prisms(measured, win, surface)
        fine_window_w = win
        for asset, geom in _surrounding_slabs(grid_z_w, x0_w, y0_w, resolution, snapped, mu_ground):
            hfields.append(asset)
            geoms.append(geom)
        effective_resolution = resolution
    elif collision_resolution is None:
        asset, geom = _hfield_piece(_MEASURED_HFIELD_NAME, grid_z_w, x0_w, y0_w, resolution, resolution, mu_ground)
        hfields.append(asset)
        geoms.append(geom)
        effective_resolution = resolution
    else:
        fine_resolution = float(collision_resolution)
        if fine_resolution <= 0.0:
            raise ValueError(f"collision resolution must be positive, got {fine_resolution}")
        # Snap the window outward to recorded-grid nodes (full extent without one).
        if collision_window_w is None:
            i_lo, i_hi, j_lo, j_hi = 0, nx - 1, 0, ny - 1
        else:
            i_lo, i_hi, j_lo, j_hi = _snap_window_to_grid(
                collision_window_w, x0_w, y0_w, resolution, nx, ny, what="collision window"
            )
        fine_x0_w = x0_w + i_lo * resolution
        fine_y0_w = y0_w + j_lo * resolution
        span_x_f = (i_hi - i_lo) * resolution
        span_y_f = (j_hi - j_lo) * resolution
        # Node counts from the snapped span; the effective spacing (span / (n - 1))
        # equals the requested resolution whenever it divides the recorded one.
        nfx = max(int(round(span_x_f / fine_resolution)), 1) + 1
        nfy = max(int(round(span_y_f / fine_resolution)), 1) + 1
        x_nodes_w = fine_x0_w + np.arange(nfx, dtype=np.float64) * (span_x_f / (nfx - 1))
        y_nodes_w = fine_y0_w + np.arange(nfy, dtype=np.float64) * (span_y_f / (nfy - 1))
        fine_z_w = _resample_measured_grid(surface, grid_z_w, x0_w, y0_w, resolution, x_nodes_w, y_nodes_w)
        full_extent = (i_lo, i_hi, j_lo, j_hi) == (0, nx - 1, 0, ny - 1)
        fine_name = _MEASURED_HFIELD_NAME if full_extent else f"{_MEASURED_HFIELD_NAME}_fine"
        asset, geom = _hfield_piece(
            fine_name, fine_z_w, fine_x0_w, fine_y0_w, span_x_f / (nfx - 1), span_y_f / (nfy - 1), mu_ground
        )
        hfields.append(asset)
        geoms.append(geom)
        if not full_extent:
            fine_window_w = (fine_x0_w, fine_x0_w + span_x_f, fine_y0_w, fine_y0_w + span_y_f)
            snapped = (i_lo, i_hi, j_lo, j_hi)
            for asset, geom in _surrounding_slabs(grid_z_w, x0_w, y0_w, resolution, snapped, mu_ground):
                hfields.append(asset)
                geoms.append(geom)
        effective_resolution = span_x_f / (nfx - 1)

    tile_origins_w = None if measured.tile_origins_w is None else np.asarray(measured.tile_origins_w, dtype=np.float64)
    return TerrainBuild(
        terrain_type="measured",
        geoms=geoms,
        ground_geom_names=[geom.name for geom in geoms],
        env_origins_w=np.asarray(measured.env_origin_w, dtype=np.float64).reshape(1, 3),
        flat_patches={},
        friction_combine_mode=combine_mode,
        ground_friction=mu_ground,
        terrain_origins_w=tile_origins_w,
        hfields=hfields,
        measured_surface=surface,
        measured_collision_resolution=effective_resolution,
        measured_fine_window_w=fine_window_w,
        measured_prisms_w=prisms_w,
    )


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------


def _build_generator(
    generator: dict[str, Any],
    mu_ground: float,
    combine_mode: str,
    rng: np.random.Generator,
    num_envs: int,
    max_init_terrain_level: int | None,
) -> TerrainBuild:
    num_rows = int(generator["num_rows"])
    num_cols = int(generator["num_cols"])
    size = (float(generator["size"][0]), float(generator["size"][1]))
    sub_terrains: dict[str, dict[str, Any]] = generator["sub_terrains"]
    if not sub_terrains:
        raise ValueError("terrain generator cfg has no sub_terrains")
    difficulty_range = generator.get("difficulty_range") or (0.0, 1.0)
    seed = generator.get("seed")
    base_seed = int(seed) if seed is not None else int(rng.integers(0, 2**31 - 1))
    # Same construction as TerrainGenerator.np_rng: drives the sub-terrain type and
    # difficulty schedule below with the identical consumption order.
    np_rng = np.random.default_rng(base_seed)

    names = list(sub_terrains.keys())
    proportions = np.array([float(sub_terrains[n].get("proportion", 1.0)) for n in names])
    proportions = proportions / proportions.sum()

    # (row, col, sub-terrain name, difficulty) in IsaacLab's generation order.
    schedule: list[tuple[int, int, str, float]] = []
    lower, upper = float(difficulty_range[0]), float(difficulty_range[1])
    if generator.get("curriculum", False):
        sub_indices = [
            int(np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions))[0])) for index in range(num_cols)
        ]
        for sub_col in range(num_cols):
            for sub_row in range(num_rows):
                difficulty = (sub_row + np_rng.uniform()) / num_rows
                difficulty = lower + (upper - lower) * difficulty
                schedule.append((sub_row, sub_col, names[sub_indices[sub_col]], difficulty))
    else:
        for index in range(num_rows * num_cols):
            sub_row, sub_col = np.unravel_index(index, (num_rows, num_cols))
            sub_index = int(np_rng.choice(len(proportions), p=proportions))
            difficulty = float(np_rng.uniform(lower, upper))
            schedule.append((int(sub_row), int(sub_col), names[sub_index], difficulty))

    tiles: list[TileBuild] = []
    geoms: list[TerrainGeom] = []
    hfields: list[HfieldAsset] = []
    for row, col, name, difficulty in schedule:
        # Tile center after TerrainGenerator's per-tile translation and final centering.
        center_x_w = (row + 0.5) * size[0] - num_rows * size[0] * 0.5
        center_y_w = (col + 0.5) * size[1] - num_cols * size[1] * 0.5
        tile, tile_geoms, tile_hfields = _build_tile(
            sub_terrains[name],
            name,
            row,
            col,
            difficulty,
            size,
            (center_x_w, center_y_w),
            base_seed,
            mu_ground,
            generator,
        )
        tiles.append(tile)
        geoms.extend(tile_geoms)
        hfields.extend(tile_hfields)

    # Surrounding border ring, replicating TerrainGenerator._add_terrain_border.
    border_width = float(generator.get("border_width", 0.0))
    border_height = float(generator.get("border_height", 1.0))
    if border_width > 0.0:
        outer = map_size_from_generator(generator)
        inner = (num_rows * size[0], num_cols * size[1])
        # make_border places the ring at the pre-centering grid center and the
        # generator's final centering shift maps that center to the world origin;
        # _border_boxes returns the ring centered at (outer / 2, outer / 2), so
        # each box shifts by -outer / 2 to land in world frame.
        for side, box in zip(_BORDER_SIDES, _border_boxes(outer, inner, abs(border_height), -border_height / 2)):
            (cx, cy, cz), dims = box
            box_w = ((cx - 0.5 * outer[0], cy - 0.5 * outer[1], cz), dims)
            geoms.append(_box_geom(f"border_{side}", box_w, mu_ground))

    terrain_origins_w = np.zeros((num_rows, num_cols, 3), dtype=np.float64)
    for tile in tiles:
        terrain_origins_w[tile.row, tile.col] = tile.origin_w

    env_origins_w = _env_origins_curriculum_w(num_envs, terrain_origins_w, max_init_terrain_level, rng)

    patch_names = sorted({patch_name for tile in tiles for patch_name in tile.flat_patch_cfgs})
    flat_patches = {
        patch_name: FlatPatchSampler(patch_name, tiles, default_seed=base_seed + 1 + index)
        for index, patch_name in enumerate(patch_names)
    }

    return TerrainBuild(
        terrain_type="generator",
        geoms=geoms,
        ground_geom_names=[geom.name for geom in geoms],
        env_origins_w=env_origins_w,
        flat_patches=flat_patches,
        friction_combine_mode=combine_mode,
        ground_friction=mu_ground,
        terrain_origins_w=terrain_origins_w,
        tiles=tiles,
        hfields=hfields,
        tile_size=size,
        grid_shape=(num_rows, num_cols),
        border_width_w=border_width,
        # make_border centers the ring at z = -border_height / 2 with depth
        # |border_height|: the top sits at 0 for positive heights (border below
        # ground) and at |border_height| for negative ones (border above ground).
        border_top_z_w=max(0.0, -border_height),
    )


def _env_origins_curriculum_w(
    num_envs: int,
    terrain_origins_w: np.ndarray,
    max_init_terrain_level: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replicates ``TerrainImporter._compute_env_origins_curriculum``.

    IsaacLab draws the initial terrain levels from the torch RNG; the caller's numpy
    ``rng`` stands in for it (level layout is not part of the strict-match surface).
    """
    num_rows, num_cols = terrain_origins_w.shape[:2]
    max_init_level = num_rows - 1 if max_init_terrain_level is None else min(max_init_terrain_level, num_rows - 1)
    terrain_levels = rng.integers(0, max_init_level + 1, size=num_envs)
    terrain_types = _curriculum_terrain_types(num_envs, num_cols)
    return terrain_origins_w[terrain_levels, terrain_types].astype(np.float64)


def _curriculum_terrain_types(num_envs: int, num_cols: int) -> np.ndarray:
    """Column assignment of ``TerrainImporter._compute_env_origins_curriculum``, bit-exact.

    IsaacLab computes ``torch.div(arange(num_envs), num_envs / num_cols,
    rounding_mode="floor")``: the int64 arange and the python-float divisor promote to
    torch's default dtype (float32), and torch's ``div_floor_floating`` kernel derives
    the quotient via ``fmod`` — ``floor((a - fmod(a, b)) / b)`` with a +1 correction
    when the intermediate lands just below an integer — rather than flooring a plain
    division. At exact column boundaries (``num_envs`` a multiple of ``num_cols``) the
    two disagree, so the kernel is replicated verbatim in float32 (both operands are
    positive, which drops the kernel's negative-sign branches).
    """
    a = np.arange(num_envs, dtype=np.float32)
    b = np.float32(num_envs / num_cols)
    mod = np.fmod(a, b)
    div = (a - mod) / b
    floordiv = np.floor(div)
    floordiv = np.where(div - floordiv > np.float32(0.5), floordiv + np.float32(1.0), floordiv)
    return floordiv.astype(np.int64)


# ---------------------------------------------------------------------------
# sub-terrain tiles
# ---------------------------------------------------------------------------

# One placed box in the tile generation frame ([0, size] x/y): center + full dims.
_Box = tuple[tuple[float, float, float], tuple[float, float, float]]


def _border_boxes(
    size: tuple[float, float], inner_size: tuple[float, float], height: float, center_z: float
) -> list[_Box]:
    """Rectangular border ring, replicating ``trimesh.utils.make_border`` (left,
    right, top, bottom order) centered at (size / 2, size / 2, center_z)."""
    cx, cy = 0.5 * size[0], 0.5 * size[1]
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    return [
        ((cx - inner_size[0] / 2.0 - thickness_x / 2.0, cy, center_z), (thickness_x, inner_size[1], height)),
        ((cx + inner_size[0] / 2.0 + thickness_x / 2.0, cy, center_z), (thickness_x, inner_size[1], height)),
        ((cx, cy + inner_size[1] / 2.0 + thickness_y / 2.0, center_z), (size[0], thickness_y, height)),
        ((cx, cy - inner_size[1] / 2.0 - thickness_y / 2.0, center_z), (size[0], thickness_y, height)),
    ]


def _box_geom(name: str, box: _Box, mu_ground: float) -> TerrainGeom:
    (cx, cy, cz), (dx, dy, dz) = box
    if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
        raise ValueError(f"terrain box '{name}' has non-positive dimensions ({dx}, {dy}, {dz})")
    return TerrainGeom(
        name=name,
        kind="box",
        pos_w=(cx, cy, cz),
        size=(0.5 * dx, 0.5 * dy, 0.5 * dz),
        friction_slide=mu_ground,
    )


def _boxes_surface_w(geoms: list[TerrainGeom]) -> np.ndarray:
    """(M, 5) axis-aligned box footprints + tops from box geoms (identity quat)."""
    rows = []
    for geom in geoms:
        (cx, cy, cz), (hx, hy, hz) = geom.pos_w, geom.size
        rows.append((cx - hx, cx + hx, cy - hy, cy + hy, cz + hz))
    return np.asarray(rows, dtype=np.float64)


def repeated_object_params(sub_terrain: dict[str, Any], difficulty: float) -> dict[str, float | int]:
    """Difficulty interpolation of ``repeated_objects_terrain`` (exact formulas).

    Returns num_objects, height, platform_height plus radius (cylinders) or
    length/width (boxes).
    """
    cp_0 = sub_terrain["object_params_start"]
    cp_1 = sub_terrain["object_params_end"]
    num_objects = int(cp_0["num_objects"]) + int(difficulty * (int(cp_1["num_objects"]) - int(cp_0["num_objects"])))
    height = float(cp_0["height"]) + difficulty * (float(cp_1["height"]) - float(cp_0["height"]))
    platform_height = float(sub_terrain.get("platform_height", -1.0))
    if platform_height < 0.0:
        platform_height = height
    params: dict[str, float | int] = {
        "num_objects": num_objects,
        "height": height,
        "platform_height": platform_height,
    }
    kind = _object_kind(sub_terrain)
    if kind == "cylinder":
        params["radius"] = float(cp_0["radius"]) + difficulty * (float(cp_1["radius"]) - float(cp_0["radius"]))
    else:
        params["length"] = float(cp_0["size"][0]) + difficulty * (float(cp_1["size"][0]) - float(cp_0["size"][0]))
        params["width"] = float(cp_0["size"][1]) + difficulty * (float(cp_1["size"][1]) - float(cp_0["size"][1]))
    return params


def _object_kind(sub_terrain: dict[str, Any]) -> str:
    object_type = sub_terrain.get("object_type")
    kind = class_name(object_type).removeprefix("make_") if isinstance(object_type, str) else str(object_type)
    if kind not in ("cylinder", "box"):
        raise NotImplementedError(f"repeated-objects type '{kind}' is not supported (cylinder/box only)")
    return kind


def platform_clear_region_w(tile: TileBuild) -> tuple[tuple[float, float], tuple[float, float]]:
    """The platform rejection box of ``repeated_objects_terrain``, in world frame.

    IsaacLab scales the *absolute* platform corner coordinates by (1 -+ 0.1), i.e. the
    clearance grows with the tile size; replicated verbatim.
    """
    half_x, half_y = 0.5 * tile.size[0], 0.5 * tile.size[1]
    low = (
        (half_x - 0.5 * tile.platform_width) * (1.0 - _PLATFORM_CLEARANCE),
        (half_y - 0.5 * tile.platform_width) * (1.0 - _PLATFORM_CLEARANCE),
    )
    high = (
        (half_x + 0.5 * tile.platform_width) * (1.0 + _PLATFORM_CLEARANCE),
        (half_y + 0.5 * tile.platform_width) * (1.0 + _PLATFORM_CLEARANCE),
    )
    # Convert from the [0, size] generation frame to world (tile center at origin_w).
    tile_min_x_w = tile.origin_w[0] - half_x
    tile_min_y_w = tile.origin_w[1] - half_y
    return (
        (tile_min_x_w + low[0], tile_min_y_w + low[1]),
        (tile_min_x_w + high[0], tile_min_y_w + high[1]),
    )


def _build_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    base_seed: int,
    mu_ground: float,
    generator: dict[str, Any],
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    func_ref = str(sub_terrain.get("function", ""))
    func = class_name(func_ref)
    is_height_field = "height_field" in func_ref or "hf_terrains" in func_ref
    flat_patch_cfgs = dict(sub_terrain.get("flat_patch_sampling") or {})
    args = (sub_terrain, name, row, col, difficulty, size, center_w, mu_ground, flat_patch_cfgs)
    if func == "flat_terrain":
        return _build_flat_tile(*args)
    if func == "repeated_objects_terrain":
        return _build_repeated_objects_tile(*args, base_seed=base_seed)
    if is_height_field and func in ("random_uniform_terrain", "pyramid_sloped_terrain"):
        return _build_hf_tile(*args, base_seed=base_seed, generator=generator, func=func)
    if not is_height_field and func in ("pyramid_stairs_terrain", "inverted_pyramid_stairs_terrain"):
        return _build_stairs_tile(*args, inverted=func == "inverted_pyramid_stairs_terrain")
    if func == "random_grid_terrain":
        return _build_random_grid_tile(*args, base_seed=base_seed)
    raise NotImplementedError(
        f"sub-terrain '{name}' uses '{func}', which has no MuJoCo mapping (supported: flat_terrain, "
        "repeated_objects_terrain, random_uniform_terrain, pyramid_sloped_terrain [inverted via cfg], "
        "pyramid_stairs_terrain, inverted_pyramid_stairs_terrain, random_grid_terrain)"
    )


def _build_flat_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
    flat_patch_cfgs: dict[str, dict[str, Any]],
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    tile = TileBuild(
        row=row,
        col=col,
        sub_terrain=name,
        difficulty=difficulty,
        origin_w=(center_w[0], center_w[1], 0.0),
        size=size,
        flat_patch_cfgs=flat_patch_cfgs,
        surface_kind="objects",
    )
    return tile, [_tile_ground_geom(tile, mu_ground)], []


def _tile_ground_geom(tile: TileBuild, mu_ground: float) -> TerrainGeom:
    """Per-tile ground box with its top at z = 0 (repeated_objects_terrain's plane)."""
    return TerrainGeom(
        name=f"tile_{tile.row}_{tile.col}_ground",
        kind="box",
        pos_w=(tile.origin_w[0], tile.origin_w[1], -_TILE_BASE_HALF_THICKNESS),
        size=(0.5 * tile.size[0], 0.5 * tile.size[1], _TILE_BASE_HALF_THICKNESS),
        friction_slide=mu_ground,
    )


# -- repeated objects --------------------------------------------------------


def _build_repeated_objects_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
    flat_patch_cfgs: dict[str, dict[str, Any]],
    *,
    base_seed: int,
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    kind = _object_kind(sub_terrain)
    params = repeated_object_params(sub_terrain, difficulty)
    cp_0, cp_1 = sub_terrain["object_params_start"], sub_terrain["object_params_end"]
    max_yx_angle = float(cp_0.get("max_yx_angle", 0.0)) + difficulty * (
        float(cp_1.get("max_yx_angle", 0.0)) - float(cp_0.get("max_yx_angle", 0.0))
    )
    if max_yx_angle != 0.0:
        raise NotImplementedError(f"sub-terrain '{name}': max_yx_angle != 0 (tilted objects) is not supported")
    abs_noise = sub_terrain.get("abs_height_noise") or (0.0, 0.0)
    rel_noise = sub_terrain.get("rel_height_noise") or (1.0, 1.0)
    platform_width = float(sub_terrain.get("platform_width", 1.0))
    platform_height = float(params["platform_height"])
    num_objects = int(params["num_objects"])
    height = float(params["height"])

    # Deterministic per-tile placement stream (see module docstring for why this
    # cannot match IsaacLab's global-np.random placement draw for draw).
    tile_rng = np.random.default_rng([base_seed, row, col])

    # Rejection-sample centers in the [0, size] generation frame, exactly like
    # repeated_objects_terrain (including the absolute-coordinate clearance scaling).
    origin_gen = np.array([0.5 * size[0], 0.5 * size[1]])
    corners_low = (origin_gen - 0.5 * platform_width) * (1.0 - _PLATFORM_CLEARANCE)
    corners_high = (origin_gen + 0.5 * platform_width) * (1.0 + _PLATFORM_CLEARANCE)
    centers_gen = np.zeros((num_objects, 2))
    mask_left = np.ones(num_objects, dtype=bool)
    while np.any(mask_left):
        num_left = int(mask_left.sum())
        centers_gen[mask_left, 0] = tile_rng.uniform(0.0, size[0], num_left)
        centers_gen[mask_left, 1] = tile_rng.uniform(0.0, size[1], num_left)
        within_x = (centers_gen[mask_left, 0] >= corners_low[0]) & (centers_gen[mask_left, 0] <= corners_high[0])
        within_y = (centers_gen[mask_left, 1] >= corners_low[1]) & (centers_gen[mask_left, 1] <= corners_high[1])
        mask_left[mask_left] = within_x & within_y

    obstacles: list[_Obstacle] = []
    for index in range(num_objects):
        abs_height_noise = tile_rng.uniform(float(abs_noise[0]), float(abs_noise[1]))
        rel_height_noise = tile_rng.uniform(float(rel_noise[0]), float(rel_noise[1]))
        ob_height = height * rel_height_noise + abs_height_noise
        if ob_height <= 0.0:
            continue
        x_w = center_w[0] + centers_gen[index, 0] - 0.5 * size[0]
        y_w = center_w[1] + centers_gen[index, 1] - 0.5 * size[1]
        # Objects are centered at z = 0 (half buried), so the top sits at ob_height / 2.
        if kind == "cylinder":
            obstacles.append(
                _Obstacle(kind="cylinder", x_w=x_w, y_w=y_w, top_z_w=0.5 * ob_height, radius=float(params["radius"]))
            )
        else:
            yaw = float(tile_rng.uniform(0.0, 2.0 * np.pi))
            obstacles.append(
                _Obstacle(
                    kind="box",
                    x_w=x_w,
                    y_w=y_w,
                    top_z_w=0.5 * ob_height,
                    half_x=0.5 * float(params["length"]),
                    half_y=0.5 * float(params["width"]),
                    yaw=yaw,
                )
            )

    tile = TileBuild(
        row=row,
        col=col,
        sub_terrain=name,
        difficulty=difficulty,
        origin_w=(center_w[0], center_w[1], 0.5 * platform_height),
        size=size,
        platform_width=platform_width,
        platform_top_z_w=0.5 * platform_height,
        obstacles=obstacles,
        flat_patch_cfgs=flat_patch_cfgs,
        surface_kind="objects",
    )
    return tile, _repeated_objects_geoms(tile, mu_ground), []


def _repeated_objects_geoms(tile: TileBuild, mu_ground: float) -> list[TerrainGeom]:
    geoms: list[TerrainGeom] = [_tile_ground_geom(tile, mu_ground)]
    prefix = f"tile_{tile.row}_{tile.col}"
    if tile.platform_top_z_w > 0.0:
        geoms.append(
            TerrainGeom(
                name=f"{prefix}_platform",
                kind="box",
                pos_w=(tile.origin_w[0], tile.origin_w[1], 0.5 * tile.platform_top_z_w),
                size=(0.5 * tile.platform_width, 0.5 * tile.platform_width, 0.5 * tile.platform_top_z_w),
                friction_slide=mu_ground,
            )
        )
    for index, obstacle in enumerate(tile.obstacles):
        if obstacle.kind == "cylinder":
            geoms.append(
                TerrainGeom(
                    name=f"{prefix}_obj{index}",
                    kind="cylinder",
                    pos_w=(obstacle.x_w, obstacle.y_w, 0.0),
                    size=(obstacle.radius, obstacle.top_z_w, 0.0),
                    friction_slide=mu_ground,
                )
            )
        else:
            half_yaw = 0.5 * obstacle.yaw
            geoms.append(
                TerrainGeom(
                    name=f"{prefix}_obj{index}",
                    kind="box",
                    pos_w=(obstacle.x_w, obstacle.y_w, 0.0),
                    size=(obstacle.half_x, obstacle.half_y, obstacle.top_z_w),
                    quat_wxyz=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
                    friction_slide=mu_ground,
                )
            )
    return geoms


# -- height-field tiles ------------------------------------------------------


def _build_hf_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
    flat_patch_cfgs: dict[str, dict[str, Any]],
    *,
    base_seed: int,
    generator: dict[str, Any],
    func: str,
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    # TerrainGenerator.__init__ stamps its own horizontal/vertical scale onto every
    # height-field sub-terrain cfg; the generator-level values always win.
    horizontal_scale = float(generator.get("horizontal_scale", 0.1))
    vertical_scale = float(generator.get("vertical_scale", 0.005))
    tile_rng = np.random.default_rng([base_seed, row, col])
    heights = _hf_height_grid(func, sub_terrain, difficulty, size, horizontal_scale, vertical_scale, tile_rng)
    grid_z_w = heights.astype(np.float64) * vertical_scale
    # Node (0, 0) sits at the tile's minimum corner: the [0, size] generation frame
    # is shifted by exactly -size/2, so a grid span below `size` leaves the sliver at
    # the maximum-x/y edge, exactly like IsaacLab's mesh.
    x0_w = center_w[0] - 0.5 * size[0]
    y0_w = center_w[1] - 0.5 * size[1]

    # Origin z, replicating height_field_to_mesh: max height in the central 2 m
    # window of the full (bordered) grid.
    x1 = int((size[0] * 0.5 - 1) / horizontal_scale)
    x2 = int((size[0] * 0.5 + 1) / horizontal_scale)
    y1 = int((size[1] * 0.5 - 1) / horizontal_scale)
    y2 = int((size[1] * 0.5 + 1) / horizontal_scale)
    origin_z = float(heights[x1:x2, y1:y2].max()) * vertical_scale

    tile = TileBuild(
        row=row,
        col=col,
        sub_terrain=name,
        difficulty=difficulty,
        origin_w=(center_w[0], center_w[1], origin_z),
        size=size,
        flat_patch_cfgs=flat_patch_cfgs,
        surface_kind="hfield",
        grid_z_w=grid_z_w,
        grid_x0_w=x0_w,
        grid_y0_w=y0_w,
        grid_step=horizontal_scale,
    )
    asset, geom = _hfield_piece(
        f"tile_{row}_{col}_hf", grid_z_w, x0_w, y0_w, horizontal_scale, horizontal_scale, mu_ground
    )
    return tile, [geom], [asset]


def _hf_height_grid(
    func: str,
    sub_terrain: dict[str, Any],
    difficulty: float,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The full bordered height grid of ``height_field_to_mesh``, in vertical-scale
    units (int16), replicating every discrete unit conversion verbatim."""
    border_width = float(sub_terrain.get("border_width", 0.0))
    if border_width > 0 and border_width < horizontal_scale:
        raise ValueError(
            f"The border width ({border_width}) must be greater than or equal to the"
            f" horizontal scale ({horizontal_scale})."
        )
    width_pixels = int(size[0] / horizontal_scale) + 1
    length_pixels = int(size[1] / horizontal_scale) + 1
    border_pixels = int(border_width / horizontal_scale) + 1
    heights = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    inner_size = (
        (width_pixels - 2 * border_pixels) * horizontal_scale,
        (length_pixels - 2 * border_pixels) * horizontal_scale,
    )
    if func == "random_uniform_terrain":
        z_gen = _hf_random_uniform(sub_terrain, inner_size, horizontal_scale, vertical_scale, rng)
    else:
        z_gen = _hf_pyramid_sloped(sub_terrain, difficulty, inner_size, horizontal_scale, vertical_scale)
    heights[border_pixels:-border_pixels, border_pixels:-border_pixels] = z_gen
    return heights


def _hf_random_uniform(
    sub_terrain: dict[str, Any],
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """``hf_terrains.random_uniform_terrain`` inner grid (difficulty is unused there).

    IsaacLab draws the node heights from the global ``np.random`` state; the per-tile
    ``rng`` replaces it (structural parity: same lattice, bounds and grid shape).
    """
    downsampled_scale = sub_terrain.get("downsampled_scale")
    downsampled_scale = horizontal_scale if downsampled_scale is None else float(downsampled_scale)
    if downsampled_scale < horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {downsampled_scale} < {horizontal_scale}."
        )
    noise_range = sub_terrain["noise_range"]
    width_pixels = int(size[0] / horizontal_scale)
    length_pixels = int(size[1] / horizontal_scale)
    width_downsampled = int(size[0] / downsampled_scale)
    length_downsampled = int(size[1] / downsampled_scale)
    height_min = int(float(noise_range[0]) / vertical_scale)
    height_max = int(float(noise_range[1]) / vertical_scale)
    height_step = int(float(sub_terrain["noise_step"]) / vertical_scale)

    height_range = np.arange(height_min, height_max + height_step, height_step)
    height_field_downsampled = rng.choice(height_range, size=(width_downsampled, length_downsampled))
    # The spline parameterization multiplies size by horizontal_scale (an IsaacLab
    # unit quirk kept verbatim); it cancels because both node sets span the same
    # interval.
    x = np.linspace(0, size[0] * horizontal_scale, width_downsampled)
    y = np.linspace(0, size[1] * horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
    x_upsampled = np.linspace(0, size[0] * horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, size[1] * horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    return np.rint(z_upsampled).astype(np.int16)


def _hf_pyramid_sloped(
    sub_terrain: dict[str, Any],
    difficulty: float,
    size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
) -> np.ndarray:
    """``hf_terrains.pyramid_sloped_terrain`` inner grid (no RNG; exact replica)."""
    slope_range = sub_terrain["slope_range"]
    if sub_terrain.get("inverted", False):
        slope = -float(slope_range[0]) - difficulty * (float(slope_range[1]) - float(slope_range[0]))
    else:
        slope = float(slope_range[0]) + difficulty * (float(slope_range[1]) - float(slope_range[0]))

    width_pixels = int(size[0] / horizontal_scale)
    length_pixels = int(size[1] / horizontal_scale)
    # The apex height makes the surface rise by slope/2 over the tile half-width.
    height_max = int(slope * size[0] / 2 / vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx = ((center_x - np.abs(center_x - x)) / center_x).reshape(width_pixels, 1)
    yy = ((center_y - np.abs(center_y - y)) / center_y).reshape(1, length_pixels)
    hf_raw = height_max * xx * yy

    # Flat platform: clip to the height at the platform corner.
    platform_width = int(float(sub_terrain.get("platform_width", 1.0)) / horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    return np.rint(hf_raw).astype(np.int16)


# -- mesh (box) tiles --------------------------------------------------------


def _build_stairs_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
    flat_patch_cfgs: dict[str, dict[str, Any]],
    *,
    inverted: bool,
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    """``mesh_terrains.pyramid_stairs_terrain`` / ``inverted_pyramid_stairs_terrain``:
    the exact border + concentric step-ring + platform box layout as box geoms."""
    height_range = sub_terrain["step_height_range"]
    step_height = float(height_range[0]) + difficulty * (float(height_range[1]) - float(height_range[0]))
    step_width = float(sub_terrain["step_width"])
    platform_width = float(sub_terrain.get("platform_width", 1.0))
    border_width = float(sub_terrain.get("border_width", 0.0))
    holes = bool(sub_terrain.get("holes", False))

    num_steps_x = (size[0] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps_y = (size[1] - 2 * border_width - platform_width) // (2 * step_width) + 1
    num_steps = int(min(num_steps_x, num_steps_y))
    total_height = (num_steps + 1) * step_height

    boxes: list[tuple[str, _Box]] = []
    if border_width > 0.0 and not holes:
        inner = (size[0] - 2 * border_width, size[1] - 2 * border_width)
        for side, box in zip(_BORDER_SIDES, _border_boxes(size, inner, step_height, -step_height / 2)):
            boxes.append((f"border_{side}", box))

    terrain_center = (0.5 * size[0], 0.5 * size[1], 0.0)
    terrain_size = (size[0] - 2 * border_width, size[1] - 2 * border_width)
    for k in range(num_steps):
        box_size = (
            (platform_width, platform_width)
            if holes
            else (
                terrain_size[0] - 2 * k * step_width,
                terrain_size[1] - 2 * k * step_width,
            )
        )
        if inverted:
            box_z = terrain_center[2] - total_height / 2 - (k + 1) * step_height / 2.0
            box_height = total_height - (k + 1) * step_height
        else:
            box_z = terrain_center[2] + k * step_height / 2.0
            box_height = (k + 2) * step_height
        box_offset = (k + 0.5) * step_width
        dims = (box_size[0], step_width, box_height)
        boxes.append(
            (f"step{k}_top", ((terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z), dims))
        )
        boxes.append(
            (
                f"step{k}_bottom",
                ((terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z), dims),
            )
        )
        dims = (
            (step_width, box_size[1], box_height) if holes else (step_width, box_size[1] - 2 * step_width, box_height)
        )
        boxes.append(
            (
                f"step{k}_right",
                ((terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z), dims),
            )
        )
        boxes.append(
            (
                f"step{k}_left",
                ((terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z), dims),
            )
        )

    middle_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps + 2) * step_height if not inverted else step_height,
    )
    middle_z = (
        terrain_center[2] + num_steps * step_height / 2
        if not inverted
        else terrain_center[2] - total_height - step_height / 2
    )
    boxes.append(("platform", ((terrain_center[0], terrain_center[1], middle_z), middle_dims)))
    origin_z = (num_steps + 1) * step_height if not inverted else -(num_steps + 1) * step_height

    geoms = _tile_box_geoms(boxes, row, col, size, center_w, mu_ground)
    tile = TileBuild(
        row=row,
        col=col,
        sub_terrain=name,
        difficulty=difficulty,
        origin_w=(center_w[0], center_w[1], origin_z),
        size=size,
        platform_width=platform_width,
        flat_patch_cfgs=flat_patch_cfgs,
        surface_kind="boxes",
        surface_boxes_w=_boxes_surface_w(geoms),
    )
    return tile, geoms, []


def _build_random_grid_tile(
    sub_terrain: dict[str, Any],
    name: str,
    row: int,
    col: int,
    difficulty: float,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
    flat_patch_cfgs: dict[str, dict[str, Any]],
    *,
    base_seed: int,
) -> tuple[TileBuild, list[TerrainGeom], list[HfieldAsset]]:
    """``mesh_terrains.random_grid_terrain``: border ring + one box per grid cell
    (top shifted by the cell's height offset) + center platform.

    IsaacLab draws the per-cell height offsets from the global torch RNG; the
    per-tile numpy stream replaces it (structural parity: same cell layout, bounds
    and platform). With ``holes`` the plus-sign cells are kept once (IsaacLab
    duplicates the boxes where the x- and y-bands overlap; coincident duplicates do
    not change the surface).
    """
    if size[0] != size[1]:
        raise ValueError(f"The terrain must be square. Received size: {size}.")
    height_range = sub_terrain["grid_height_range"]
    grid_height = float(height_range[0]) + difficulty * (float(height_range[1]) - float(height_range[0]))
    grid_width = float(sub_terrain["grid_width"])
    platform_width = float(sub_terrain.get("platform_width", 1.0))
    holes = bool(sub_terrain.get("holes", False))
    terrain_height = _RANDOM_GRID_TERRAIN_HEIGHT

    num_boxes_x = int(size[0] / grid_width)
    num_boxes_y = int(size[1] / grid_width)
    border_width = size[0] - min(num_boxes_x, num_boxes_y) * grid_width
    if border_width <= 0:
        raise RuntimeError("Border width must be greater than 0! Adjust the parameter 'cfg.grid_width'.")

    tile_rng = np.random.default_rng([base_seed, row, col])
    height_offsets = tile_rng.uniform(-grid_height, grid_height, size=(num_boxes_x, num_boxes_y))

    boxes: list[tuple[str, _Box]] = []
    inner = (size[0] - border_width, size[1] - border_width)
    for side, box in zip(_BORDER_SIDES, _border_boxes(size, inner, terrain_height, -terrain_height / 2)):
        boxes.append((f"border_{side}", box))
    for i in range(num_boxes_x):
        for j in range(num_boxes_y):
            if holes:
                x_lo, x_hi = border_width / 2 + i * grid_width, border_width / 2 + (i + 1) * grid_width
                y_lo, y_hi = border_width / 2 + j * grid_width, border_width / 2 + (j + 1) * grid_width
                in_x_band = x_lo > (size[0] - border_width - platform_width) / 2 and (
                    x_hi < (size[0] + border_width + platform_width) / 2
                )
                in_y_band = y_lo > (size[1] - border_width - platform_width) / 2 and (
                    y_hi < (size[1] + border_width + platform_width) / 2
                )
                if not (in_x_band or in_y_band):
                    continue
            offset = float(height_offsets[i, j])
            center = (
                border_width / 2 + (i + 0.5) * grid_width,
                border_width / 2 + (j + 0.5) * grid_width,
                (offset - terrain_height) / 2,
            )
            boxes.append((f"cell_{i}_{j}", (center, (grid_width, grid_width, terrain_height + offset))))
    platform_center = (0.5 * size[0], 0.5 * size[1], -terrain_height / 2 + grid_height / 2)
    boxes.append(("platform", (platform_center, (platform_width, platform_width, terrain_height + grid_height))))

    geoms = _tile_box_geoms(boxes, row, col, size, center_w, mu_ground)
    tile = TileBuild(
        row=row,
        col=col,
        sub_terrain=name,
        difficulty=difficulty,
        origin_w=(center_w[0], center_w[1], grid_height),
        size=size,
        platform_width=platform_width,
        platform_top_z_w=grid_height,
        flat_patch_cfgs=flat_patch_cfgs,
        surface_kind="boxes",
        surface_boxes_w=_boxes_surface_w(geoms),
    )
    return tile, geoms, []


def _tile_box_geoms(
    boxes: list[tuple[str, _Box]],
    row: int,
    col: int,
    size: tuple[float, float],
    center_w: tuple[float, float],
    mu_ground: float,
) -> list[TerrainGeom]:
    """Boxes from the [0, size] generation frame -> world-frame geoms (tile centering
    shift of -size/2 plus the tile's grid translation)."""
    geoms = []
    for suffix, ((cx, cy, cz), dims) in boxes:
        pos_w = (center_w[0] + cx - 0.5 * size[0], center_w[1] + cy - 0.5 * size[1], cz)
        geoms.append(_box_geom(f"tile_{row}_{col}_{suffix}", (pos_w, dims), mu_ground))
    return geoms
