"""Height-scan observation provider mirroring IsaacLab's ``RayCaster`` sensor.

Numpy port of the exact IsaacLab v2.3 pipeline behind the ``height_scan``
observation term:

* ``sensors.ray_caster.patterns.grid_pattern`` — parallel rays on a regular
  grid spanning ``(-size/2, size/2)`` in the sensor frame. ``ordering`` picks
  the flattening order (``"xy"``: inner loop over x, outer over y; ``"yx"``:
  the transpose); every ray shares one direction.
* ``RayCaster._initialize_rays_impl`` — the cfg offset rotation is applied to
  the ray **directions only**; the offset position translates the ray starts.
* ``RayCaster._update_buffers_impl`` — the stored sensor position
  (``data.pos_w``) is the attach body's world position plus the world drift.
  With ``ray_alignment="yaw"`` the ray starts rotate with the body's yaw only
  and the directions stay fixed (``"world"``: no rotation at all). The x/y
  components of ``ray_cast_drift`` shift the ray origins in the projection
  frame; its z component shifts the hit points.
* ``envs.mdp.observations.height_scan`` — value = sensor_z - hit_z - offset,
  where sensor_z is ``data.pos_w[:, 2]`` (world drift included,
  ``ray_cast_drift`` excluded).

Hits come from a terrain height map ``height_at(x_w, y_w) -> z_w`` (highest
surface under the point, NaN where a downward ray misses) instead of a warp
mesh raycast. A missed ray yields a ``+inf`` hit z exactly like IsaacLab's
``raycast_mesh``, so its scan value is ``-inf`` and the term's clip saturates
it. Only straight-down world-frame ray directions can be cast against a height
map; anything else raises at construction.

Drift draws use the runtime's seeded numpy RNG (IsaacLab draws them from the
torch RNG, so draw-for-draw parity is not possible); strict mode pins both
drifts to zero.

Frame suffixes: ``_b`` attach-body projection frame, ``_w`` world.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from lab2mj.env_yaml import class_name
from lab2mj.ir import HeightScannerIR, ObsGroupIR
from lab2mj.quat import heading_w_from_quat, quat_to_rotmat

# Terrain surface lookup: broadcastable (x_w, y_w) arrays -> surface z_w
# (NaN where a downward ray misses), e.g. ``TerrainBuild.height_at``.
HeightAtFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

_DOWN_W = np.array([0.0, 0.0, -1.0])
# grid_pattern's arange end epsilon (keeps the +size/2 endpoint included).
_GRID_END_EPS = 1.0e-9


def grid_pattern(pattern: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Ray starts and directions of ``patterns.grid_pattern``, each (N, 3).

    ``pattern`` is the dumped ``pattern_cfg`` dict (``func``, ``resolution``,
    ``size``, optional ``direction``/``ordering``).
    """
    func = pattern.get("func")
    if func is not None and class_name(str(func)) != "grid_pattern":
        raise NotImplementedError(f"unsupported ray-caster pattern '{func}': only grid_pattern converts")
    resolution = float(pattern["resolution"])
    size = pattern["size"]
    ordering = str(pattern.get("ordering", "xy"))
    direction = pattern.get("direction") or (0.0, 0.0, -1.0)
    if ordering not in ("xy", "yx"):
        raise ValueError(f"Ordering must be 'xy' or 'yx'. Received: '{ordering}'.")
    if resolution <= 0:
        raise ValueError(f"Resolution must be greater than 0. Received: '{resolution}'.")

    x = np.arange(-float(size[0]) / 2, float(size[0]) / 2 + _GRID_END_EPS, resolution)
    y = np.arange(-float(size[1]) / 2, float(size[1]) / 2 + _GRID_END_EPS, resolution)
    # torch.meshgrid(indexing="ij") is numpy's "ij"; IsaacLab passes "xy" for
    # ordering "xy" and "ij" for "yx" — flattening row-major reproduces both orders.
    indexing = "xy" if ordering == "xy" else "ij"
    grid_x, grid_y = np.meshgrid(x, y, indexing=indexing)

    ray_starts = np.zeros((grid_x.size, 3), dtype=np.float64)
    ray_starts[:, 0] = grid_x.reshape(-1)
    ray_starts[:, 1] = grid_y.reshape(-1)
    ray_directions = np.tile(np.asarray(direction, dtype=np.float64), (ray_starts.shape[0], 1))
    return ray_starts, ray_directions


def pattern_num_rays(pattern: dict[str, Any]) -> int:
    """Number of rays the pattern produces (= the height_scan term dimension)."""
    return int(grid_pattern(pattern)[0].shape[0])


def height_scan_extras_dims(group: ObsGroupIR, scanners: Iterable[HeightScannerIR]) -> dict[str, int]:
    """Extras dims for the group's ``height_scan`` terms servable by ``scanners``.

    Terms whose ``sensor_cfg.name`` matches no scanner are left out (the caller
    decides whether the group is then unservable).
    """
    by_name = {scanner.name: scanner for scanner in scanners}
    dims: dict[str, int] = {}
    for term in group.terms:
        if class_name(term.func) != "height_scan":
            continue
        sensor_name = (term.params.get("sensor_cfg") or {}).get("name")
        scanner = by_name.get(sensor_name)
        if scanner is not None:
            dims[term.name] = pattern_num_rays(scanner.pattern)
    return dims


class HeightScanProvider:
    """Serves one RayCaster sensor's ``height_scan`` values from terrain heights.

    Args:
        scanner: Parsed sensor cfg (manifest ``height_scanners`` entry).
        height_at: Terrain surface lookup (see :data:`HeightAtFn`).

    Raises:
        NotImplementedError: If the pattern, alignment, or ray direction cannot
            be served by a height-map raycast.
    """

    def __init__(self, scanner: HeightScannerIR, height_at: HeightAtFn) -> None:
        if scanner.ray_alignment not in ("yaw", "world"):
            raise NotImplementedError(
                f"height scanner '{scanner.name}': ray_alignment '{scanner.ray_alignment}' is not supported "
                "(a height-map raycast needs world-fixed straight-down rays: 'yaw' or 'world')"
            )
        if not callable(height_at):
            raise TypeError(f"height scanner '{scanner.name}': height_at must be callable, got {height_at!r}")
        self.name = scanner.name
        self.ray_alignment = scanner.ray_alignment
        self.max_distance = float(scanner.max_distance)
        self._height_at = height_at

        ray_starts_b, ray_directions = grid_pattern(scanner.pattern)
        # Offset rotation rotates the directions only; the offset position
        # translates the starts (RayCaster._initialize_rays_impl).
        rot_offset = quat_to_rotmat(np.asarray(scanner.offset_quat_wxyz, dtype=np.float64))
        ray_directions_w = ray_directions @ rot_offset.T
        if not np.allclose(ray_directions_w, _DOWN_W, rtol=0.0, atol=1.0e-6):
            raise NotImplementedError(
                f"height scanner '{scanner.name}': rays must point straight down (0, 0, -1) in the "
                "projection frame to cast against a terrain height map"
            )
        self.ray_starts_b = ray_starts_b + np.asarray(scanner.offset_pos, dtype=np.float64)
        self.num_rays = int(self.ray_starts_b.shape[0])

        self._drift_range = (float(scanner.drift_range[0]), float(scanner.drift_range[1]))
        ranges = [scanner.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
        self._ray_cast_drift_range = np.asarray(ranges, dtype=np.float64)
        self._drift_w = np.zeros(3, dtype=np.float64)
        self._ray_cast_drift_b = np.zeros(3, dtype=np.float64)

    def reset(self, rng: np.random.Generator | None = None, *, strict: bool = False) -> None:
        """Resample the sensor drifts (``RayCaster.reset``); strict pins them to zero."""
        if strict or rng is None:
            self._drift_w[:] = 0.0
            self._ray_cast_drift_b[:] = 0.0
            return
        self._drift_w = rng.uniform(self._drift_range[0], self._drift_range[1], size=3)
        self._ray_cast_drift_b = rng.uniform(self._ray_cast_drift_range[:, 0], self._ray_cast_drift_range[:, 1])

    def height_scan(self, attach_pos_w: np.ndarray, attach_quat_wxyz: np.ndarray, offset: float = 0.5) -> np.ndarray:
        """``height_scan`` term values (num_rays,) for the attach body's current pose.

        ``offset`` is the observation term's ``params.offset`` (IsaacLab default
        0.5). Missed rays return ``-inf`` (the term clip saturates them).
        """
        pos_w = np.asarray(attach_pos_w, dtype=np.float64) + self._drift_w
        # data.pos_w is stored before the ray_cast_drift shift, so the scan's
        # sensor height carries the world drift only.
        sensor_z_w = pos_w[2]
        drift_b = self._ray_cast_drift_b
        if self.ray_alignment == "yaw":
            # heading_w_from_quat is the yaw angle isaaclab.utils.math.yaw_quat extracts.
            yaw = heading_w_from_quat(np.asarray(attach_quat_wxyz, dtype=np.float64))
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            origin_x_w = pos_w[0] + cos_yaw * drift_b[0] - sin_yaw * drift_b[1]
            origin_y_w = pos_w[1] + sin_yaw * drift_b[0] + cos_yaw * drift_b[1]
            x_b, y_b = self.ray_starts_b[:, 0], self.ray_starts_b[:, 1]
            x_w = cos_yaw * x_b - sin_yaw * y_b + origin_x_w
            y_w = sin_yaw * x_b + cos_yaw * y_b + origin_y_w
        else:  # "world": no rotation of starts or drift
            x_w = self.ray_starts_b[:, 0] + pos_w[0] + drift_b[0]
            y_w = self.ray_starts_b[:, 1] + pos_w[1] + drift_b[1]
        start_z_w = self.ray_starts_b[:, 2] + pos_w[2]

        hit_z_w = np.asarray(self._height_at(x_w, y_w), dtype=np.float64)
        if hit_z_w.shape != x_w.shape:
            raise ValueError(
                f"height scanner '{self.name}': height_at returned shape {hit_z_w.shape}, expected {x_w.shape} "
                "(expected signature height_at(x_w, y_w) -> z_w)"
            )
        # A downward ray hits the surface iff it lies at/below the ray start and
        # within max_distance; everything else misses (inf, like raycast_mesh).
        miss = ~np.isfinite(hit_z_w) | (hit_z_w > start_z_w) | (start_z_w - hit_z_w > self.max_distance)
        hit_z_w = np.where(miss, np.inf, hit_z_w) + drift_b[2]
        return sensor_z_w - hit_z_w - float(offset)
