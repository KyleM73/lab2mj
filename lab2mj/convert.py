"""CLI converter: IsaacLab run directory -> self-contained MuJoCo bundle.

Usage::

    uv run lab2mj-convert --run <run_dir> [--usd PATH] [--out DIR] [--dump NPZ] [--substeps N]

``<run_dir>`` must contain ``params/env.yaml`` (the fully-resolved config both
contact_lab's and IsaacLab's rsl_rl ``train.py`` dump); a direct path to an
``env.yaml`` file is also accepted. ``--substeps N`` integrates each Isaac
physics step as N MuJoCo steps at ``physics_dt / N`` with contact solref
authored against that substep timestep (``2 * physics_dt / N``): PhysX TGS
resolves a touchdown impulse iteratively within one step, while a single
MuJoCo step at ``physics_dt`` can only do so through soft contact, so finer
contact integration substantially reduces open-loop touchdown divergence
(default 4). The policy rate and the 1/physics_dt actuator-torque/delay
cadence are unchanged. The robot USD is taken from ``--usd``, else
from the yaml's ``usd_path`` (http(s) URLs are downloaded once and cached under
``data/usd_cache/``). The output bundle (``scene.xml`` + ``robot.xml`` +
``assets/`` + ``manifest.json``) is written to ``--out`` (default
``<run_dir>/mj_bundle``).

With ``--dump`` the converter cross-checks the bundle against an Isaac
reference dump (joint order, per-body masses, env-0 origin, gravity) and
reports mismatches as non-fatal warnings. If the dump carries a
measured-terrain record (generator terrains dumped by
``lab2mj/isaac/dump_reference.py``), the bundle terrain is built from
that exact measured instance instead of procedural regeneration — the record
is copied into the bundle (``assets/terrain_measured.npz``) and the manifest
terrain entry gains ``source = "measured-from-dump"`` plus grid metadata (see
``lab2mj.terrain``). ``--terrain_exact`` (default) collides the exact recorded
mesh inside the dump trajectory's window — one convex prism per upward
triangle, so stair risers are true vertical walls — falling back automatically
when the window's triangle count exceeds the prism cap. On fallback (or
``--no-terrain_exact``), ``--terrain_collision_res`` (default ``auto``)
re-raycasts the recorded mesh onto a finer collision grid around the dump
trajectory so risers stay near-vertical instead of one-record-cell ramps
(``--terrain_fine_margin`` / ``--terrain_fine_full`` shape the fine window).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lab2mj import bundle
from lab2mj.actuators import NET_MODELS, ActuatorSet, resolve_matching_names, resolve_matching_names_values
from lab2mj.commands import command_dim
from lab2mj.env import build_low_level_obs_pipeline, build_robot_map, resolve_action_arrays
from lab2mj.env_yaml import class_name, find_robot_cfg, load_env_yaml, parse_env_dict
from lab2mj.events import write_root_state
from lab2mj.heightscan import height_scan_extras_dims
from lab2mj.ir import EnvIR
from lab2mj.obs import ObsPipeline, extras_term_names
from lab2mj.terrain import MeasuredTerrainData, add_to_spec, build_terrain, pair_friction, plan_measured_exact_window
from lab2mj.usd2mjcf import RobotIR, build_mjcf, parse_usd, verify_build
from lab2mj.usd2mjcf.builder import CONTACT_PROFILES


def _data_dir() -> Path:
    """Root for downloaded USDs / actuator nets / low-level policies.

    ``$LAB2MJ_DATA_DIR`` wins. Otherwise ``<repo>/data`` when running from a source
    checkout (so convert works from any cwd), else ``./data`` for an installed copy
    (site-packages has no repo root to anchor to).
    """
    env = os.environ.get("LAB2MJ_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if (repo / "pyproject.toml").exists():
        return repo / "data"
    return Path.cwd() / "data"


DATA_DIR = _data_dir()
USD_CACHE_DIR = DATA_DIR / "usd_cache"
NET_CACHE_DIR = DATA_DIR / "actuator_nets"
POLICY_CACHE_DIR = DATA_DIR / "policy_cache"

# Actuator-net weights are copied into the bundle under this directory so bundles
# stay self-contained; manifest actuator entries record the relative path.
NET_BUNDLE_DIR = "assets/actuator_nets"

# PreTrainedPolicyAction low-level policies are copied here; the manifest action
# entry records the relative path (``ir.policy_bundle_path``).
POLICY_BUNDLE_DIR = "assets/policies"

# Measured-terrain record (height grid + terrain mesh) copied out of the reference
# dump so the bundle stays self-contained; manifest terrain.measured.file records it.
MEASURED_TERRAIN_NAME = "assets/terrain_measured.npz"

# Measured-terrain collision resampling. The recorded height grid approximates
# vertical mesh faces (stair risers) as one-cell ramps; re-raycasting the recorded
# mesh onto a finer collision grid shrinks that deviation to one fine cell. "auto"
# picks the finest of (record/4, record/2, record) whose fine-grid node count stays
# within MEASURED_FINE_NODE_BUDGET; the fine grid is windowed to the dump
# trajectory's xy extent + MEASURED_FINE_MARGIN_M (recorded resolution elsewhere)
# unless the caller forces full extent. The budget bounds scene.xml growth — MjSpec
# embeds hfield elevations as XML text at roughly 6-10 bytes per node.
MEASURED_FINE_NODE_BUDGET = 4_000_000
MEASURED_FINE_MARGIN_M = 2.0

# MuJoCo physics substeps per Isaac physics step. Default: the articulation's PhysX
# TGS ``solver_position_iteration_count`` from env.yaml — TGS sub-integrates contact
# positionally that many times per physics step, so matching it reproduces the
# reference's effective contact granularity. Fallback when the config does not
# specify one; substepping without solref rescaling does not help (MuJoCo clamps
# solref timeconst to 2 * timestep, so a finer timestep is the only way to harden
# contact — the builder authors solref against the substep dt).
DEFAULT_SUBSTEPS = 4


def _cached_download(url: str, cache_dir: Path, override_hint: str) -> Path:
    """Download ``url`` into ``cache_dir`` (or reuse the cached copy).

    Cached downloads are keyed by URL (``<urlsha8>_<basename>``) and take
    precedence, so two URLs sharing a basename never shadow each other. A
    plain ``<basename>`` file is accepted as a fallback for pre-seeded caches,
    with a notice — its content is not checked against the URL.
    """
    basename = url.rsplit("/", 1)[-1]
    url_sha = hashlib.sha256(url.encode()).hexdigest()[:8]
    keyed = cache_dir / f"{url_sha}_{basename}"
    if keyed.is_file():
        return keyed
    plain = cache_dir / basename
    if plain.is_file():
        print(
            f"[convert] NOTICE: using pre-seeded cache file {plain} for {url} "
            f"(not keyed to this URL; delete it or {override_hint} to force the exact asset)"
        )
        return plain
    _fetch_atomic(url, keyed)
    return keyed


def _fetch_atomic(url: str, dest: Path) -> None:
    """Fetch ``url`` to ``dest`` via a temp file + rename, so an interrupted
    download never leaves a truncated file later converts treat as a cache hit."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[convert] downloading {url} -> {dest}")
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60.0) as response, open(tmp, "wb") as f:
            f.write(response.read())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def resolve_usd(usd_arg: str | Path | None, ir: EnvIR, cache_dir: Path = USD_CACHE_DIR) -> Path:
    """Resolve the robot USD: explicit path > local yaml path > cache > download."""
    if usd_arg is not None:
        path = Path(usd_arg)
        if not path.is_file():
            raise FileNotFoundError(f"--usd {path} does not exist")
        return path
    if ir.usd_path is None:
        raise ValueError("env.yaml has no robot usd_path; pass --usd explicitly")
    if not ir.usd_path.startswith(("http://", "https://")):
        path = Path(ir.usd_path)
        if not path.is_file():
            raise FileNotFoundError(f"env.yaml usd_path {path} does not exist locally; pass --usd")
        return path
    return _download_usd_with_dependencies(ir.usd_path, cache_dir)


def _download_usd_with_dependencies(url: str, cache_dir: Path, max_files: int = 64) -> Path:
    """Download a USD and, recursively, its relative composition dependencies.

    Robot USDs reference companion layers by relative path (e.g.
    ``./Props/instanceable_meshes.usd``), which USD resolves relative to the
    referencing layer's directory — so each root URL gets its own cache
    directory (``<urlsha8>_<stem>/``) mirroring the remote layout, and
    dependencies are fetched next to it. Only relative references are
    followed; absolute paths and URLs inside layers are reported and skipped.
    """
    from pxr import Sdf

    basename = url.rsplit("/", 1)[-1]
    plain = cache_dir / basename
    if plain.is_file():
        print(
            f"[convert] NOTICE: using pre-seeded cache file {plain} for {url} "
            f"(not keyed to this URL; delete it or pass --usd to force the exact asset)"
        )
        return plain
    url_sha = hashlib.sha256(url.encode()).hexdigest()[:8]
    root_dir = cache_dir / f"{url_sha}_{Path(basename).stem}"
    root_path = root_dir / basename

    pending: list[tuple[str, Path]] = [(url, root_path)]
    seen: set[str] = set()
    fetched = 0
    while pending:
        file_url, dest = pending.pop()
        if file_url in seen:
            continue
        seen.add(file_url)
        if not dest.is_file():
            if fetched >= max_files:
                raise RuntimeError(f"USD dependency download exceeded {max_files} files for {url}")
            _fetch_atomic(file_url, dest)
            fetched += 1
        layer = Sdf.Layer.FindOrOpen(str(dest))
        if layer is None:
            continue
        for dep in layer.GetCompositionAssetDependencies():
            if dep.startswith(("http://", "https://")) or dep.startswith("/"):
                print(f"[convert] NOTICE: skipping non-relative USD dependency {dep!r} in {dest.name}")
                continue
            dep_url = urllib.parse.urljoin(file_url, dep)
            dep_dest = (dest.parent / dep).resolve()
            if not dep_dest.is_relative_to(root_dir.resolve()):
                raise ValueError(f"USD dependency {dep!r} in {dest.name} escapes the cache directory {root_dir}")
            pending.append((dep_url, dep_dest))
    return root_path


def resolve_network_file(source: str, cache_dir: Path = NET_CACHE_DIR, what: str = "actuator network_file") -> Path:
    """Resolve a TorchScript file reference (``network_file``, low-level policy): local path > cache > download."""
    if not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"{what} {path} does not exist locally")
        return path
    return _cached_download(source, cache_dir, f"place the file at {cache_dir}")


def _resolve_actuator_nets(ir: EnvIR, cache_dir: Path = NET_CACHE_DIR) -> dict[str, Path]:
    """Resolve every actuator-net group's weights; returns bundle-relative -> local path.

    Sets each group's ``network_bundle_path`` so the manifest records where the
    runtime finds the copied file inside the bundle. Basenames are shared when
    groups reference the same source and disambiguated by group name otherwise.
    """
    net_files: dict[str, Path] = {}
    source_of: dict[str, str] = {}
    for group in ir.actuators:
        if group.model not in NET_MODELS:
            continue
        if not group.network_file:
            raise ValueError(f"actuator group '{group.name}' is an actuator-net group but has no network_file")
        local = resolve_network_file(group.network_file, cache_dir=cache_dir)
        name = group.network_file.rsplit("/", 1)[-1]
        if source_of.get(name, group.network_file) != group.network_file:
            name = f"{group.name}_{name}"
        source_of[name] = group.network_file
        rel_path = f"{NET_BUNDLE_DIR}/{name}"
        group.network_bundle_path = rel_path
        net_files[rel_path] = local
    return net_files


def _resolve_low_level_policy(ir: EnvIR, cache_dir: Path = POLICY_CACHE_DIR) -> dict[str, Path]:
    """Resolve a PreTrainedPolicyAction low-level policy; returns bundle-relative -> local path.

    Sets ``ir.action.policy_bundle_path`` so the manifest records where the
    runtime finds the copied TorchScript file inside the bundle. Empty dict for
    plain action terms.
    """
    if ir.action.policy_path is None:
        return {}
    local = resolve_network_file(ir.action.policy_path, cache_dir=cache_dir, what="low-level policy")
    rel_path = f"{POLICY_BUNDLE_DIR}/{ir.action.policy_path.rsplit('/', 1)[-1]}"
    ir.action.policy_bundle_path = rel_path
    return {rel_path: local}


def _check_height_scanner_bodies(ir: EnvIR, body_map: dict[str, str]) -> None:
    """Fail fast when an obs-consumed height scanner attaches to an untracked body.

    The runtime reads the scanner pose from the attach body's MuJoCo frame, so
    the body must survive conversion under its own name (a body welded into its
    parent loses its frame). Scanners no observation term consumes are recorded
    for provenance only and never checked.
    """
    groups = list(ir.obs_groups)
    if ir.action.low_level_obs is not None:
        groups.append(ir.action.low_level_obs)
    used = {
        (term.params.get("sensor_cfg") or {}).get("name")
        for group in groups
        for term in group.terms
        if class_name(term.func) == "height_scan"
    }
    for scanner in ir.height_scanners:
        if scanner.name not in used:
            continue
        target = body_map.get(scanner.attach_body_name)
        if target is None:
            raise ValueError(
                f"height scanner '{scanner.name}' attaches to '{scanner.attach_body_name}' "
                f"(prim_path {scanner.prim_path!r}), which is not an articulation body"
            )
        if target != scanner.attach_body_name:
            raise ValueError(
                f"height scanner '{scanner.name}' attaches to '{scanner.attach_body_name}', which the "
                f"converter welds into '{target}'; the welded frame is not tracked at runtime"
            )


def _soft_joint_pos_limit_factor(raw: dict[str, Any]) -> float:
    _, robot = find_robot_cfg(raw.get("scene") or {})
    factor = robot.get("soft_joint_pos_limit_factor")
    return 1.0 if factor is None else float(factor)


def _resolve_regex_defaults(patterns: dict[str, float], names: list[str], what: str) -> np.ndarray:
    out = np.zeros(len(names), dtype=np.float64)
    if patterns:
        indices, values = resolve_matching_names_values(patterns, names, what=what)
        out[indices] = values
    return out


def _joint_limits(model: mujoco.MjModel, isaac_joint_order: list[str], factor: float) -> np.ndarray:
    """Soft joint position limits (J, 2) in Isaac order (mid +- factor * half-range)."""
    limits = np.empty((len(isaac_joint_order), 2), dtype=np.float64)
    for k, name in enumerate(isaac_joint_order):
        jid = model.joint(name).id
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
            limits[k] = (mid - factor * half, mid + factor * half)
        else:
            limits[k] = (-np.inf, np.inf)
    return limits


def _joint_vel_limits(ir: EnvIR, isaac_joint_order: list[str]) -> np.ndarray:
    """Soft joint velocity limits (J,) from the actuator configs (inf when unset)."""
    out = np.full(len(isaac_joint_order), np.inf, dtype=np.float64)
    for group in ir.actuators:
        if group.velocity_limit is None:
            continue
        ids = resolve_matching_names(group.joint_names_expr, isaac_joint_order, f"actuator '{group.name}'")
        names = [isaac_joint_order[i] for i in ids]
        if isinstance(group.velocity_limit, dict):
            indices, values = resolve_matching_names_values(group.velocity_limit, names, "velocity_limit")
            for local, value in zip(indices, values):
                out[ids[local]] = value
        else:
            for i in ids:
                out[i] = float(group.velocity_limit)
    return out


def _mj_joint_order(model: mujoco.MjModel) -> list[str]:
    order = []
    for jid in range(model.njnt):
        if int(model.jnt_type[jid]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            order.append(model.joint(jid).name)
    return order


def _geom_map(robot: RobotIR, isaac_body_names: list[str]) -> dict[str, list[str]]:
    """Isaac body name -> collision geom names, from the parser-recorded source bodies."""
    out: dict[str, list[str]] = {name: [] for name in isaac_body_names}
    for geom in robot.geoms:
        if geom.is_collision:
            out[geom.source_body].append(geom.name)
    return out


def _physx_plant_from_npz(data: Any) -> dict[str, tuple[float, np.ndarray, np.ndarray]] | None:
    """Body name -> (mass, com_b, inertia_b 3x3) from a dump's PhysX plant record.

    ``body_coms_physx`` rows are (x, y, z, qx, qy, qz, qw); the quaternion is the
    principal-axes rotation, already folded into the full link-frame inertia matrix
    ``body_inertias_physx`` carries, so only the position is needed here. Returns
    None when the npz predates plant recording.
    """
    if "body_inertias_physx" not in data or "body_names" not in data:
        return None
    plant: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    masses = np.asarray(data["body_masses"], dtype=np.float64)
    coms = np.asarray(data["body_coms_physx"], dtype=np.float64)
    inertias = np.asarray(data["body_inertias_physx"], dtype=np.float64)
    for i, name in enumerate(data["body_names"]):
        plant[str(name)] = (float(masses[i]), coms[i, :3].copy(), inertias[i].reshape(3, 3).copy())
    return plant


def _trajectory_window_w(dump: Any, margin: float) -> tuple[float, float, float, float]:
    """Dump-trajectory xy extent + margin, world frame: (xmin, xmax, ymin, ymax)."""
    root_xy = np.asarray(dump["root_pos_w"], dtype=np.float64)[:, :2]
    return (
        float(root_xy[:, 0].min() - margin),
        float(root_xy[:, 0].max() + margin),
        float(root_xy[:, 1].min() - margin),
        float(root_xy[:, 1].max() + margin),
    )


def _measured_collision_plan(
    measured: "MeasuredTerrainData",
    dump: Any,
    collision_res: float | str | None,
    fine_margin: float,
    fine_full: bool,
) -> tuple[float | None, tuple[float, float, float, float] | None]:
    """Resolve the fine collision resolution + window for a measured terrain.

    Returns ``(resolution, window_w)`` for :func:`build_terrain`;
    ``(None, None)`` keeps the recorded grid as the collision surface. The window
    is the dump trajectory's xy extent + ``fine_margin`` (``None`` = full extent
    when ``fine_full``); "auto" picks the finest of (record / 4, record / 2,
    record) whose windowed node count fits MEASURED_FINE_NODE_BUDGET.
    """
    record_res = float(measured.grid_resolution)
    window: tuple[float, float, float, float] | None = None
    if not fine_full:
        window = _trajectory_window_w(dump, fine_margin)
    if collision_res == "record" or collision_res is None:
        return None, None
    nx, ny = measured.height_grid.shape
    if window is None:
        span_x, span_y = (nx - 1) * record_res, (ny - 1) * record_res
    else:
        # Snapped-outward window span, mirroring terrain._build_measured.
        span_x = min((nx - 1) * record_res, (np.ceil((window[1] - window[0]) / record_res) + 1) * record_res)
        span_y = min((ny - 1) * record_res, (np.ceil((window[3] - window[2]) / record_res) + 1) * record_res)
    if collision_res == "auto":
        for candidate in (record_res / 4.0, record_res / 2.0):
            nodes = (int(np.ceil(span_x / candidate)) + 1) * (int(np.ceil(span_y / candidate)) + 1)
            if nodes <= MEASURED_FINE_NODE_BUDGET:
                return candidate, window
        return None, None
    resolution = float(collision_res)
    if resolution >= record_res:
        return None, None
    nodes = (int(np.ceil(span_x / resolution)) + 1) * (int(np.ceil(span_y / resolution)) + 1)
    if nodes > MEASURED_FINE_NODE_BUDGET:
        raise ValueError(
            f"measured collision resolution {resolution} m needs ~{nodes} hfield nodes over a "
            f"{span_x:.1f} x {span_y:.1f} m extent (budget {MEASURED_FINE_NODE_BUDGET}); pass a coarser "
            "resolution or drop --terrain_fine_full so the fine grid windows to the dump trajectory"
        )
    return resolution, window


def _resolve_obs_layouts(
    ir: EnvIR,
    manifest: dict[str, Any],
    isaac_joint_order: list[str],
    command_dims: dict[str, int],
    resample_s: dict[str, float],
) -> int:
    """Record every obs group's per-term layout in the manifest; returns the policy obs dim.

    Degradation contract: only the policy group must be servable by the MuJoCo
    runtime. Any other group (critic, estimator targets) that is not — extras-backed
    terms without a runtime provider (wrench ground truth, ...), unknown funcs,
    unsupported cfg layouts — records a null layout so conversion still succeeds
    for the runnable policy; the same problem on the policy group aborts.
    """
    action_dim = manifest["action"]["dim"]
    policy_obs_dim = 0

    def degrade(group_name: str, reason: str) -> None:
        print(f"[convert] WARNING: obs group '{group_name}' {reason}; recording a null layout")
        manifest["obs"]["layouts"][group_name] = None
        manifest["obs"]["obs_dims"][group_name] = None

    for group in ir.obs_groups:
        # height_scan terms backed by a recorded scanner are served by the runtime's
        # HeightScanProvider; their dims resolve from the sensor's ray pattern
        # (NotImplementedError for non-grid patterns — degradable like any other
        # unservable layout).
        try:
            extras_dims = height_scan_extras_dims(group, ir.height_scanners)
        except NotImplementedError as err:
            if group.name == "policy":
                raise
            degrade(group.name, f"is not servable ({err})")
            continue
        extras = [name for name in extras_term_names(group) if name not in extras_dims]
        if extras:
            if group.name == "policy":
                raise ValueError(
                    f"policy obs group terms {extras} need runtime extras providers; "
                    "the converted MuJoCo runtime cannot serve this policy observation"
                )
            degrade(group.name, f"has extras-backed terms {extras} (not runnable by the MuJoCo runtime)")
            continue
        try:
            pipeline = ObsPipeline(
                group,
                num_joints=len(isaac_joint_order),
                action_dim=action_dim,
                command_dims=command_dims,
                command_resample_time_s=resample_s,
                extras_dims=extras_dims,
                entity=ir.robot_entity,
            )
            term_slices = pipeline.layout()
        except ValueError as err:
            if group.name == "policy":
                raise
            degrade(group.name, f"is not servable ({err})")
            continue
        term_of = {term.name: term for term in group.terms}
        layout = []
        for name, sl, dim in term_slices:
            term = term_of[name]
            layout.append(
                {
                    "name": name,
                    "func": term.func,
                    "start": sl.start,
                    "stop": sl.stop,
                    "dim": dim,
                    "noise": None if term.noise is None else term.noise.to_dict(),
                    "clip": term.clip,
                    "scale": term.scale,
                    "history_length": term.history_length,
                }
            )
        manifest["obs"]["layouts"][group.name] = layout
        manifest["obs"]["obs_dims"][group.name] = pipeline.obs_dim
        if group.name == "policy":
            policy_obs_dim = pipeline.obs_dim
    if "policy" not in manifest["obs"]["groups"]:
        raise ValueError("env.yaml defines no 'policy' observation group")
    return policy_obs_dim


def parse_legacy_friction(spec: str) -> dict[str, tuple[float, float, float]]:
    """Parse ``"regex=static,dynamic,viscous;regex=..."`` (N*m, N*m, N*m*s/rad) into a dict."""
    out: dict[str, tuple[float, float, float]] = {}
    for item in filter(None, (part.strip() for part in spec.split(";"))):
        pattern, sep, values = item.rpartition("=")
        numbers = [float(v) for v in values.split(",")]
        if not sep or len(numbers) != 3:
            raise ValueError(f"--legacy_friction entry '{item}' must read 'regex=static,dynamic,viscous'")
        out[pattern] = (numbers[0], numbers[1], numbers[2])
    return out


def apply_legacy_friction(
    ir: EnvIR, equivalents: dict[str, tuple[float, float, float]] | None, isaac_joint_order: list[str]
) -> dict[str, Any] | None:
    """Realize contact_lab's ``set_legacy_joint_friction`` event as fixed joint friction.

    PhysX's legacy friction coefficient bounds a joint's friction by a load-dependent
    wrench norm, which MuJoCo has no counterpart for. ``equivalents`` gives fixed
    (static, dynamic, viscous) efforts per joint-name regex — the event's own keys —
    and is written into the actuator groups (mutating ``ir``), so the converter's
    regular friction path (``dof_frictionloss``, damping, stiction) carries it.
    Returns the manifest record, or None when the env has no such event.
    """
    events = [e for e in ir.events if class_name(e.func) == "set_legacy_joint_friction"]
    if not events:
        if equivalents:
            raise ValueError("--legacy_friction given, but the env has no set_legacy_joint_friction event")
        return None
    coefficients: dict[str, float] = {}
    for event in events:
        coefficients.update({str(k): float(v) for k, v in event.params["coefficients"].items()})
    if equivalents is None:
        raise ValueError(
            f"the env applies PhysX's legacy joint friction {coefficients} (set_legacy_joint_friction), which "
            "MuJoCo cannot reproduce; pass --legacy_friction 'regex=static,dynamic,viscous;...' with fixed "
            "N*m equivalents for the same regex keys"
        )
    if set(equivalents) != set(coefficients):
        raise ValueError(f"--legacy_friction keys {sorted(equivalents)} must equal the event's {sorted(coefficients)}")
    for group in ir.actuators:
        names = [isaac_joint_order[i] for i in resolve_matching_names(group.joint_names_expr, isaac_joint_order)]
        patterns = {p: v for p, v in equivalents.items() if any(re.fullmatch(p, n) for n in names)}
        if not patterns:
            continue
        if any(getattr(group, f) not in (None, 0, 0.0) for f in ("friction", "dynamic_friction", "viscous_friction")):
            raise ValueError(f"actuator group '{group.name}': legacy-friction joints must carry no actuator friction")
        group.friction = {p: v[0] for p, v in patterns.items()}
        group.dynamic_friction = {p: v[1] for p, v in patterns.items()}
        group.viscous_friction = {p: v[2] for p, v in patterns.items()}
    return {"coefficients": coefficients, "equivalents": {p: list(v) for p, v in equivalents.items()}}


def convert_run(
    run: str | Path,
    *,
    usd: str | Path | None = None,
    out: str | Path | None = None,
    dump: str | Path | None = None,
    substeps: int | None = None,
    contact_profile: str = "auto",
    contact_solimp: tuple[float, ...] | None = None,
    contact_impratio: float | None = None,
    contact_solref: tuple[float, float] | None = None,
    physx_plant: str | Path | None = None,
    terrain_exact: bool = True,
    terrain_collision_res: float | str | None = "auto",
    terrain_fine_margin: float = MEASURED_FINE_MARGIN_M,
    terrain_fine_full: bool = False,
    legacy_friction: dict[str, tuple[float, float, float]] | None = None,
) -> Path:
    """Convert an IsaacLab run into a MuJoCo bundle; returns the bundle directory.

    ``legacy_friction``: fixed (static, dynamic, viscous) joint friction per regex for an
    env that applies PhysX's legacy friction model (:func:`apply_legacy_friction`).

    ``substeps`` MuJoCo steps integrate each Isaac physics step (model timestep
    and default contact solref author against the substep dt); ``None`` resolves
    to the PhysX TGS ``solver_position_iteration_count``, else ``DEFAULT_SUBSTEPS``.

    ``contact_profile``: ``"auto"`` (default) follows
    :meth:`ActuatorSet.contact_profile`; force ``"default"`` / ``"engagement"``
    for robots the heuristic was not validated on. Recorded as
    ``sim.contact_profile``. ``contact_solimp`` / ``contact_impratio`` /
    ``contact_solref`` author explicit values on top of the profile
    (calibration output) and are recorded when given; ``contact_solref``
    replaces the default ``(2 * substep_dt, 1.0)``.

    ``physx_plant`` is an npz carrying ``body_inertias_physx``/``body_coms_physx``
    (any reference or freespace dump of the same robot); the PhysX-resolved mass
    properties replace the USD-authored ones (PhysX silently recomputes invalid
    authored inertias). When ``dump`` itself carries the record it is used
    automatically; an explicit ``physx_plant`` takes precedence.

    ``terrain_exact`` (default) collides the exact recorded mesh inside the dump
    trajectory's window — one convex prism per upward triangle, so stair risers are
    true vertical walls — with recorded-resolution hfield slabs elsewhere; it falls
    back to hfield collision automatically when the window's triangle count exceeds
    the prism cap or ``terrain_fine_full`` is requested.

    ``terrain_collision_res`` controls the measured-terrain collision grid (used
    only when ``dump`` carries a measured-terrain record): "auto" (default, fits
    MEASURED_FINE_NODE_BUDGET), "record"/None for the recorded grid as-is, or an
    explicit node spacing [m]; the module docstring covers the fine-window
    mechanics (``terrain_fine_margin`` / ``terrain_fine_full``).
    """
    if substeps is not None and substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}")
    if contact_profile != "auto" and contact_profile not in CONTACT_PROFILES:
        raise ValueError(f"unknown contact_profile {contact_profile!r}; expected 'auto' or one of {CONTACT_PROFILES}")
    run = Path(run)
    if run.is_dir():
        env_yaml_path = run / "params" / "env.yaml"
        if not env_yaml_path.is_file():
            raise FileNotFoundError(f"{env_yaml_path} not found")
        out_dir = Path(out) if out is not None else run / "mj_bundle"
    else:
        env_yaml_path = run
        if out is None:
            raise ValueError("--out is required when --run points at an env.yaml file directly")
        out_dir = Path(out)

    raw = load_env_yaml(env_yaml_path)
    ir = parse_env_dict(raw)
    if substeps is None:
        substeps = ir.solver_position_iterations or DEFAULT_SUBSTEPS
    usd_path = resolve_usd(usd, ir)
    # Actuator-net weights: download/cache and record bundle-relative paths in the IR
    # (mutates group.network_bundle_path) so the manifest below carries them.
    net_files = _resolve_actuator_nets(ir)
    # Low-level policy of a PreTrainedPolicyAction term (mutates action.policy_bundle_path).
    policy_files = _resolve_low_level_policy(ir)

    # -- robot model -------------------------------------------------------------------
    plant = None
    if physx_plant is not None:
        plant = _physx_plant_from_npz(np.load(Path(physx_plant)))
        if plant is None:
            raise ValueError(f"--physx_plant {physx_plant} carries no body_inertias_physx record")
    elif dump is not None:
        plant = _physx_plant_from_npz(np.load(Path(dump)))
    robot = parse_usd(usd_path, physx_plant=plant)
    # A Y-up robot USD parses (the up-axis correction is baked into the root body
    # frame, so the compiled keyframe is upright), but the corrected MuJoCo root
    # frame then differs from the Isaac root link frame that manifest resets,
    # reference-dump states, and every body-frame observation assume.
    if not np.allclose(robot.root_link.quat_p, (1.0, 0.0, 0.0, 0.0), atol=1e-12):
        raise ValueError(
            "the robot USD stage is not Z-up: the up-axis correction would make the MuJoCo root "
            "body frame differ from the Isaac root link frame that dumps, resets, and body-frame "
            "observations assume — re-export the robot USD with upAxis Z to convert it for sim2sim"
        )
    legacy_friction_record = apply_legacy_friction(ir, legacy_friction, robot.isaac_joint_order)
    actuator_set = ActuatorSet.from_ir(ir.actuators, robot.isaac_joint_order)
    joint_overrides = actuator_set.builder_overrides()
    # USD-authored joint armature is the PhysX default whenever the actuator cfg
    # leaves armature unset (IsaacLab armature=None semantics; measured: dropping
    # h1's authored 0.1 put 2.2 rad of free-space plant error inside 0.5 s).
    cfg_armature_joints: set[str] = set()
    for group in ir.actuators:
        if group.armature is not None:
            ids = resolve_matching_names(group.joint_names_expr, robot.isaac_joint_order)
            cfg_armature_joints.update(robot.isaac_joint_order[i] for i in ids)
    for joint in robot.joints:
        if joint.armature_usd is not None and joint.name not in cfg_armature_joints:
            joint_overrides.setdefault(joint.name, {}).setdefault("armature", float(joint.armature_usd))
            print(
                f"[convert] joint '{joint.name}': armature {joint.armature_usd:g} from the USD "
                "(actuator cfg authors none)"
            )
    # PhysX implicit PD drives are unconditionally stable; a purely explicit kd torque at
    # the same physics dt is not (kd * dt can exceed the joint's reflected inertia). For
    # implicit_pd groups the kd term is therefore authored as MuJoCo ``dof_damping``,
    # which the Euler integrator integrates implicitly like PhysX does; the runtime adds
    # ``+kd*qd`` back into ctrl so the effort clamp still applies to the full PD torque.
    for group in actuator_set.groups:
        if group.model != "implicit_pd":
            continue
        for local_idx, joint_name in enumerate(group.joint_names):
            kd = float(actuator_set.kd[group.joint_ids[local_idx]])
            if kd > 0.0:
                # Composes with any viscous-friction damping the overrides already carry.
                entry = joint_overrides.setdefault(joint_name, {})
                entry["damping"] = entry.get("damping", 0.0) + kd
    default_joint_pos_isaac = _resolve_regex_defaults(
        ir.robot_init.joint_pos, robot.isaac_joint_order, "init_state.joint_pos"
    )
    default_joint_vel_isaac = _resolve_regex_defaults(
        ir.robot_init.joint_vel, robot.isaac_joint_order, "init_state.joint_vel"
    )

    # -- terrain layout first: env 0's origin decides where the robot spawns ------------
    # Replicates the TerrainImporter layout with the training run's num_envs /
    # env_spacing / max_init_terrain_level, so a grid-plane env 0 gets its real grid
    # origin and generator terrain spawns on a tile origin instead of the map center.
    # A --dump carrying a measured-terrain record overrides the procedural
    # regeneration entirely: the bundle gets the exact terrain instance (and env-0
    # origin) the reference robot walked on.
    measured_terrain = None
    measured_collision_res: float | None = None
    measured_collision_window: tuple[float, float, float, float] | None = None
    measured_exact_window: tuple[float, float, float, float] | None = None
    if dump is not None:
        dump_npz = np.load(Path(dump))
        measured_terrain = MeasuredTerrainData.from_npz(dump_npz)
        if measured_terrain is not None:
            if terrain_exact and not terrain_fine_full:
                # Exact collision replaces the windowed fine hfield with one convex
                # prism per upward mesh triangle; the terrain planner runs the same
                # snapping/classification as the build, so a returned window cannot
                # fail to build.
                measured_exact_window = plan_measured_exact_window(
                    measured_terrain, _trajectory_window_w(dump_npz, terrain_fine_margin)
                )
            elif terrain_exact:
                print("[convert] NOTICE: --terrain_fine_full requests full-extent collision; exact prisms need a")
                print("[convert]         trajectory window — falling back to hfield collision")
            if measured_exact_window is None:
                measured_collision_res, measured_collision_window = _measured_collision_plan(
                    measured_terrain, dump_npz, terrain_collision_res, terrain_fine_margin, terrain_fine_full
                )
    terrain_rng = np.random.default_rng(0 if ir.seed is None else int(ir.seed))
    terrain_num_envs, terrain_env_spacing, terrain_max_init_level = 1, None, None
    if ir.terrain is not None:
        terrain_num_envs = ir.terrain.num_envs or 1
        terrain_env_spacing = ir.terrain.env_spacing
        terrain_max_init_level = ir.terrain.max_init_terrain_level
    terrain_build = build_terrain(
        ir.terrain,
        rng=terrain_rng,
        num_envs=terrain_num_envs,
        env_spacing=terrain_env_spacing,
        max_init_terrain_level=terrain_max_init_level,
        measured=measured_terrain,
        measured_collision_resolution=measured_collision_res,
        measured_collision_window_w=measured_collision_window,
        measured_exact_window_w=measured_exact_window,
    )
    env_origin_w = np.asarray(terrain_build.env_origins_w[0], dtype=np.float64)
    root_pos_w = np.asarray(ir.robot_init.root_pos_env, dtype=np.float64) + env_origin_w

    # The model timestep is the SUBSTEP dt: the runtime steps the model `substeps`
    # times per Isaac physics step. build_mjcf authors the default contact solref as
    # 2 * its timestep (the stiffest reference MuJoCo permits), so solref is scaled
    # against the substep dt automatically.
    # Contact model: "auto" follows the actuator family — soft-gain DC-motor robots get
    # the PhysX-style progressive-engagement profile (see ActuatorSet.contact_profile).
    resolved_contact_profile = actuator_set.contact_profile() if contact_profile == "auto" else contact_profile
    built = build_mjcf(
        robot,
        physics_dt=ir.timing.physics_dt / substeps,
        joint_overrides=joint_overrides,
        contact_profile=resolved_contact_profile,
        solimp=contact_solimp,
        impratio=contact_impratio,
        solref=contact_solref,
        default_qpos=dict(zip(robot.isaac_joint_order, default_joint_pos_isaac)),
        root_pos_w=tuple(root_pos_w),
        root_quat_w=(
            float(ir.robot_init.root_quat_wxyz[0]),
            float(ir.robot_init.root_quat_wxyz[1]),
            float(ir.robot_init.root_quat_wxyz[2]),
            float(ir.robot_init.root_quat_wxyz[3]),
        ),
        gravity_w=(float(ir.gravity_w[0]), float(ir.gravity_w[1]), float(ir.gravity_w[2])),
    )
    verify_build(robot, built)

    # -- scene: robot material + terrain + lights, robot-geom pair friction -------------
    # The robot's authored material (spawn.physics_material, falling back to the sim's
    # default material like PhysX does) seeds the collision geoms' slide friction;
    # add_to_spec then replaces it with the PhysX pair value against the ground.
    # Stamp BEFORE snapshotting robot.xml: the contact-drop replay builds its scene
    # from robot.xml and must pair the same robot material the task scene pairs.
    robot_material = ir.robot_physics_material or ir.sim_physics_material or {}
    mu_robot = float(robot_material.get("static_friction", 1.0))
    # Self-collision robots get BOTH PhysX pairings precomputed (terrain at priority 2
    # with the robot-ground pair value, robot geoms with the self-pair value) — unless
    # a material DR event exists: it rewrites robot-geom friction per draw and needs
    # robot geoms to win, so those bundles keep the ground-pair stamping.
    material_dr = any(class_name(e.func) == "randomize_rigid_body_material" for e in ir.events)
    robot_self_pair = robot.self_collisions_enabled and not material_dr
    if robot.self_collisions_enabled and material_dr:
        mu_self_physx = pair_friction(mu_robot, mu_robot, terrain_build.friction_combine_mode)
        mu_stamped = terrain_build.robot_pair_friction(mu_robot)
        if abs(mu_self_physx - mu_stamped) > 1e-12:
            print(
                f"[convert] WARNING: self-collisions + material DR: robot-robot contacts resolve at the "
                f"stamped robot-ground pair friction {mu_stamped:g} instead of PhysX's self-pair value "
                f"{mu_self_physx:g} (the DR event needs robot geoms to win the friction combination)"
            )
    for geom in built.spec.geoms:
        if geom.contype != 0 or geom.conaffinity != 0:
            geom.friction[0] = mu_robot
    robot_xml = built.spec.to_xml()
    add_to_spec(built.spec, terrain_build, robot_self_pair=robot_self_pair)
    # Directional (sun-like) light: a point light at the origin leaves everything dark
    # once the robot is meters away (rough-terrain env origins routinely are), and the
    # slight tilt keeps vertical faces (stair risers, torsos) from rendering flat.
    built.spec.worldbody.add_light(
        pos=[0.0, 0.0, 10.0],
        dir=[0.25, 0.15, -0.95],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    model = built.spec.compile()
    scene_xml = built.spec.to_xml()

    # -- joint / body indexing ----------------------------------------------------------
    isaac_joint_order = list(robot.isaac_joint_order)
    mj_joint_order = _mj_joint_order(model)
    if sorted(mj_joint_order) != sorted(isaac_joint_order):
        raise ValueError("compiled model joints do not match the USD articulation joints")
    isaac_index = {name: k for k, name in enumerate(isaac_joint_order)}
    isaac_to_mj = [isaac_index[name] for name in mj_joint_order]
    mj_to_isaac = list(np.argsort(np.asarray(isaac_to_mj)))

    isaac_body_names, weld_parent = robot.isaac_body_names, robot.weld_parent
    surviving = {link.name for link in robot.links}
    body_map: dict[str, str] = {}
    for name in isaac_body_names:
        target = name
        while target not in surviving:
            if target not in weld_parent:
                raise ValueError(f"Isaac body '{name}' resolves to no surviving MuJoCo body")
            target = weld_parent[target]
        body_map[name] = target

    _check_height_scanner_bodies(ir, body_map)

    factor = _soft_joint_pos_limit_factor(raw)

    # Manifest terrain entry: the IR dict, plus provenance + grid metadata when the
    # bundle terrain is measured from the dump (the runtime rebuilds from the record).
    terrain_manifest = None if ir.terrain is None else ir.terrain.to_dict()
    if measured_terrain is not None:
        assert terrain_manifest is not None
        terrain_manifest["source"] = "measured-from-dump"
        terrain_manifest["measured"] = {
            "file": MEASURED_TERRAIN_NAME,
            "grid_shape": [int(n) for n in measured_terrain.height_grid.shape],
            "grid_origin_xy": [float(v) for v in measured_terrain.grid_origin_xy],
            "grid_resolution": float(measured_terrain.grid_resolution),
            "num_mesh_vertices": int(measured_terrain.mesh_vertices_w.shape[0]),
            "num_mesh_faces": int(measured_terrain.mesh_faces.shape[0]),
            "dump_path": str(dump),
            # Collision-grid provenance (scene.xml is the collision authority; the
            # record above is resolution-independent and reload paths ignore these).
            "collision_resolution": float(
                terrain_build.measured_collision_resolution
                if terrain_build.measured_collision_resolution is not None
                else measured_terrain.grid_resolution
            ),
            "collision_fine_window_w": (
                None
                if terrain_build.measured_fine_window_w is None
                else [float(v) for v in terrain_build.measured_fine_window_w]
            ),
            "collision_hfield_nodes": int(sum(h.nrow * h.ncol for h in terrain_build.hfields)),
            "collision_exact_prisms": (
                None if terrain_build.measured_prisms_w is None else int(terrain_build.measured_prisms_w.shape[0])
            ),
        }

    # -- manifest -----------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "schema": bundle.MANIFEST_SCHEMA,
        "robot": {
            "isaac_joint_order": isaac_joint_order,
            "mj_joint_order": mj_joint_order,
            "isaac_to_mj": isaac_to_mj,
            "mj_to_isaac": mj_to_isaac,
            "entity": ir.robot_entity,
            "root_body": robot.root_link.name,
            "isaac_body_names": isaac_body_names,
            "body_map": body_map,
            "geom_map": _geom_map(robot, isaac_body_names),
            "self_collisions": robot.self_collisions_enabled,
            # Which friction scheme the scene was stamped with (see terrain.add_to_spec);
            # the contact-drop replay re-stamps its own scene and must match.
            "friction_self_pair": robot_self_pair,
            "default_qpos": [],  # filled below (needs the RobotMap)
            "default_qvel": [],
            "default_joint_pos_isaac": default_joint_pos_isaac,
            "default_joint_vel_isaac": default_joint_vel_isaac,
            "joint_pos_limits_isaac": _joint_limits(model, isaac_joint_order, factor),
            "joint_vel_limits_isaac": _joint_vel_limits(ir, isaac_joint_order),
            "soft_joint_pos_limit_factor": factor,
            "total_mass": robot.total_mass,
            # Exact principal moments for triangle-inequality-violating links (see
            # usd2mjcf.parser.principal_inertia; restored by env.stamp_exact_inertia).
            "body_inertia_exact": {
                link.name: [float(v) for v in link.inertia_diag_exact]
                for link in robot.links
                if link.inertia_diag_exact is not None
                and not np.allclose(link.inertia_diag_exact, link.inertia_diag, rtol=0.0, atol=0.0)
            },
        },
        "timing": {**ir.timing.to_dict(), "physics_substeps": substeps},
        "obs": {
            "groups": {group.name: group.to_dict() for group in ir.obs_groups},
            "layouts": {},  # filled below
            "obs_dims": {},
            "policy_group": "policy",
        },
        "action": resolve_action_arrays(ir.action.to_dict(), isaac_joint_order, default_joint_pos_isaac),
        "actuators": [group.to_dict() for group in ir.actuators],
        "commands": [{**c.to_dict(), "dim": command_dim(c.type)} for c in ir.commands],
        "events": [e.to_dict() for e in ir.events],
        "terminations": [t.to_dict() for t in ir.terminations],
        "terrain": terrain_manifest,
        "contact_sensors": [s.to_dict() for s in ir.contact_sensors],
        "height_scanners": [s.to_dict() for s in ir.height_scanners],
        "sim": {
            "gravity_w": list(ir.gravity_w),
            "physics_material": ir.sim_physics_material,
            "robot_physics_material": ir.robot_physics_material,
            "contact_profile": resolved_contact_profile,
            "legacy_friction": legacy_friction_record,
            "contact_solimp": None if contact_solimp is None else [float(v) for v in contact_solimp],
            "contact_impratio": contact_impratio,
            "contact_solref": None if contact_solref is None else [float(v) for v in contact_solref],
        },
        "init": {**ir.robot_init.to_dict(), "env_origin_w": env_origin_w, "seed": ir.seed},
        "sources": {
            "env_yaml_sha256": bundle.sha256_file(env_yaml_path),
            "usd_sha256": bundle.sha256_file(usd_path),
            "env_yaml_path": str(env_yaml_path),
            "usd_path": str(usd_path),
            "usd_url": ir.usd_path,
        },
    }

    # Obs layouts (per-term slices of the flat vector) for every group.
    command_dims = {c.name: command_dim(c.type) for c in ir.commands}
    resample_s = {
        c.name: float(c.params["resampling_time_range"][1]) for c in ir.commands if "resampling_time_range" in c.params
    }
    policy_obs_dim = _resolve_obs_layouts(ir, manifest, isaac_joint_order, command_dims, resample_s)

    # PreTrainedPolicyAction: the low-level obs group must be fully servable by the
    # runtime (it feeds the bundled low-level policy); building the pipeline here
    # fails conversion on unsupported terms and records the group's obs dim.
    if "low_level" in manifest["action"]:
        _, ll_pipeline = build_low_level_obs_pipeline(
            manifest["action"],
            num_joints=len(isaac_joint_order),
            command_dims=command_dims,
            command_resample_time_s=resample_s,
            height_scanners=ir.height_scanners,
            entity=ir.robot_entity,
        )
        manifest["action"]["low_level_obs_dim"] = ll_pipeline.obs_dim

    # Default qpos/qvel through the shared root-state conversion (Isaac init velocities
    # are CoM/world quantities; the free joint stores link-origin/body-local ones).
    robot_map = build_robot_map(model, manifest)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qvel[:] = 0.0
    data.qvel[robot_map.dof_adr] = default_joint_vel_isaac
    write_root_state(
        model,
        data,
        robot_map,
        pos_w=root_pos_w,
        quat_wxyz=np.asarray(ir.robot_init.root_quat_wxyz, dtype=np.float64),
        vel_com_w=robot_map.default_root_vel_w,
    )
    manifest["robot"]["default_qpos"] = data.qpos.copy()
    manifest["robot"]["default_qvel"] = data.qvel.copy()

    # -- write bundle -------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / bundle.ROBOT_XML_NAME).write_text(robot_xml)
    (out_dir / bundle.SCENE_XML_NAME).write_text(scene_xml)
    for rel_path, blob in built.assets.items():
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    for rel_path, local in {**net_files, **policy_files}.items():
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, target)
    if measured_terrain is not None:
        target = out_dir / MEASURED_TERRAIN_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        measured_terrain.save_npz(target)
    manifest = bundle.write_manifest(out_dir, manifest)

    _print_report(manifest, model, policy_obs_dim)
    if dump is not None:
        _cross_check_dump(Path(dump), manifest, model)
    return out_dir


def _print_report(manifest: dict[str, Any], model: mujoco.MjModel, policy_obs_dim: int) -> None:
    robot = manifest["robot"]
    rows = [
        ("joints", str(len(robot["isaac_joint_order"]))),
        ("isaac bodies", str(len(robot["isaac_body_names"]))),
        ("mj bodies", str(model.nbody - 1)),
        ("total mass [kg]", f"{float(np.sum(model.body_mass)):.3f}"),
        ("obs dim (policy)", str(policy_obs_dim)),
        ("action dim", str(manifest["action"]["dim"])),
        ("physics dt [s]", str(manifest["timing"]["physics_dt"])),
        ("decimation", str(manifest["timing"]["decimation"])),
        ("physics substeps", str(manifest["timing"]["physics_substeps"])),
        ("mj timestep [s]", str(model.opt.timestep)),
    ]
    terrain = manifest["terrain"]
    if terrain is not None:
        source = "measured-from-dump" if terrain.get("source") == "measured-from-dump" else "procedural"
        rows.append(("terrain", f"{terrain['terrain_type']} ({source})"))
        measured = terrain.get("measured")
        if measured is not None:
            window = measured.get("collision_fine_window_w")
            prisms = measured.get("collision_exact_prisms")
            if prisms is not None and window is not None:
                detail = (
                    f"exact mesh ({prisms} prisms) in [{window[0]:.1f}, {window[1]:.1f}] x "
                    f"[{window[2]:.1f}, {window[3]:.1f}] m, {measured['grid_resolution']:.4g} m hfield elsewhere"
                )
            else:
                detail = f"{measured['collision_resolution']:.4g} m"
                if window is not None:
                    detail += (
                        f" in [{window[0]:.1f}, {window[1]:.1f}] x [{window[2]:.1f}, {window[3]:.1f}] m, "
                        f"{measured['grid_resolution']:.4g} m elsewhere"
                    )
            detail += f" ({measured['collision_hfield_nodes']:,} hfield nodes)"
            rows.append(("collision grid", detail))
    if "low_level" in manifest["action"]:
        rows.append(("low-level action dim", str(manifest["action"]["low_level"]["dim"])))
        rows.append(("low-level obs dim", str(manifest["action"]["low_level_obs_dim"])))
        rows.append(("low-level decimation", str(manifest["action"]["low_level_decimation"])))
    width = max(len(label) for label, _ in rows)
    print("[convert] conversion report")
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")


def _cross_check_dump(dump_path: Path, manifest: dict[str, Any], model: mujoco.MjModel) -> None:
    dump = np.load(dump_path)
    warnings_found = False

    dump_joints = [str(n) for n in dump["joint_names"]]
    if dump_joints != manifest["robot"]["isaac_joint_order"]:
        warnings_found = True
        print(
            "[convert] WARNING: dump joint order differs from the converted Isaac joint order:\n"
            f"  dump:      {dump_joints}\n  converted: {manifest['robot']['isaac_joint_order']}"
        )

    if "env_origin_w" in dump:
        bundle_origin = np.asarray(manifest["init"]["env_origin_w"], dtype=np.float64)
        dump_origin = np.asarray(dump["env_origin_w"], dtype=np.float64)
        if not np.allclose(bundle_origin, dump_origin, atol=1e-6):
            warnings_found = True
            print(
                f"[convert] WARNING: env-0 origin differs: bundle {bundle_origin.tolist()} vs dump "
                f"{dump_origin.tolist()} (dump recorded with a different num_envs than the training run?); "
                "strict validation initializes from the dump's world state and is unaffected"
            )
    if "gravity_w" in dump:
        bundle_gravity = np.asarray(manifest["sim"]["gravity_w"], dtype=np.float64)
        dump_gravity = np.asarray(dump["gravity_w"], dtype=np.float64)
        if not np.allclose(bundle_gravity, dump_gravity, atol=1e-9):
            warnings_found = True
            print(
                f"[convert] WARNING: gravity differs: bundle {bundle_gravity.tolist()} vs dump {dump_gravity.tolist()}"
            )

    if "body_names" in dump and "body_masses" in dump:
        body_map = manifest["robot"]["body_map"]
        mass_of_mj: dict[str, float] = {}
        for name, mass in zip(dump["body_names"], dump["body_masses"]):
            mj_name = body_map.get(str(name))
            if mj_name is None:
                warnings_found = True
                print(f"[convert] WARNING: dump body '{name}' is unknown to the bundle body_map")
                continue
            mass_of_mj[mj_name] = mass_of_mj.get(mj_name, 0.0) + float(mass)
        for mj_name, dump_mass in sorted(mass_of_mj.items()):
            model_mass = float(model.body(mj_name).mass[0])
            if abs(model_mass - dump_mass) > 1e-3 * max(1.0, dump_mass):
                warnings_found = True
                print(
                    f"[convert] WARNING: body '{mj_name}' mass differs: model {model_mass:.4f} kg "
                    f"vs dump {dump_mass:.4f} kg (startup mass DR in the dump run?)"
                )
    if not warnings_found:
        print(f"[convert] dump cross-check OK ({dump_path.name})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="IsaacLab run directory (or a direct path to env.yaml).")
    parser.add_argument("--usd", default=None, help="Robot USD path (default: resolve from env.yaml usd_path).")
    parser.add_argument("--out", default=None, help="Bundle output directory (default: <run>/mj_bundle).")
    parser.add_argument("--dump", default=None, help="Optional Isaac reference dump npz for cross-checks.")
    parser.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="MuJoCo steps per Isaac physics step; the model timestep and contact solref are "
        "authored against physics_dt / substeps. Default: the articulation's PhysX TGS "
        f"solver_position_iteration_count from env.yaml, else {DEFAULT_SUBSTEPS}.",
    )
    parser.add_argument(
        "--contact_profile",
        default="auto",
        choices=("auto", *CONTACT_PROFILES),
        help="Ground-contact model: 'auto' (default) follows the actuator family "
        "(ActuatorSet.contact_profile); force 'default' or 'engagement' for robots the "
        "heuristic misjudges. Recorded in the manifest as sim.contact_profile.",
    )
    parser.add_argument(
        "--contact_solimp",
        default=None,
        help="Explicit contact solimp 'd0,d1,width,mid,power' authored on top of the profile "
        "(per-robot calibration output; see scripts/calibrate_contact.py).",
    )
    parser.add_argument(
        "--contact_impratio",
        type=float,
        default=None,
        help="Explicit option/impratio authored on top of the profile (calibration output).",
    )
    parser.add_argument(
        "--contact_solref",
        default=None,
        help="Explicit contact solref 'timeconst,dampratio' replacing the default 2*substep_dt (calibration output).",
    )
    parser.add_argument(
        "--physx_plant",
        default=None,
        help="npz with PhysX-resolved body inertias/CoMs (any reference or freespace dump of this robot); "
        "overrides USD-authored mass properties. Defaults to --dump's record when present.",
    )
    parser.add_argument(
        "--terrain_exact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collide the exact recorded terrain mesh (convex prisms) inside the dump trajectory's "
        "window instead of a fine hfield; falls back to hfield collision automatically when the "
        "window's triangle count exceeds the prism cap. Ignored without a measured-terrain --dump.",
    )
    parser.add_argument(
        "--terrain_collision_res",
        default="auto",
        help="Measured-terrain collision node spacing in meters, or 'auto' (finest of record/4, record/2, "
        "record within the node budget; default) or 'record' (the dump's grid as-is). Finer grids are "
        "re-raycast from the recorded terrain mesh. Ignored without a measured-terrain --dump.",
    )
    parser.add_argument(
        "--terrain_fine_margin",
        type=float,
        default=MEASURED_FINE_MARGIN_M,
        help="Margin in meters around the dump trajectory's xy extent for the fine measured-collision "
        f"window; the recorded resolution covers the rest of the terrain (default {MEASURED_FINE_MARGIN_M}).",
    )
    parser.add_argument(
        "--terrain_fine_full",
        action="store_true",
        help="Author the fine measured-collision grid over the full terrain extent instead of a window "
        "around the dump trajectory (subject to the node budget).",
    )
    parser.add_argument(
        "--legacy_friction",
        default=None,
        help="Fixed joint friction for an env using PhysX's legacy friction model (contact_lab's "
        "set_legacy_joint_friction): 'regex=static,dynamic,viscous;...' in N*m, N*m, N*m*s/rad, keyed "
        "like the event's coefficients.",
    )
    args = parser.parse_args(argv)
    collision_res: str | float = args.terrain_collision_res
    if collision_res not in ("auto", "record"):
        collision_res = float(collision_res)
    solref_arg: tuple[float, float] | None = None
    if args.contact_solref is not None:
        timeconst, dampratio = (float(v) for v in args.contact_solref.split(","))
        solref_arg = (timeconst, dampratio)
    convert_run(
        args.run,
        usd=args.usd,
        out=args.out,
        dump=args.dump,
        substeps=args.substeps,
        physx_plant=args.physx_plant,
        terrain_exact=args.terrain_exact,
        contact_profile=args.contact_profile,
        contact_solimp=(
            None if args.contact_solimp is None else tuple(float(v) for v in args.contact_solimp.split(","))
        ),
        contact_impratio=args.contact_impratio,
        contact_solref=solref_arg,
        terrain_collision_res=collision_res,
        terrain_fine_margin=args.terrain_fine_margin,
        terrain_fine_full=args.terrain_fine_full,
        legacy_friction=None if args.legacy_friction is None else parse_legacy_friction(args.legacy_friction),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
