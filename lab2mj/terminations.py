"""Episode terminations for the MuJoCo runtime, mirroring isaaclab.envs.mdp.terminations.

Single-env numpy ports of the termination terms used by the velocity tasks, plus
contact_lab's ``terrain_out_of_bounds`` and ``vertical_contact``. Terms flagged
``time_out`` in the config are truncations (Isaac's TerminationManager splits the two).

Contact-force terms read MuJoCo contacts directly (``mj_contactForce`` summed per tracked
body, world frame), the raw-MuJoCo equivalent of the Isaac contact sensor's net force.
With ``geom_map`` (Isaac body -> collision geom ids; the runtime passes it) each *Isaac*
body sums its own collision geoms, so a welded child folded into its MuJoCo parent (Spot's
``*_foot`` into ``*_lleg``) keeps its contacts separate, as Isaac's sensor does. Without
it, forces attribute per MuJoCo body and folded Isaac bodies share one row (see
:func:`lab2mj.events.resolve_body_ids`).

``illegal_contact`` in Isaac checks the max over the contact sensor's *history* window
(one entry per physics step). Call :meth:`TerminationSet.push_contact_forces` once per
physics step to reproduce that; without pushes, checks fall back to the current contact
state only.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from lab2mj.commands import RobotState
from lab2mj.env_yaml import class_name
from lab2mj.events import check_scene_entity_cfg, resolve_body_ids, resolve_names
from lab2mj.ir import TerminationIR
from lab2mj.quat import quat_apply_inverse
from lab2mj.terrain import map_size_from_generator

__all__ = [
    "TerminationSet",
    "contact_forces_by_geom_w",
    "map_size_from_generator",
    "net_contact_forces_w",
    "resolve_body_ids",
]

KNOWN_TERMINATION_FUNCS = (
    "time_out",
    "illegal_contact",
    "bad_orientation",
    "root_height_below_minimum",
    "terrain_out_of_bounds",
    "vertical_contact",
)

_CONTACT_FUNCS = ("illegal_contact", "vertical_contact")


def contact_forces_by_geom_w(
    model: mujoco.MjModel, data: mujoco.MjData, geom_index: np.ndarray, n_rows: int
) -> np.ndarray:
    """Net world-frame contact force per row, ``(n_rows, 3)``.

    ``geom_index`` is the ``(ngeom,)`` lookup from geom id to output row (-1 = untracked).
    Sums ``mj_contactForce`` over ``data.contact``; the contact-frame normal points from
    geom1 to geom2, so the rotated force acts on geom2's row (+) and geom1's row (-).
    """
    out = np.zeros((n_rows, 3), dtype=np.float64)
    if data.ncon == 0:
        return out
    idx = geom_index[data.contact.geom]  # (ncon, 2) tracked rows or -1
    frames = data.contact.frame
    wrench = np.zeros(6, dtype=np.float64)
    # Per-contact loop on purpose: mj_contactForce has no batched form, and a
    # vectorized rotate/scatter (matmul + np.add.at) measured ~60% SLOWER at the
    # few-contacts counts these robots produce (ncon <= ~10, feet only).
    for i in np.nonzero((idx[:, 0] >= 0) | (idx[:, 1] >= 0))[0]:
        mujoco.mj_contactForce(model, data, int(i), wrench)
        force_w = frames[i].reshape(3, 3).T @ wrench[:3]
        if idx[i, 1] >= 0:
            out[idx[i, 1]] += force_w
        if idx[i, 0] >= 0:
            out[idx[i, 0]] -= force_w
    return out


def net_contact_forces_w(
    model: mujoco.MjModel, data: mujoco.MjData, body_ids: list[int], body_index: np.ndarray | None = None
) -> np.ndarray:
    """Net world-frame contact force per MuJoCo body, ``(len(body_ids), 3)``.

    ``body_index`` is the optional precomputed ``(nbody,)`` lookup (-1 = untracked,
    else row into the output).
    """
    if body_index is None:
        body_index = _body_row_index(model, body_ids)
    return contact_forces_by_geom_w(model, data, body_index[model.geom_bodyid], len(body_ids))


def _body_row_index(model: mujoco.MjModel, body_ids: list[int]) -> np.ndarray:
    # Sized to cover ids beyond nbody (tracked ids may come from a larger model).
    size = max(model.nbody, max(body_ids, default=0) + 1)
    body_index = np.full(size, -1, dtype=np.intp)
    body_index[np.asarray(body_ids, dtype=np.intp)] = np.arange(len(body_ids), dtype=np.intp)
    return body_index


def _sensor_body_map(sensor: dict[str, Any], body_map: dict[str, list[int]]) -> dict[str, list[int]]:
    """Tracked-body subset of ``body_map`` for a contact sensor.

    Isaac resolves ``sensor_cfg.body_names`` against ``ContactSensor.body_names``
    — the bodies matched by the sensor's prim-path leaf pattern — not against the
    full articulation. The prim path's last segment is the body-name pattern
    (e.g. ``/World/envs/env_.*/Robot/.*_FOOT`` tracks only the feet).
    """
    leaf = (sensor.get("prim_path") or "").rsplit("/", 1)[-1] or ".*"
    return {name: body_map[name] for name in resolve_names(leaf, list(body_map))}


def _find_sensor(name: Any, contact_sensors: list[dict[str, Any]], term_name: str) -> dict[str, Any]:
    for sensor in contact_sensors:
        if sensor["name"] == name:
            return sensor
    known = [s["name"] for s in contact_sensors]
    raise ValueError(f"termination '{term_name}': sensor_cfg names contact sensor '{name}', manifest has {known}")


@dataclass
class _Term:
    name: str
    func: str
    time_out: bool
    params: dict[str, Any]
    rows: list[Any] = field(default_factory=list)  # tracked keys: Isaac body names (geom_map) or mj body ids
    history_slice: np.ndarray | None = None  # indices into the tracked-body axis
    window: int = 1  # physics-step contact-history window (the term's sensor's history_length)


class TerminationSet:
    """Evaluates a list of :class:`TerminationIR` terms against the MuJoCo state.

    Args:
        terms: Parsed termination terms; unknown funcs raise ``ValueError``.
        step_dt: Policy step dt (Isaac ``env.step_dt``).
        episode_length_s: Episode length; ``time_out`` fires at
            ``ceil(episode_length_s / step_dt)`` policy steps.
        body_map: Multimap ``isaac body name -> mj body id(s)`` in Isaac body order.
        terrain_size_xy: Full terrain map (width, height) for ``terrain_out_of_bounds``;
            ``None`` means an infinite plane (never out of bounds).
        contact_history_length: Physics-step history window for ``illegal_contact``
            (the Isaac contact sensor's ``history_length``; 1 = current step only).
            Fallback for standalone use — ignored when ``contact_sensors`` is given.
        contact_sensors: The manifest's contact sensor entries (name, prim_path,
            history_length). When given, each contact term resolves its
            ``sensor_cfg.body_names`` against its named sensor's tracked-body
            subset and maxes over that sensor's own history window, exactly like
            Isaac's per-sensor ``ContactSensor.body_names`` / history buffers.
        gravity_dir_w: Unit gravity direction in world frame for ``bad_orientation``
            (Isaac derives its projected gravity from the sim's normalized gravity).
        geom_map: Multimap ``isaac body name -> mj collision geom id(s)``. When given,
            contact terms track Isaac bodies through their own geoms (exact even where
            welded bodies share an mj body); otherwise they track mj bodies.
    """

    def __init__(
        self,
        terms: list[TerminationIR],
        *,
        step_dt: float,
        episode_length_s: float,
        body_map: dict[str, list[int]],
        terrain_size_xy: tuple[float, float] | None = None,
        contact_history_length: int = 1,
        contact_sensors: list[dict[str, Any]] | None = None,
        gravity_dir_w: Any = (0.0, 0.0, -1.0),
        entity: str = "robot",
        geom_map: dict[str, list[int]] | None = None,
    ) -> None:
        self._step_dt = float(step_dt)
        self._gravity_dir_w = np.asarray(gravity_dir_w, dtype=np.float64)
        self._max_episode_length = math.ceil(episode_length_s / step_dt)
        self._terrain_size_xy = terrain_size_xy
        self._terms: list[_Term] = []
        for ir in terms:
            func = class_name(ir.func)
            if func not in KNOWN_TERMINATION_FUNCS:
                raise ValueError(f"unsupported termination func '{ir.func}' (known: {list(KNOWN_TERMINATION_FUNCS)})")
            check_scene_entity_cfg(
                ir.params.get("asset_cfg"), f"termination '{ir.name}'", entity=entity, check_selectors=False
            )
            term = _Term(name=ir.name, func=func, time_out=ir.time_out, params=dict(ir.params))
            if func in _CONTACT_FUNCS:
                sensor_cfg = ir.params.get("sensor_cfg") or {}
                check_scene_entity_cfg(sensor_cfg, f"termination '{ir.name}' sensor_cfg", entity=None)
                if contact_sensors is None:
                    tracked_map = body_map
                    term.window = max(1, int(contact_history_length))
                else:
                    sensor = _find_sensor(sensor_cfg.get("name"), contact_sensors, ir.name)
                    tracked_map = _sensor_body_map(sensor, body_map)
                    term.window = max(1, int(sensor.get("history_length") or 0))
                if geom_map is None:
                    term.rows = resolve_body_ids(sensor_cfg.get("body_names"), tracked_map)
                else:
                    term.rows = resolve_names(sensor_cfg.get("body_names"), list(tracked_map))
            self._terms.append(term)

        # Union of bodies any contact term watches, for the shared per-physics-step
        # history; each term maxes over the last `window` entries only.
        self._geom_map = geom_map
        self._tracked: list[Any] = []
        for term in self._terms:
            for key in term.rows:
                if key not in self._tracked:
                    self._tracked.append(key)
        for term in self._terms:
            if term.func in _CONTACT_FUNCS:
                term.history_slice = np.array([self._tracked.index(key) for key in term.rows], dtype=np.int64)
        history = max([t.window for t in self._terms if t.func in _CONTACT_FUNCS] + [1])
        self._history: deque[np.ndarray] = deque(maxlen=history)
        self._geom_index: np.ndarray | None = None  # (ngeom,) tracked-row lookup, built on first use

    @property
    def tracked(self) -> list[Any]:
        """Rows of the contact history: Isaac body names with ``geom_map``, else mj body ids."""
        return list(self._tracked)

    def reset(self) -> None:
        """Clear the contact-force history (call on episode reset)."""
        self._history.clear()

    def push_contact_forces(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Record current net contact forces for the tracked bodies (call per physics step)."""
        if not self._tracked:
            return
        self._history.append(self._contact_forces(model, data))

    def _contact_forces(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        if self._geom_index is None:
            if self._geom_map is None:
                self._geom_index = _body_row_index(model, self._tracked)[model.geom_bodyid]
            else:
                self._geom_index = np.full(model.ngeom, -1, dtype=np.intp)
                for row, name in enumerate(self._tracked):
                    self._geom_index[np.asarray(self._geom_map.get(name, []), dtype=np.intp)] = row
        return contact_forces_by_geom_w(model, data, self._geom_index, len(self._tracked))

    def check(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: RobotState,
        episode_t: float,
    ) -> tuple[bool, bool, list[str]]:
        """Evaluate all terms; returns ``(terminated, time_out, reasons)``.

        ``episode_t`` is the episode time in seconds at the end of the current policy
        step (``num_steps * step_dt``). ``terminated`` ORs the non-time-out terms,
        ``time_out`` ORs the ones flagged ``time_out`` (Isaac's terminated/truncated
        split); ``reasons`` lists every triggered term name.
        """
        episode_steps = int(round(episode_t / self._step_dt))
        forces_history = self._contact_history(model, data)
        terminated = False
        timed_out = False
        reasons: list[str] = []
        for term in self._terms:
            triggered = self._evaluate(term, data, state, episode_steps, forces_history)
            if triggered:
                reasons.append(term.name)
                if term.time_out:
                    timed_out = True
                else:
                    terminated = True
        return terminated, timed_out, reasons

    # -- internals ------------------------------------------------------------------

    def _contact_history(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray | None:
        if not self._tracked:
            return None
        if self._history:
            return np.stack(self._history)  # (T, n_tracked, 3)
        return self._contact_forces(model, data)[None]

    def _evaluate(
        self,
        term: _Term,
        data: mujoco.MjData,
        state: RobotState,
        episode_steps: int,
        forces_history: np.ndarray | None,
    ) -> bool:
        params = term.params
        if term.func == "time_out":
            return episode_steps >= self._max_episode_length
        if term.func == "illegal_contact":
            assert forces_history is not None and term.history_slice is not None
            if len(term.history_slice) == 0:
                return False
            # Max over the term's OWN sensor window: Isaac's per-sensor history
            # buffers expire a transient after that sensor's history_length steps
            # even when another sensor keeps a longer shared history.
            norms = np.linalg.norm(forces_history[-term.window :, term.history_slice], axis=-1)
            return bool(norms.max() > float(params["threshold"]))
        if term.func == "vertical_contact":
            assert forces_history is not None and term.history_slice is not None
            if len(term.history_slice) == 0:
                return False
            # Current-step net forces only (no history in the Isaac term).
            z_forces = np.abs(forces_history[-1, term.history_slice, 2])
            return bool(z_forces.max() > float(params.get("threshold", 1.0)))
        if term.func == "bad_orientation":
            projected_gravity_b = quat_apply_inverse(state.root_quat_w_wxyz, self._gravity_dir_w)
            angle = math.acos(float(np.clip(-projected_gravity_b[2], -1.0, 1.0)))
            return abs(angle) > float(params["limit_angle"])
        if term.func == "root_height_below_minimum":
            return float(state.root_pos_w[2]) < float(params["minimum_height"])
        if term.func == "terrain_out_of_bounds":
            if self._terrain_size_xy is None:
                return False  # plane terrain is infinite
            distance_buffer = float(params.get("distance_buffer", 3.0))
            map_width, map_height = self._terrain_size_xy
            x_out = abs(float(state.root_pos_w[0])) > 0.5 * map_width - distance_buffer
            y_out = abs(float(state.root_pos_w[1])) > 0.5 * map_height - distance_buffer
            return x_out or y_out
        raise AssertionError(f"unhandled termination func '{term.func}'")
