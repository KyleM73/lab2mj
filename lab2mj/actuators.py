"""Actuator torque computation mirroring IsaacLab's PD actuator models.

Builds vectorized per-joint gain/limit arrays from :class:`ActuatorGroupIR` and
computes torques exactly as IsaacLab v2.3.0 does on the GPU side:

- ``implicit_pd`` / ``ideal_pd``: ``tau = kp*(q_des - q) + kd*(qd_des - qd)``,
  clipped to ``+-effort_limit`` per joint (PhysX clamps the implicit drive's
  total force the same way).
- ``delayed_pd``: same PD law, but the position/velocity targets are lagged by
  a per-reset integer number of physics steps drawn from
  ``randint(min_delay, max_delay + 1)``. The delay buffer replicates IsaacLab's
  ``DelayBuffer``: the first push after a reset fills the whole buffer with the
  current value, and a lag larger than the number of pushes returns the oldest
  stored value.
- ``remotized_pd``: delayed PD whose effort box limit is forced to infinity;
  instead the torque is clamped to an angle-dependent envelope linearly
  interpolated from ``joint_parameter_lookup`` columns (angle, ratio,
  max_torque), with zero-order hold outside the table range (``np.interp``
  matches IsaacLab's ``LinearInterpolation`` exactly).
- ``dc_motor``: same PD law, clipped to the linear four-quadrant DC-motor
  torque-speed curve instead of the plain box (:func:`dc_motor_clip`, an exact
  mirror of ``DCMotor._clip_effort``).
- ``actuator_net_lstm``: torque comes from a TorchScript LSTM
  (``ActuatorNetLSTM``) evaluated once per physics step on the per-joint input
  ``[q_des - q, qd]`` (unscaled) with persistent hidden/cell state, then
  DCMotor-clipped. IsaacLab's ``ActuatorNetLSTM.compute`` never writes the
  ``DCMotor._joint_vel`` buffer, so its clip evaluates the torque-speed curve
  at zero velocity — i.e. a constant ``+-min(saturation_effort, effort_limit)``
  box; this mirror reproduces that exactly (isaaclab 0.47.2).
- ``actuator_net_mlp``: torque comes from a TorchScript MLP (``ActuatorNetMLP``)
  fed with pos-error/velocity histories: per joint, the history rows selected
  by ``input_idx`` (0 = current physics step, n = n steps ago) scaled by
  ``pos_scale``/``vel_scale``, concatenated in ``input_order``; the output is
  scaled by ``torque_scale``. Unlike the LSTM, ``ActuatorNetMLP.compute``
  *does* write ``DCMotor._joint_vel``, so its clip evaluates the torque-speed
  curve at the current joint velocity. ``reset`` zeroes the histories (Isaac
  allocates them zero-filled and never seeds them with the current state).

Usage per physics step::

    q_des, qd_des = actuator_set.step_delay_buffers(q_target_isaac)  # stateful
    tau = actuator_set.compute_torques(q_isaac, qd_isaac, q_des, qd_des)  # pure*

``step_delay_buffers`` must be called exactly once per physics step (it pushes
into the delay buffers); ``compute_torques`` is stateless (*) except when an
actuator-net group exists — the LSTM's hidden state and the MLP's history
buffers advance on every call, so it too must run exactly once per physics
step (Isaac evaluates the network once per physics step and applies the
torque explicitly). When no group has a delay model, ``step_delay_buffers``
is a pass-through and the raw targets may be fed to ``compute_torques``
directly. When initializing from a settled reference dump,
:meth:`ActuatorSet.warm_start` (after :meth:`ActuatorSet.reset`) seeds the MLP
actuator-net histories that Isaac's settle phase left warm (LSTM state stays
cold — see the method doc); plain resets keep the zeroed state, matching
Isaac's ``reset``.

This module is numpy-only, with one documented exception to the no-torch rule:
actuator-net groups need ``torch.jit`` to evaluate the network, so ``torch``
is imported lazily and only when such a group actually loads its network.
Bundles without actuator-net groups never touch torch.

Armature and joint friction are *model* parameters in MuJoCo (``dof_armature``,
``dof_frictionloss``, ``dof_damping``), not runtime torque terms —
:meth:`ActuatorSet.builder_overrides` exports them for the MJCF builder and they
never enter ``compute_torques``. Friction follows the PhysX >= 5 joint-friction
model (PxJointAxis): the cfg's ``friction`` is a STATIC friction effort that only
acts on stationary joints (breakaway threshold — measured in free fall: a moving
PhysX joint with static friction authored and zero dynamic/viscous never decays),
``dynamic_friction`` is the Coulomb effort resisting motion (-> MuJoCo
``dof_frictionloss``), and ``viscous_friction`` the velocity-proportional
coefficient (-> plant ``dof_damping``). The static effort has no MuJoCo
counterpart and is dropped with a warning when it exceeds the dynamic effort.

All joint-space vectors are in Isaac articulation joint order (``*_isaac``).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lab2mj.ir import ActuatorGroupIR, ActuatorModel

_DELAYED_MODELS: tuple[ActuatorModel, ...] = ("delayed_pd", "remotized_pd")
# Models whose applied torque is clamped by the DCMotor torque-speed curve.
_DC_MOTOR_MODELS: tuple[ActuatorModel, ...] = ("dc_motor", "actuator_net_lstm", "actuator_net_mlp")
# Models whose torque comes from a TorchScript actuator net (weights bundled by convert).
NET_MODELS: tuple[ActuatorModel, ...] = ("actuator_net_lstm", "actuator_net_mlp")
# Actuator families whose robots get the builder's "engagement" contact profile: the
# soft-gain DC-motor quadrupeds (explicit dc_motor groups, and the MLP actuator nets
# that share the DCMotor torque-speed clip and hardware class). See
# :meth:`ActuatorSet.contact_profile`.
_ENGAGEMENT_CONTACT_MODELS: tuple[ActuatorModel, ...] = ("dc_motor", "actuator_net_mlp")


def dc_motor_clip(
    effort: np.ndarray,
    joint_vel: np.ndarray,
    saturation_effort: np.ndarray,
    effort_limit: np.ndarray,
    velocity_limit: np.ndarray,
) -> np.ndarray:
    """Exact mirror of IsaacLab ``DCMotor._clip_effort`` (per-joint arrays).

    The joint velocity is first clamped to the corner velocity
    ``velocity_limit * (1 + effort_limit / saturation_effort)`` (where the
    torque-speed curve intersects ``-effort_limit``), then the instantaneous
    torque bounds are::

        max_effort = min(saturation_effort * (1 - vel / velocity_limit), effort_limit)
        min_effort = max(saturation_effort * (-1 - vel / velocity_limit), -effort_limit)
    """
    vel_at_effort_lim = velocity_limit * (1.0 + effort_limit / saturation_effort)
    vel = np.clip(joint_vel, -vel_at_effort_lim, vel_at_effort_lim)
    torque_speed_top = saturation_effort * (1.0 - vel / velocity_limit)
    torque_speed_bottom = saturation_effort * (-1.0 - vel / velocity_limit)
    max_effort = np.minimum(torque_speed_top, effort_limit)
    min_effort = np.maximum(torque_speed_bottom, -effort_limit)
    return np.clip(effort, min_effort, max_effort)


def resolve_matching_names(patterns: list[str], names: list[str], what: str = "patterns") -> list[int]:
    """Match regex patterns against ``names``; mirror of isaaclab ``resolve_matching_names``.

    Returns indices into ``names`` ordered by the name list (not by pattern).
    Raises ``ValueError`` if a name is matched by more than one pattern, or if
    any pattern matches no name at all.
    """
    matched_by: list[str | None] = [None] * len(names)
    pattern_hits: list[list[str]] = [[] for _ in patterns]
    indices: list[int] = []
    for name_idx, name in enumerate(names):
        for pattern_idx, pattern in enumerate(patterns):
            if re.fullmatch(pattern, name):
                previous = matched_by[name_idx]
                if previous is not None:
                    raise ValueError(f"{what}: '{name}' matches multiple patterns: '{previous}' and '{pattern}'")
                matched_by[name_idx] = pattern
                indices.append(name_idx)
                pattern_hits[pattern_idx].append(name)
    unmatched = [pattern for pattern, hits in zip(patterns, pattern_hits) if not hits]
    if unmatched:
        raise ValueError(f"{what}: patterns {unmatched} match none of {names}")
    return indices


def resolve_matching_names_values(
    data: dict[str, float], names: list[str], what: str = "values"
) -> tuple[list[int], list[float]]:
    """Regex-dict resolution; mirror of isaaclab ``resolve_matching_names_values`` (strict).

    Returns (indices into ``names``, values), ordered by the name list. Raises
    ``ValueError`` if a name matches more than one key or a key matches no name.
    """
    indices = resolve_matching_names(list(data.keys()), names, what=what)
    values: list[float] = []
    for idx in indices:
        for pattern, value in data.items():
            if re.fullmatch(pattern, names[idx]):
                values.append(float(value))
                break
    return indices, values


def _resolve_group_param(
    value: float | dict[str, float] | None,
    joint_names: list[str],
    default: float,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a scalar-or-regex-dict actuator parameter onto a group's joints.

    Returns ``(values, provided)`` where ``provided`` marks joints the config
    explicitly set. Dict values resolve like isaaclab ``_parse_joint_parameter``:
    the whole group buffer is zero-filled first, so joints not matched by any
    key end up at 0.0 (a real footgun for e.g. effort limits, hence the
    ``UserWarning`` naming them).
    """
    out = np.full(len(joint_names), default, dtype=np.float64)
    provided = np.zeros(len(joint_names), dtype=bool)
    if value is None:
        return out, provided
    if isinstance(value, dict):
        indices, values = resolve_matching_names_values(value, joint_names, what=what)
        out[:] = 0.0
        out[indices] = values
        provided[:] = True
        matched = set(indices)
        unmatched = [name for idx, name in enumerate(joint_names) if idx not in matched]
        if unmatched:
            warnings.warn(
                f"{what}: joints {unmatched} are not matched by any regex key; their value is "
                "zero-filled (IsaacLab _parse_joint_parameter semantics)",
                UserWarning,
                stacklevel=2,
            )
    elif isinstance(value, (int, float)):
        out[:] = float(value)
        provided[:] = True
    else:
        raise TypeError(f"{what}: expected float or dict, got {type(value).__name__}")
    return out, provided


@dataclass
class ActuatorGroup:
    """Runtime state for one actuator group (joint indices, delay buffers, LUT, actuator net)."""

    name: str
    model: ActuatorModel
    joint_ids: np.ndarray
    joint_names: list[str]
    min_delay: int = 0
    max_delay: int = 0
    lag: int = 0
    lut_angle: np.ndarray | None = None
    lut_max_torque: np.ndarray | None = None
    # DCMotor torque-speed curve parameters (per joint), for models in _DC_MOTOR_MODELS.
    saturation_effort: np.ndarray | None = None
    dc_effort_limit: np.ndarray | None = None
    dc_velocity_limit: np.ndarray | None = None
    # Actuator-net weights: resolved local path (None when unresolvable) + source string
    # for error messages. The TorchScript module and LSTM state are loaded lazily.
    network_path: Path | None = None
    network_source: str | None = None
    # MLP actuator-net input/output contract (actuator_net_mlp).
    pos_scale: float = 1.0
    vel_scale: float = 1.0
    torque_scale: float = 1.0
    input_order: str = "pos_vel"
    input_idx: np.ndarray | None = None
    _net: Any = field(default=None, repr=False)
    _hidden: Any = field(default=None, repr=False)
    _cell: Any = field(default=None, repr=False)
    _sea_input: Any = field(default=None, repr=False)
    # Constant torque box for the LSTM's zero-velocity DCMotor clip (see the module docstring).
    dc_zero_vel_box: np.ndarray | None = None
    # MLP histories, shape (max(input_idx) + 1, num_joints) float32, as a ring buffer:
    # row ``(_hist_ptr + n) % len`` is the value n physics steps in the past (Isaac's
    # on-device roll-by-one buffers with num_envs = 1, without the per-step copies).
    _pos_error_history: np.ndarray | None = field(default=None, repr=False)
    _vel_history: np.ndarray | None = field(default=None, repr=False)
    _hist_ptr: int = field(default=0, repr=False)
    # Preallocated MLP network input (num_joints, 2 * len(input_idx)) float32 and the
    # torch tensor sharing its memory (built on network load).
    _net_input: np.ndarray | None = field(default=None, repr=False)
    _net_input_t: Any = field(default=None, repr=False)
    _pos_cols: slice = field(default_factory=lambda: slice(0, 0), repr=False)
    _vel_cols: slice = field(default_factory=lambda: slice(0, 0), repr=False)
    _buf_q_target: np.ndarray | None = field(default=None, repr=False)
    _buf_qd_target: np.ndarray | None = field(default=None, repr=False)
    _pointer: int = field(default=-1, repr=False)
    _num_pushes: int = field(default=0, repr=False)

    @property
    def has_delay(self) -> bool:
        return self.model in _DELAYED_MODELS

    @property
    def has_net(self) -> bool:
        return self.model in NET_MODELS

    def load_network(self) -> None:
        """Load the TorchScript actuator net (idempotent); allocate LSTM state.

        LSTM (mirrors ``ActuatorNetLSTM.__init__``): layer count and hidden dim
        come from the jit module's ``lstm`` submodule state dict; the input
        buffer is ``(num_joints, 1, 2)`` and hidden/cell states are
        ``(num_layers, num_joints, hidden_dim)`` (Isaac's per-env view with
        ``num_envs = 1``), all zero-initialized float32. MLP histories are
        numpy and already allocated at build time — only the jit module loads
        here.
        """
        if self._net is not None:
            return
        if self.network_path is None:
            raise ValueError(
                f"actuator group '{self.name}': actuator-net file not found "
                f"(source {self.network_source!r}); convert copies it into the bundle under "
                "assets/actuator_nets/ — was the bundle produced by an older converter?"
            )
        import torch  # documented exception to the no-torch rule (see module docstring)

        self._net = torch.jit.load(str(self.network_path), map_location="cpu").eval()
        if self.model == "actuator_net_mlp":
            assert self.input_idx is not None
            k = len(self.input_idx)
            self._net_input = np.zeros((len(self.joint_names), 2 * k), dtype=np.float32)
            self._net_input_t = torch.from_numpy(self._net_input)
            if self.input_order == "pos_vel":
                self._pos_cols, self._vel_cols = slice(0, k), slice(k, 2 * k)
            elif self.input_order == "vel_pos":
                self._pos_cols, self._vel_cols = slice(k, 2 * k), slice(0, k)
            else:
                raise ValueError(
                    f"actuator group '{self.name}': invalid input order '{self.input_order}'; "
                    "must be 'pos_vel' or 'vel_pos'"
                )
        if self.model != "actuator_net_lstm":
            return
        state_dict = self._net.lstm.state_dict()
        num_layers = len(state_dict) // 4
        hidden_dim = state_dict["weight_hh_l0"].shape[1]
        num_joints = len(self.joint_names)
        self._sea_input = torch.zeros(num_joints, 1, 2)
        self._hidden = torch.zeros(num_layers, num_joints, hidden_dim)
        self._cell = torch.zeros(num_layers, num_joints, hidden_dim)

    def reset_network_state(self) -> None:
        """Zero the actuator-net state (Isaac ``ActuatorNetLSTM.reset`` / ``ActuatorNetMLP.reset``).

        LSTM: hidden/cell states. MLP: pos-error/velocity histories — zeroed,
        not seeded with the current state (the first compute after a reset sees
        zeros in every non-current history row).
        """
        if self._pos_error_history is not None:
            self._pos_error_history[:] = 0.0
        if self._vel_history is not None:
            self._vel_history[:] = 0.0
        self._hist_ptr = 0
        if self._hidden is None:
            return  # not loaded yet; the state is allocated zeroed on load
        import torch

        self._hidden = torch.zeros_like(self._hidden)
        self._cell = torch.zeros_like(self._cell)

    def warm_start_network_state(self, pos_error: np.ndarray, joint_vel: np.ndarray) -> None:
        """Seed an MLP group's history buffers with the settled response to a constant input.

        Isaac reference dumps record a state produced after a settle phase, so
        the recorded t0 pairs with a *warm* actuator-net state; a zeroed state
        (the plain-reset semantics) makes the first replayed torques disagree
        with the reference. ``pos_error``/``joint_vel`` must be the
        t0-consistent input: the position error against the targets that
        produced the recorded state, and the recorded joint velocity. Every
        history row is filled with the input, exactly the buffer contents after
        the input has been constant for the full history length (the next
        :meth:`compute_mlp_torques` rolls a no-op and lands the fresh input in
        row 0). Non-MLP groups are a no-op: the LSTM has no warm start at all —
        iterating the net to its constant-input fixed point was tried and
        measured to track reference dumps worse than a cold state (see
        :meth:`ActuatorSet.warm_start`), so LSTM state stays cold by design.
        """
        if self.model != "actuator_net_mlp":
            return
        assert self._pos_error_history is not None and self._vel_history is not None
        self._pos_error_history[:] = np.asarray(pos_error, dtype=np.float32)
        self._vel_history[:] = np.asarray(joint_vel, dtype=np.float32)

    def compute_net_torques(self, pos_error: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """Raw (unclipped) actuator-net torques for one physics step; advances the LSTM state.

        Mirrors ``ActuatorNetLSTM.compute``'s network evaluation: per-joint
        input ``[q_des - q, qd]`` with no scaling, float32, batch = the group's
        joints. Clipping is applied by the caller.
        """
        self.load_network()
        import torch

        self._sea_input[:, 0, 0] = torch.from_numpy(np.asarray(pos_error, dtype=np.float32))
        self._sea_input[:, 0, 1] = torch.from_numpy(np.asarray(joint_vel, dtype=np.float32))
        with torch.inference_mode():
            torques, (hidden, cell) = self._net(self._sea_input, (self._hidden, self._cell))
            self._hidden, self._cell = hidden, cell
        return torques.reshape(-1).cpu().numpy().astype(np.float64)

    def compute_mlp_torques(self, pos_error: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        """Raw (unclipped) MLP actuator-net torques for one physics step; advances the histories.

        Mirrors ``ActuatorNetMLP.compute``: the current values land in the ring
        buffers' newest slot; ``input_idx`` selects the history rows fed to the
        network (per joint, columns in ``input_idx`` order); pos-error columns
        are scaled by ``pos_scale`` and velocity columns by ``vel_scale``,
        concatenated per ``input_order`` into a ``(num_joints,
        2 * len(input_idx))`` float32 batch; the output is scaled by
        ``torque_scale``. Clipping is applied by the caller.
        """
        self.load_network()
        import torch

        assert self._pos_error_history is not None and self._vel_history is not None
        assert self.input_idx is not None and self._net_input is not None
        hist_len = self._pos_error_history.shape[0]
        self._hist_ptr = (self._hist_ptr - 1) % hist_len
        self._pos_error_history[self._hist_ptr] = pos_error
        self._vel_history[self._hist_ptr] = joint_vel
        rows = (self._hist_ptr + self.input_idx) % hist_len
        np.multiply(self._pos_error_history[rows].T, np.float32(self.pos_scale), out=self._net_input[:, self._pos_cols])
        np.multiply(self._vel_history[rows].T, np.float32(self.vel_scale), out=self._net_input[:, self._vel_cols])
        with torch.inference_mode():
            torques = self._net(self._net_input_t)
        # torque_scale multiplies in float32 (Isaac scales the on-device output tensor).
        return (torques.reshape(-1).cpu().numpy() * np.float32(self.torque_scale)).astype(np.float64)

    def reset(self, lag: int) -> None:
        """Set the per-reset lag and mark the buffers for refill on the next push."""
        if not self.has_delay:
            return
        if lag < 0:
            raise ValueError(f"actuator group '{self.name}': time lag cannot be negative, got {lag}")
        if lag > self.max_delay:
            raise ValueError(f"actuator group '{self.name}': time lag {lag} exceeds max_delay {self.max_delay}")
        self.lag = lag
        self._num_pushes = 0

    def push(self, q_target: np.ndarray, qd_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Push targets for one physics step and return the lagged targets.

        Mirrors ``DelayBuffer.compute``: the first push after a reset fills the
        whole ring buffer, and the effective lag is clamped to the number of
        pushes since the last reset.
        """
        assert self._buf_q_target is not None and self._buf_qd_target is not None
        length = self.max_delay + 1
        self._pointer = (self._pointer + 1) % length
        self._buf_q_target[self._pointer] = q_target
        self._buf_qd_target[self._pointer] = qd_target
        if self._num_pushes == 0:
            self._buf_q_target[:] = q_target
            self._buf_qd_target[:] = qd_target
        self._num_pushes += 1
        valid_lag = min(self.lag, self._num_pushes - 1)
        idx = (self._pointer - valid_lag) % length
        return self._buf_q_target[idx].copy(), self._buf_qd_target[idx].copy()


@dataclass
class ActuatorSet:
    """All actuator groups of one articulation, resolved onto the Isaac joint order."""

    joint_names: list[str]
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    groups: list[ActuatorGroup]
    # Per-joint PhysX >= 5 friction triple (0 where the cfg authors nothing):
    # static_friction is the breakaway effort acting on stationary joints only,
    # dynamic_friction the moving Coulomb effort (realized as dof_frictionloss),
    # viscous_friction the velocity-proportional coefficient (realized as plant
    # dof_damping, ON TOP of any implicit_pd kd damping).
    static_friction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dynamic_friction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    viscous_friction: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _overrides: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def has_delay(self) -> bool:
        return any(group.has_delay for group in self.groups)

    @property
    def has_net(self) -> bool:
        return any(group.has_net for group in self.groups)

    def load_networks(self) -> None:
        """Eagerly load every actuator-net group's TorchScript network (fail fast).

        Optional: networks otherwise load lazily on the first
        :meth:`compute_torques` call that touches them.
        """
        for group in self.groups:
            if group.has_net:
                group.load_network()

    @classmethod
    def from_ir(
        cls,
        groups: list[ActuatorGroupIR],
        isaac_joint_order: list[str],
        *,
        net_dir: str | Path | None = None,
    ) -> "ActuatorSet":
        """Build an :class:`ActuatorSet` from parsed actuator-group IR.

        Every joint must be claimed by exactly one group; unmatched patterns
        and doubly-claimed joints raise (IsaacLab semantics, made strict).
        ``stiffness``/``damping`` must be authored: IsaacLab resolves None to
        the USD drive gains, which the IR does not carry, so None raises here
        (actuator-net groups excepted — their cfgs force both to None and the
        network replaces the PD law entirely).

        ``net_dir`` is the directory actuator-net ``network_bundle_path``
        entries are relative to (the bundle directory at runtime). Network
        files are resolved here but loaded lazily; call :meth:`load_networks`
        to fail fast.
        """
        num_joints = len(isaac_joint_order)
        kp = np.zeros(num_joints, dtype=np.float64)
        kd = np.zeros(num_joints, dtype=np.float64)
        effort_limit = np.full(num_joints, np.inf, dtype=np.float64)
        static_per_joint = np.zeros(num_joints, dtype=np.float64)
        dynamic_per_joint = np.zeros(num_joints, dtype=np.float64)
        viscous_per_joint = np.zeros(num_joints, dtype=np.float64)
        owner: list[str | None] = [None] * num_joints
        overrides: dict[str, dict[str, float]] = {}
        runtime_groups: list[ActuatorGroup] = []

        for group in groups:
            what = f"actuator group '{group.name}'"
            ids = resolve_matching_names(group.joint_names_expr, isaac_joint_order, what=what)
            for joint_idx in ids:
                if owner[joint_idx] is not None:
                    raise ValueError(
                        f"joint '{isaac_joint_order[joint_idx]}' is claimed by both actuator group "
                        f"'{owner[joint_idx]}' and '{group.name}'"
                    )
                owner[joint_idx] = group.name
            group_names = [isaac_joint_order[i] for i in ids]

            if group.model in NET_MODELS:
                # The actuator-net cfgs force stiffness/damping to None; the network output
                # replaces the PD law, so the vectorized PD sees zero gains for these joints.
                kp_group = np.zeros(len(ids), dtype=np.float64)
                kd_group = np.zeros(len(ids), dtype=np.float64)
            else:
                # IsaacLab resolves a None gain to the USD-authored drive gain, which the IR
                # does not carry; silently substituting 0.0 would zero the PD torque.
                for gain_name, gain_value in (("stiffness", group.stiffness), ("damping", group.damping)):
                    if gain_value is None:
                        raise ValueError(
                            f"{what}: {gain_name} is None, which IsaacLab resolves to the USD-authored "
                            "drive gain; the converter does not carry USD gains, so the actuator config "
                            f"must author {gain_name} explicitly"
                        )
                kp_group, _ = _resolve_group_param(group.stiffness, group_names, 0.0, f"{what} stiffness")
                kd_group, _ = _resolve_group_param(group.damping, group_names, 0.0, f"{what} damping")

            lut_angle: np.ndarray | None = None
            lut_max_torque: np.ndarray | None = None
            if group.model == "remotized_pd":
                if group.joint_parameter_lookup is None:
                    raise ValueError(f"{what}: remotized_pd requires joint_parameter_lookup")
                lut = np.asarray(group.joint_parameter_lookup, dtype=np.float64)
                if lut.ndim != 2 or lut.shape[1] != 3 or lut.shape[0] < 2:
                    raise ValueError(f"{what}: joint_parameter_lookup must have shape (N>=2, 3), got {lut.shape}")
                if np.any(np.diff(lut[:, 0]) < 0):
                    raise ValueError(f"{what}: joint_parameter_lookup angles must be sorted ascending")
                lut_angle = lut[:, 0].copy()
                lut_max_torque = lut[:, 2].copy()
                # RemotizedPDActuator forces the box effort/velocity limits to inf.
                effort_group = np.full(len(ids), np.inf, dtype=np.float64)
            else:
                if group.effort_limit is None:
                    warnings.warn(
                        f"{what}: effort_limit (and effort_limit_sim) is unset; IsaacLab would clamp at the "
                        "USD-authored drive maxForce, which the converter does not read — the PD torque is "
                        "UNCLAMPED here",
                        UserWarning,
                        stacklevel=2,
                    )
                effort_group, _ = _resolve_group_param(group.effort_limit, group_names, np.inf, f"{what} effort_limit")

            saturation: np.ndarray | None = None
            dc_effort_limit: np.ndarray | None = None
            dc_velocity_limit: np.ndarray | None = None
            dc_zero_vel_box: np.ndarray | None = None
            if group.model in _DC_MOTOR_MODELS:
                # DCMotor.__init__ raises on a missing saturation_effort/velocity_limit;
                # a None effort_limit resolves to the USD-authored joint limit, which the
                # IR does not carry, so it must be authored (same rule as the gains).
                for param_name, param_value in (
                    ("saturation_effort", group.saturation_effort),
                    ("velocity_limit", group.velocity_limit),
                    ("effort_limit", group.effort_limit),
                ):
                    if param_value is None:
                        raise ValueError(
                            f"{what}: {param_name} is None; the DC-motor torque-speed curve of "
                            f"'{group.model}' needs saturation_effort, velocity_limit, and "
                            "effort_limit authored explicitly in the actuator config"
                        )
                saturation, _ = _resolve_group_param(
                    group.saturation_effort, group_names, np.nan, f"{what} saturation_effort"
                )
                dc_velocity_limit, _ = _resolve_group_param(
                    group.velocity_limit, group_names, np.nan, f"{what} velocity_limit"
                )
                dc_effort_limit = effort_group.copy()
                if np.any(saturation <= 0.0) or np.any(dc_velocity_limit <= 0.0):
                    raise ValueError(f"{what}: saturation_effort and velocity_limit must be positive")
                if group.model == "actuator_net_lstm":
                    # The LSTM's zero-velocity DCMotor clip (see the module docstring).
                    dc_zero_vel_box = np.minimum(saturation, dc_effort_limit)

            network_path: Path | None = None
            if group.model in NET_MODELS:
                if not group.network_file and not group.network_bundle_path:
                    raise ValueError(f"{what}: {group.model} requires a network_file")
                candidates: list[Path] = []
                if net_dir is not None and group.network_bundle_path:
                    candidates.append(Path(net_dir) / group.network_bundle_path)
                if group.network_file and "://" not in group.network_file:
                    candidates.append(Path(group.network_file))
                network_path = next((p for p in candidates if p.is_file()), None)

            pos_error_history: np.ndarray | None = None
            vel_history: np.ndarray | None = None
            input_idx_arr: np.ndarray | None = None
            if group.model == "actuator_net_mlp":
                # ActuatorNetMLPCfg has no defaults for the input/output contract; a dump
                # missing any of these fields cannot reproduce the network's inputs.
                for field_name in ("pos_scale", "vel_scale", "torque_scale", "input_order", "input_idx"):
                    if getattr(group, field_name) is None:
                        raise ValueError(f"{what}: actuator_net_mlp requires {field_name}")
                assert group.input_idx is not None and group.input_order is not None
                if group.input_order not in ("pos_vel", "vel_pos"):
                    raise ValueError(
                        f"{what}: invalid input order '{group.input_order}'; must be 'pos_vel' or 'vel_pos'"
                    )
                input_idx_arr = np.asarray(group.input_idx, dtype=np.intp)
                if input_idx_arr.size == 0 or np.any(input_idx_arr < 0):
                    raise ValueError(f"{what}: input_idx must be non-empty and non-negative, got {group.input_idx}")
                # ActuatorNetMLP.__init__ allocates float32 histories of max(input_idx) + 1
                # steps, zero-filled (row 0 = current step after the first compute).
                history_length = int(input_idx_arr.max()) + 1
                pos_error_history = np.zeros((history_length, len(ids)), dtype=np.float32)
                vel_history = np.zeros((history_length, len(ids)), dtype=np.float32)

            armature, armature_set = _resolve_group_param(group.armature, group_names, np.nan, f"{what} armature")
            static_friction, static_set = _resolve_group_param(group.friction, group_names, np.nan, f"{what} friction")
            dynamic_friction, dynamic_set = _resolve_group_param(
                group.dynamic_friction, group_names, np.nan, f"{what} dynamic_friction"
            )
            viscous, viscous_set = _resolve_group_param(
                group.viscous_friction, group_names, np.nan, f"{what} viscous_friction"
            )
            # PhysX >= 5 semantics: `friction` is the STATIC friction effort, a breakaway
            # threshold that only acts on stationary joints; a moving joint's friction is
            # dynamic_friction + viscous_friction * |qd|. MuJoCo's dof_frictionloss is a
            # single-valued Coulomb effort (stiction and moving resistance alike), so it
            # maps to dynamic_friction; a static effort above the dynamic one has no
            # MuJoCo counterpart and is dropped.
            dynamic_effective = np.where(dynamic_set, dynamic_friction, 0.0)
            unmodeled = static_set & (np.nan_to_num(static_friction) > dynamic_effective)
            if np.any(unmodeled):
                worst = float(np.max(np.nan_to_num(static_friction)[unmodeled] - dynamic_effective[unmodeled]))
                warnings.warn(
                    f"{what}: static friction (breakaway) exceeds dynamic friction on joints "
                    f"{[n for n, u in zip(group_names, unmodeled) if u]} by up to {worst:g}; "
                    "PhysX applies it only to stationary joints and MuJoCo has no static/dynamic "
                    "split, so the compiled model drops it (the runtime supplies it via "
                    "lab2mj.stiction.JointStiction)",
                    UserWarning,
                    stacklevel=2,
                )

            ids_arr = np.asarray(ids, dtype=np.intp)
            kp[ids_arr] = kp_group
            kd[ids_arr] = kd_group
            effort_limit[ids_arr] = effort_group
            for local_idx, joint_name in enumerate(group_names):
                entry: dict[str, float] = {}
                if armature_set[local_idx]:
                    entry["armature"] = float(armature[local_idx])
                if dynamic_set[local_idx]:
                    entry["frictionloss"] = float(dynamic_friction[local_idx])
                if viscous_set[local_idx]:
                    entry["damping"] = float(viscous[local_idx])
                if entry:
                    overrides[joint_name] = entry
                if static_set[local_idx]:
                    static_per_joint[ids_arr[local_idx]] = float(static_friction[local_idx])
                if dynamic_set[local_idx]:
                    dynamic_per_joint[ids_arr[local_idx]] = float(dynamic_friction[local_idx])
                if viscous_set[local_idx]:
                    viscous_per_joint[ids_arr[local_idx]] = float(viscous[local_idx])

            min_delay = 0
            max_delay = 0
            buf_q = buf_qd = None
            if group.model in _DELAYED_MODELS:
                min_delay = int(group.min_delay or 0)
                max_delay = int(group.max_delay or 0)
                if not 0 <= min_delay <= max_delay:
                    raise ValueError(f"{what}: invalid delay range [{min_delay}, {max_delay}]")
                buf_q = np.zeros((max_delay + 1, len(ids)), dtype=np.float64)
                buf_qd = np.zeros((max_delay + 1, len(ids)), dtype=np.float64)

            runtime_groups.append(
                ActuatorGroup(
                    name=group.name,
                    model=group.model,
                    joint_ids=ids_arr,
                    joint_names=group_names,
                    min_delay=min_delay,
                    max_delay=max_delay,
                    lut_angle=lut_angle,
                    lut_max_torque=lut_max_torque,
                    saturation_effort=saturation,
                    dc_effort_limit=dc_effort_limit,
                    dc_velocity_limit=dc_velocity_limit,
                    dc_zero_vel_box=dc_zero_vel_box,
                    network_path=network_path,
                    network_source=group.network_bundle_path or group.network_file,
                    pos_scale=1.0 if group.pos_scale is None else float(group.pos_scale),
                    vel_scale=1.0 if group.vel_scale is None else float(group.vel_scale),
                    torque_scale=1.0 if group.torque_scale is None else float(group.torque_scale),
                    input_order=group.input_order or "pos_vel",
                    input_idx=input_idx_arr,
                    _pos_error_history=pos_error_history,
                    _vel_history=vel_history,
                    _buf_q_target=buf_q,
                    _buf_qd_target=buf_qd,
                )
            )

        uncovered = [name for name, group_name in zip(isaac_joint_order, owner) if group_name is None]
        if uncovered:
            raise ValueError(
                f"joints {uncovered} are not covered by any actuator group; the converter cannot "
                "reproduce PhysX's USD-authored fallback drive for unactuated joints"
            )
        return cls(
            joint_names=list(isaac_joint_order),
            kp=kp,
            kd=kd,
            effort_limit=effort_limit,
            groups=runtime_groups,
            static_friction=static_per_joint,
            dynamic_friction=dynamic_per_joint,
            viscous_friction=viscous_per_joint,
            _overrides=overrides,
        )

    def reset(self, rng: np.random.Generator | None = None, strict: bool = False) -> None:
        """Resample per-group delay lags, clear delay buffers, and zero LSTM states.

        ``strict=True`` pins every lag to 0 (strict-match eval); otherwise the
        lag is drawn as ``rng.integers(min_delay, max_delay + 1)`` per group,
        matching ``DelayedPDActuator.reset``. Actuator-net groups zero their
        state (LSTM hidden/cell, MLP histories) in both modes — the net has
        no randomness.
        """
        for group in self.groups:
            if group.has_net:
                group.reset_network_state()
            if not group.has_delay:
                continue
            if strict:
                lag = 0
            else:
                if rng is None:
                    raise ValueError("reset with strict=False requires a numpy Generator to sample delay lags")
                lag = int(rng.integers(group.min_delay, group.max_delay + 1))
            group.reset(lag)

    def set_lags(self, lags: dict[str, int]) -> None:
        """Pin per-group delay lags to recorded Isaac values (after :meth:`reset`).

        ``lags`` maps group name -> lag in physics steps (a dump's
        ``init_actuator_lag_<group>`` keys). Unknown group names raise — a
        mismatch means the dump and bundle disagree about the actuator layout.
        """
        by_name = {group.name: group for group in self.groups}
        for name, lag in lags.items():
            group = by_name.get(name)
            if group is None:
                raise KeyError(f"actuator lag recorded for unknown group '{name}' (bundle has {sorted(by_name)})")
            group.reset(int(lag))

    def warm_start(self, q_isaac: np.ndarray, qd_isaac: np.ndarray, q_des_isaac: np.ndarray) -> None:
        """Warm-start the MLP actuator-net groups' state for a reference-dump init.

        ``q_des_isaac`` must be the position targets that produced the recorded
        init state (the dump's last pre-recording action through the action
        processor); each MLP group receives the t0-consistent constant input
        ``[q_des - q, qd]`` (see :meth:`ActuatorGroup.warm_start_network_state`).
        Delayed groups' target buffers are filled with ``q_des`` (zero velocity
        targets) so a nonzero lag serves the settled target instead of a
        reset-tile-filled first push; Isaac's settled DelayBuffer holds the
        recent target history, which the constant settled target approximates.
        Call after :meth:`reset`; other groups are untouched.

        LSTM groups deliberately stay cold: for the MLP the constant-input
        buffer fill is the exact Isaac history at a settled state, but the
        LSTM's constant-input fixed point is not the state Isaac's settle phase
        (oscillating stance inputs) actually leaves behind — against ANYdrive
        reference dumps the fixed-point state tracks the reference *worse* than
        a cold state (open-loop replay error grows by up to ~0.02 rad and flips
        a passing gate), so only the exactly-reconstructable MLP state warms.
        """
        q = self._as_joint_vector(q_isaac, "q_isaac")
        qd = self._as_joint_vector(qd_isaac, "qd_isaac")
        q_des = self._as_joint_vector(q_des_isaac, "q_des_isaac")
        for group in self.groups:
            ids = group.joint_ids
            if group.has_delay:
                assert group._buf_q_target is not None and group._buf_qd_target is not None
                group._buf_q_target[:] = q_des[ids]
                group._buf_qd_target[:] = 0.0
                group._num_pushes = group.max_delay + 1
            if group.model == "actuator_net_mlp":
                group.warm_start_network_state(q_des[ids] - q[ids], qd[ids])

    def step_delay_buffers(
        self, q_target_isaac: np.ndarray, qd_target_isaac: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the delay buffers by one physics step; return effective targets.

        Call exactly once per physics step. Groups without a delay model pass
        their targets through unchanged.
        """
        q_target = self._as_joint_vector(q_target_isaac, "q_target_isaac")
        qd_target = (
            np.zeros(self.num_joints, dtype=np.float64)
            if qd_target_isaac is None
            else self._as_joint_vector(qd_target_isaac, "qd_target_isaac")
        )
        q_eff = q_target.copy()
        qd_eff = qd_target.copy()
        for group in self.groups:
            if not group.has_delay:
                continue
            ids = group.joint_ids
            q_eff[ids], qd_eff[ids] = group.push(q_target[ids], qd_target[ids])
        return q_eff, qd_eff

    def compute_torques(
        self,
        q_isaac: np.ndarray,
        qd_isaac: np.ndarray,
        q_target_isaac: np.ndarray,
        qd_target_isaac: np.ndarray | None = None,
    ) -> np.ndarray:
        """Joint torques with IsaacLab clamping semantics.

        ``tau = kp*(q_des - q) + kd*(qd_des - qd)`` clipped to ``+-effort_limit``
        (inf for remotized groups), then per-group post-processing:

        - remotized groups are clamped to the interpolated angle-dependent
          torque envelope;
        - ``dc_motor`` groups are clamped to the torque-speed curve at the
          current joint velocity (:func:`dc_motor_clip`);
        - ``actuator_net_lstm`` groups replace the PD torque with the network
          output (advancing the LSTM state — call exactly once per physics
          step), clamped by the torque-speed curve at **zero** velocity,
          mirroring Isaac's ``ActuatorNetLSTM`` whose ``_joint_vel`` clip
          buffer is never written;
        - ``actuator_net_mlp`` groups replace the PD torque with the network
          output (advancing the history buffers — call exactly once per
          physics step), clamped by the torque-speed curve at the **current**
          joint velocity, mirroring Isaac's ``ActuatorNetMLP`` which does
          write ``_joint_vel`` before clipping.

        Targets for delayed groups should already have passed through
        :meth:`step_delay_buffers`. Stateless unless an actuator-net group
        exists.
        """
        q = self._as_joint_vector(q_isaac, "q_isaac")
        qd = self._as_joint_vector(qd_isaac, "qd_isaac")
        q_des = self._as_joint_vector(q_target_isaac, "q_target_isaac")
        qd_des = (
            np.zeros(self.num_joints, dtype=np.float64)
            if qd_target_isaac is None
            else self._as_joint_vector(qd_target_isaac, "qd_target_isaac")
        )
        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd)
        np.clip(tau, -self.effort_limit, self.effort_limit, out=tau)
        for group in self.groups:
            ids = group.joint_ids
            if group.model == "actuator_net_lstm":
                assert group.dc_zero_vel_box is not None
                tau_net = group.compute_net_torques(q_des[ids] - q[ids], qd[ids])
                tau[ids] = np.clip(tau_net, -group.dc_zero_vel_box, group.dc_zero_vel_box)
            elif group.model == "actuator_net_mlp":
                assert group.saturation_effort is not None
                assert group.dc_effort_limit is not None and group.dc_velocity_limit is not None
                tau_net = group.compute_mlp_torques(q_des[ids] - q[ids], qd[ids])
                tau[ids] = dc_motor_clip(
                    tau_net, qd[ids], group.saturation_effort, group.dc_effort_limit, group.dc_velocity_limit
                )
            elif group.model == "dc_motor":
                assert group.saturation_effort is not None
                assert group.dc_effort_limit is not None and group.dc_velocity_limit is not None
                tau[ids] = dc_motor_clip(
                    tau[ids], qd[ids], group.saturation_effort, group.dc_effort_limit, group.dc_velocity_limit
                )
            elif group.lut_angle is not None and group.lut_max_torque is not None:
                tau_max = np.interp(q[ids], group.lut_angle, group.lut_max_torque)
                tau[ids] = np.clip(tau[ids], -tau_max, tau_max)
        return tau

    def contact_profile(self) -> str:
        """Contact profile the compiled model should author: ``"default"`` or ``"engagement"``.

        Soft-gain DC-motor robots (``dc_motor`` groups, and ``actuator_net_mlp``
        groups — MLP nets carry the same DCMotor clip and hardware class) resolve
        ground forces through light legs and low PD stiffness, so contact-force
        error converts directly into joint error; their policies also tend to
        slide the feet in near-continuous contact, where MuJoCo's default
        hard-at-first-touch pyramidal contact systematically overshoots the PhysX
        response. Validated against Isaac reference dumps, these robots' strict
        open-loop replay error drops several-fold under the builder's
        "engagement" profile (elliptic cone, impratio 3, progressive-engagement
        solimp), while stiff-PD and series-elastic (``actuator_net_lstm``) robots
        track their references better with the MuJoCo default — so the profile is
        selected from the actuator family rather than authored globally.
        """
        if any(group.model in _ENGAGEMENT_CONTACT_MODELS for group in self.groups):
            return "engagement"
        return "default"

    def builder_overrides(self) -> dict[str, dict[str, float]]:
        """Per-joint MJCF model overrides: ``{joint_name: {armature, frictionloss, damping}}``.

        Only joints whose actuator config explicitly set armature /
        dynamic_friction / viscous_friction appear; these belong in the
        compiled model, not in the torque math. The cfg's static ``friction``
        never converts (see the module docstring).
        """
        return {name: dict(entry) for name, entry in self._overrides.items()}

    def _as_joint_vector(self, value: np.ndarray, what: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (self.num_joints,):
            raise ValueError(f"{what}: expected shape ({self.num_joints},), got {arr.shape}")
        return arr
