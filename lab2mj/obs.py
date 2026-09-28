"""Observation pipeline for the MuJoCo runtime, mirroring IsaacLab's ``ObservationManager``.

Pure numpy — no mujoco, torch, or isaaclab imports. The runtime env fills an
:class:`ObsContext` each policy step; :class:`ObsPipeline` (built from an
:class:`~lab2mj.ir.ObsGroupIR`) turns it into the flat observation
vector the policy consumes.

Per-term processing order matches ``ObservationManager.compute_group``:

1. evaluate the term function
2. apply noise (only when the group has ``enable_corruption`` and noise is on)
3. clip
4. scale (scalar or per-element)
5. append to the term's history buffer (oldest-first; the first append after a
   reset fills every slot with the current value)
6. flatten the history dimension when ``flatten_history_dim`` is set

A group-level ``history_length`` overrides both ``history_length`` and
``flatten_history_dim`` on every term. Concatenation is per-term contiguous
along the group's ``concatenate_dim``.

Extras convention: observation terms that need state the runtime context does
not model (height scans, estimator ground-truth wrenches, ...) are served from
``ctx.extras``. The key is the **term name** and the value is the raw term
output of shape ``(dim,)`` (pre noise/clip/scale/history). Such terms build
only when their dimension is declared via ``extras_dims={term_name: dim}``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lab2mj.env_yaml import class_name
from lab2mj.ir import NoiseIR, ObsGroupIR, ObsTermIR

_NoiseFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]
_EvalFn = Callable[["ObsContext"], np.ndarray]

# Terms whose IsaacLab implementations read simulator state the runtime context
# does not model; they resolve through ``ctx.extras`` (see module docstring).
_EXTRAS_FUNCS = frozenset(
    {
        "height_scan",
        "external_wrench_b",
        "external_force_b",
        "external_torque_b",
        "foot_grf_magnitude",
        "time_remaining",
        "body_incoming_wrench",
    }
)

_KNOWN_FUNCS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "generated_commands",
    "command_time_remaining",
    "phase",
)
# contact_lab's ``gait_state`` reports phase 0 below this commanded frequency (no gait).
_GAIT_FREQ_EPS = 1e-6


def extras_term_names(group: ObsGroupIR) -> list[str]:
    """Names of the group's terms that need a runtime extras provider (``ctx.extras``)."""
    return [term.name for term in group.terms if class_name(term.func) in _EXTRAS_FUNCS]


@dataclass
class ObsContext:
    """Robot/command state the runtime env populates once per policy step.

    All arrays are 1D float-like; joint quantities are in **Isaac joint order**
    (PhysX breadth-first), not MJCF order.

    Attributes:
        root_lin_vel_b: Root **CoM** linear velocity expressed in the body
            frame, shape (3,) (Isaac's ``root_lin_vel_b``).
        root_ang_vel_b: Root angular velocity in the body frame, shape (3,).
        projected_gravity_b: Unit gravity direction in the body frame
            (``R_wb^T @ gravity_dir_w``), shape (3,).
        joint_pos_isaac: Joint positions, shape (num_joints,).
        joint_vel_isaac: Joint velocities, shape (num_joints,).
        default_joint_pos_isaac: Default joint positions, shape (num_joints,).
        default_joint_vel_isaac: Default joint velocities, shape (num_joints,).
        last_action_raw: RAW policy output from the previous step (pre
            scale/offset), shape (action_dim,).
        commands: Command name -> current command vector.
        command_time_left: Command name -> seconds until the command resamples.
        extras: Term name -> raw term value for extras-backed terms (see
            module docstring).
    """

    root_lin_vel_b: np.ndarray
    root_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    joint_pos_isaac: np.ndarray
    joint_vel_isaac: np.ndarray
    default_joint_pos_isaac: np.ndarray
    default_joint_vel_isaac: np.ndarray
    last_action_raw: np.ndarray
    commands: dict[str, np.ndarray] = field(default_factory=dict)
    command_time_left: dict[str, float] = field(default_factory=dict)
    extras: dict[str, np.ndarray] = field(default_factory=dict)


class _History:
    """Single-env mirror of ``isaaclab.utils.buffers.CircularBuffer``."""

    def __init__(self, max_len: int, dim: int) -> None:
        if max_len < 1:
            raise ValueError(f"history length must be >= 1, got {max_len}")
        self._max_len = max_len
        self._buf = np.zeros((max_len, dim), dtype=np.float32)
        self._pointer = -1
        self._num_pushes = 0

    def reset(self) -> None:
        self._num_pushes = 0
        self._buf[:] = 0.0

    def append(self, value: np.ndarray) -> None:
        self._pointer = (self._pointer + 1) % self._max_len
        self._buf[self._pointer] = value
        if self._num_pushes == 0:
            # First push after reset fills every slot with the current value.
            self._buf[:] = value
        self._num_pushes += 1

    def buffer(self) -> np.ndarray:
        """History with the oldest entry first, shape (max_len, dim)."""
        return np.roll(self._buf, self._max_len - self._pointer - 1, axis=0)


def _apply_operation(data: np.ndarray, draw: np.ndarray, operation: str) -> np.ndarray:
    if operation == "add":
        return data + draw
    if operation == "scale":
        return data * draw
    if operation == "abs":
        return np.broadcast_to(draw, data.shape).astype(np.float64, copy=True)
    raise ValueError(f"Unknown operation in noise: {operation}")


def _build_noise_fn(noise: NoiseIR, label: str) -> _NoiseFn:
    operation = noise.operation
    if operation not in ("add", "scale", "abs"):
        raise ValueError(f"obs term '{label}': unknown noise operation '{operation}'")
    params = noise.params
    if noise.kind == "uniform":
        n_min = np.asarray(params.get("n_min", -1.0), dtype=np.float64)
        n_max = np.asarray(params.get("n_max", 1.0), dtype=np.float64)

        def fn(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            # Mirror IsaacLab's torch.rand_like(data) * (n_max - n_min) + n_min,
            # which tolerates n_min > n_max (samples the reversed interval).
            draw = n_min + rng.random(size=data.shape) * (n_max - n_min)
            return _apply_operation(data, draw, operation).astype(np.float32)

    elif noise.kind == "gaussian":
        mean = np.asarray(params.get("mean", 0.0), dtype=np.float64)
        std = np.asarray(params.get("std", 1.0), dtype=np.float64)

        def fn(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            draw = mean + std * rng.standard_normal(size=data.shape)
            return _apply_operation(data, draw, operation).astype(np.float32)

    elif noise.kind == "constant":
        bias = np.asarray(params.get("bias", 0.0), dtype=np.float64)

        def fn(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return _apply_operation(data, bias, operation).astype(np.float32)

    else:
        raise ValueError(
            f"obs term '{label}': unsupported noise kind '{noise.kind}' (known: uniform, gaussian, constant)"
        )
    return fn


def _check_asset_cfg(term: ObsTermIR, label: str, *, entity: str, check_joint_ids: bool) -> None:
    asset_cfg = term.params.get("asset_cfg")
    if asset_cfg is None:
        return
    name = asset_cfg.get("name")
    if name not in (None, entity):
        raise ValueError(f"obs term '{label}': only the '{entity}' articulation is supported, got asset '{name}'")
    if check_joint_ids:
        if asset_cfg.get("joint_ids") is not None:
            raise ValueError(f"obs term '{label}': joint_ids subsets are not supported (all joints only)")
        if asset_cfg.get("joint_names") is not None:
            raise ValueError(f"obs term '{label}': joint_names subsets are not supported (all joints only)")
        if asset_cfg.get("preserve_order"):
            raise ValueError(f"obs term '{label}': preserve_order joint reordering is not supported")


@dataclass
class _TermSpec:
    name: str
    evaluate: _EvalFn
    dim: int  # per-frame dimension (before history stacking)
    noise_fn: _NoiseFn | None
    clip: tuple[float, float] | None
    scale: np.ndarray | None
    history: _History | None
    flatten_history_dim: bool

    @property
    def out_shape(self) -> tuple[int, ...]:
        if self.history is None:
            return (self.dim,)
        max_len = self.history._max_len
        if self.flatten_history_dim:
            return (max_len * self.dim,)
        return (max_len, self.dim)


class ObsPipeline:
    """Evaluates one observation group over an :class:`ObsContext`.

    Args:
        group: Observation group IR (from ``parse_env_yaml``).
        num_joints: Number of articulation joints (Isaac order).
        action_dim: Dimension of the raw policy action.
        command_dims: Command name -> command vector dimension; required for
            every ``generated_commands`` term.
        command_resample_time_s: Command name -> max of the command's
            ``resampling_time_range``; required for ``command_time_remaining``.
        extras_dims: Term name -> dimension for extras-backed terms (see
            module docstring).
        enable_noise: Construction-time noise default. ``None`` follows the
            group's ``enable_corruption``. Noise never applies when the group
            has ``enable_corruption=False`` (IsaacLab drops term noise at
            manager init in that case).
        entity: Scene key of the robot articulation; term ``asset_cfg``s
            naming any other entity raise at construction.
    """

    def __init__(
        self,
        group: ObsGroupIR,
        *,
        num_joints: int,
        action_dim: int,
        command_dims: dict[str, int] | None = None,
        command_resample_time_s: dict[str, float] | None = None,
        extras_dims: dict[str, int] | None = None,
        enable_noise: bool | None = None,
        entity: str = "robot",
    ) -> None:
        if not group.concatenate_terms:
            raise ValueError(
                f"observation group '{group.name}' has concatenate_terms=False; the MuJoCo runtime "
                "requires a flat concatenated observation vector"
            )
        self._group = group
        self._enable_noise_default = group.enable_corruption if enable_noise is None else bool(enable_noise)
        command_dims = command_dims or {}
        command_resample_time_s = command_resample_time_s or {}
        extras_dims = extras_dims or {}

        self._entity = str(entity)
        self._specs: list[_TermSpec] = []
        for term in group.terms:
            label = f"{group.name}/{term.name}"
            evaluate, dim = self._resolve_evaluator(
                term,
                label,
                num_joints=num_joints,
                action_dim=action_dim,
                command_dims=command_dims,
                command_resample_time_s=command_resample_time_s,
                extras_dims=extras_dims,
            )

            # Group-level history overrides both fields on every term.
            history_length = term.history_length
            flatten_history_dim = term.flatten_history_dim
            if group.history_length is not None:
                history_length = group.history_length
                flatten_history_dim = group.flatten_history_dim

            noise_fn = None
            if group.enable_corruption and term.noise is not None:
                noise_fn = _build_noise_fn(term.noise, label)

            scale = self._resolve_scale(term.scale, dim, label)
            history = _History(history_length, dim) if history_length > 0 else None
            self._specs.append(
                _TermSpec(
                    name=term.name,
                    evaluate=evaluate,
                    dim=dim,
                    noise_fn=noise_fn,
                    clip=term.clip,
                    scale=scale,
                    history=history,
                    flatten_history_dim=flatten_history_dim,
                )
            )

        self._cat_axis = self._validate_concat()
        self._layout = self._build_layout()
        self._obs_dim = int(sum(int(np.prod(spec.out_shape)) for spec in self._specs))

    """
    Properties.
    """

    @property
    def group_name(self) -> str:
        return self._group.name

    @property
    def obs_dim(self) -> int:
        """Total flattened dimension of the group observation vector."""
        return self._obs_dim

    """
    Operations.
    """

    def reset(self) -> None:
        """Reset all history buffers (next compute refills them)."""
        for spec in self._specs:
            if spec.history is not None:
                spec.history.reset()

    def seed_history_from_flat(self, flat_obs: np.ndarray) -> None:
        """Seed history buffers from a recorded flat observation (settled-dump init).

        A settled Isaac dump's first observation carries distinct rolling
        settle-phase frames in each history term; a plain reset tile-fills the
        buffers with the t0 frame instead. Seeding copies the recorded older
        frames (all but the newest) into the buffers so the next
        :meth:`compute` appends the fresh t0 frame on top, reproducing the
        recorded history exactly. Values are post noise/clip/scale — the same
        form the buffers store. Call after :meth:`reset`.
        """
        if self._layout is None:
            raise ValueError(
                f"observation group '{self._group.name}' contains non-flattened history terms; "
                "its flat observation cannot seed the history buffers"
            )
        flat = np.asarray(flat_obs, dtype=np.float32).reshape(-1)
        if flat.shape[0] != self._obs_dim:
            raise ValueError(f"flat obs has {flat.shape[0]} elements, expected {self._obs_dim}")
        for spec, (_, sl, _) in zip(self._specs, self._layout):
            if spec.history is None:
                continue
            frames = flat[sl].reshape(spec.history._max_len, spec.dim)
            for frame in frames[:-1]:  # oldest first; the newest lands via the next compute
                spec.history.append(frame)

    def layout(self) -> list[tuple[str, slice, int]]:
        """Ordered ``(term_name, slice, dim)`` entries for the bundle manifest."""
        if self._layout is None:
            raise ValueError(
                f"observation group '{self._group.name}' contains non-flattened history terms; "
                "per-term slices in the flat vector are interleaved and layout() is undefined"
            )
        return list(self._layout)

    def compute(
        self,
        ctx: ObsContext,
        rng: np.random.Generator | None = None,
        *,
        enable_noise: bool | None = None,
    ) -> np.ndarray:
        """Compute the flat float32 observation vector for the current step.

        History buffers are appended to on every call (one call per policy
        step). ``enable_noise`` overrides the construction-time default;
        strict mode is ``enable_noise=False``.
        """
        noise_on = self._enable_noise_default if enable_noise is None else bool(enable_noise)
        outputs: list[np.ndarray] = []
        for spec in self._specs:
            value = np.array(spec.evaluate(ctx), dtype=np.float32).reshape(-1)
            if value.shape[0] != spec.dim:
                raise ValueError(
                    f"obs term '{self._group.name}/{spec.name}' produced {value.shape[0]} elements, expected {spec.dim}"
                )
            if noise_on and spec.noise_fn is not None:
                if rng is None:
                    raise ValueError(
                        f"obs term '{self._group.name}/{spec.name}' has noise enabled but no rng was provided"
                    )
                value = spec.noise_fn(value, rng)
            if spec.clip is not None:
                np.clip(value, spec.clip[0], spec.clip[1], out=value)
            if spec.scale is not None:
                value *= spec.scale
            if spec.history is not None:
                spec.history.append(value)
                stacked = spec.history.buffer()
                value = stacked.reshape(-1) if spec.flatten_history_dim else stacked
            outputs.append(value)

        if self._layout is not None:
            return np.concatenate(outputs, axis=0, dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(outputs, axis=self._cat_axis, dtype=np.float32)).reshape(-1)

    """
    Helper functions.
    """

    def _resolve_evaluator(
        self,
        term: ObsTermIR,
        label: str,
        *,
        num_joints: int,
        action_dim: int,
        command_dims: dict[str, int],
        command_resample_time_s: dict[str, float],
        extras_dims: dict[str, int],
    ) -> tuple[_EvalFn, int]:
        func_name = class_name(term.func)

        if func_name == "base_lin_vel":
            _check_asset_cfg(term, label, entity=self._entity, check_joint_ids=False)
            return (lambda ctx: ctx.root_lin_vel_b), 3
        if func_name == "base_ang_vel":
            _check_asset_cfg(term, label, entity=self._entity, check_joint_ids=False)
            return (lambda ctx: ctx.root_ang_vel_b), 3
        if func_name == "projected_gravity":
            _check_asset_cfg(term, label, entity=self._entity, check_joint_ids=False)
            return (lambda ctx: ctx.projected_gravity_b), 3
        if func_name == "joint_pos_rel":
            _check_asset_cfg(term, label, entity=self._entity, check_joint_ids=True)
            return (lambda ctx: np.asarray(ctx.joint_pos_isaac) - np.asarray(ctx.default_joint_pos_isaac)), num_joints
        if func_name == "joint_vel_rel":
            _check_asset_cfg(term, label, entity=self._entity, check_joint_ids=True)
            return (lambda ctx: np.asarray(ctx.joint_vel_isaac) - np.asarray(ctx.default_joint_vel_isaac)), num_joints
        if func_name == "last_action":
            if term.params.get("action_name") is not None:
                raise ValueError(f"obs term '{label}': last_action with a named action term is not supported")
            return (lambda ctx: ctx.last_action_raw), action_dim
        if func_name == "generated_commands":
            command_name = term.params.get("command_name")
            if command_name is None:
                raise ValueError(f"obs term '{label}': generated_commands requires params.command_name")
            if command_name not in command_dims:
                raise ValueError(
                    f"obs term '{label}': dimension of command '{command_name}' unknown; "
                    f"pass command_dims={{'{command_name}': <dim>}}"
                )

            def eval_command(ctx: ObsContext, _name: str = command_name) -> np.ndarray:
                if _name not in ctx.commands:
                    raise KeyError(f"ObsContext.commands missing '{_name}' (have {sorted(ctx.commands)})")
                return ctx.commands[_name]

            return eval_command, command_dims[command_name]
        if func_name == "command_time_remaining":
            command_name = term.params.get("command_name", "pose_command")
            if command_name not in command_resample_time_s:
                raise ValueError(
                    f"obs term '{label}': max resampling time of command '{command_name}' unknown; "
                    f"pass command_resample_time_s={{'{command_name}': <resampling_time_range[1]>}}"
                )
            max_time = float(command_resample_time_s[command_name])

            def eval_time_remaining(ctx: ObsContext, _name: str = command_name, _max: float = max_time) -> np.ndarray:
                if _name not in ctx.command_time_left:
                    have = sorted(ctx.command_time_left)
                    raise KeyError(f"ObsContext.command_time_left missing '{_name}' (have {have})")
                value = min(max(ctx.command_time_left[_name] / _max, 0.0), 1.0)
                return np.array([value], dtype=np.float32)

            return eval_time_remaining, 1
        if func_name == "phase":
            # contact_lab gait phase: [sin, cos](2 pi phase) of a [frequency, phase] command.
            if term.params.get("source", "command") != "command":
                raise ValueError(f"obs term '{label}': phase from a gait-frequency action is not supported")
            command_name = term.params.get("command_name", "frequency")
            if command_dims.get(command_name) != 2:
                raise ValueError(
                    f"obs term '{label}': phase needs a [frequency, phase] command '{command_name}' "
                    f"(command dims: {command_dims})"
                )

            def eval_phase(ctx: ObsContext, _name: str = command_name) -> np.ndarray:
                freq, phase = ctx.commands[_name][:2]
                phase = 0.0 if abs(freq) < _GAIT_FREQ_EPS else phase
                return np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float32)

            return eval_phase, 2
        if func_name in _EXTRAS_FUNCS:
            if term.name not in extras_dims:
                raise ValueError(
                    f"obs term '{label}' (func '{term.func}') requires a runtime extras provider: "
                    f"pass extras_dims={{'{term.name}': <dim>}} and supply ctx.extras['{term.name}'] each step"
                )

            def eval_extras(ctx: ObsContext, _name: str = term.name) -> np.ndarray:
                if _name not in ctx.extras:
                    raise KeyError(f"ObsContext.extras missing '{_name}' (have {sorted(ctx.extras)})")
                return ctx.extras[_name]

            return eval_extras, int(extras_dims[term.name])

        raise ValueError(
            f"obs term '{label}': unknown observation func '{term.func}' "
            f"(supported: {', '.join(_KNOWN_FUNCS)}; extras-backed: {', '.join(sorted(_EXTRAS_FUNCS))})"
        )

    @staticmethod
    def _resolve_scale(scale: float | list[float] | dict[str, float] | None, dim: int, label: str) -> np.ndarray | None:
        if scale is None:
            return None
        if isinstance(scale, dict):
            raise ValueError(f"obs term '{label}': scale must be a float or per-element sequence, got a dict")
        if isinstance(scale, (list, tuple)):
            if len(scale) != dim:
                raise ValueError(f"obs term '{label}': per-element scale has {len(scale)} entries, expected {dim}")
            return np.asarray(scale, dtype=np.float32)
        return np.asarray(scale, dtype=np.float32)

    def _validate_concat(self) -> int:
        """Validate term shapes for concatenation; return the numpy concat axis.

        IsaacLab concatenates batched ``(num_envs, ...)`` tensors along
        ``concatenate_dim + 1`` for non-negative dims; without the batch axis
        that maps back to ``concatenate_dim`` itself.
        """
        cat_axis = self._group.concatenate_dim
        ndims = {len(spec.out_shape) for spec in self._specs}
        if len(ndims) > 1:
            shapes = {spec.name: spec.out_shape for spec in self._specs}
            raise ValueError(
                f"observation group '{self._group.name}': cannot concatenate terms with mixed shapes {shapes}; "
                "set flatten_history_dim consistently"
            )
        if not self._specs:
            raise ValueError(f"observation group '{self._group.name}' has no observation terms")
        ndim = ndims.pop()
        if cat_axis >= ndim or cat_axis < -ndim:
            raise ValueError(
                f"observation group '{self._group.name}': concatenate_dim {cat_axis} is out of range for "
                f"{ndim}-dimensional terms"
            )
        if ndim == 2:
            axis = cat_axis % 2
            other = 1 - axis
            other_sizes = {spec.out_shape[other] for spec in self._specs}
            if len(other_sizes) > 1:
                shapes = {spec.name: spec.out_shape for spec in self._specs}
                raise ValueError(
                    f"observation group '{self._group.name}': term shapes {shapes} are incompatible for "
                    f"concatenation along dim {cat_axis}"
                )
        return cat_axis

    def _build_layout(self) -> list[tuple[str, slice, int]] | None:
        if any(len(spec.out_shape) != 1 for spec in self._specs):
            return None
        layout = []
        offset = 0
        for spec in self._specs:
            dim = spec.out_shape[0]
            layout.append((spec.name, slice(offset, offset + dim), dim))
            offset += dim
        return layout
