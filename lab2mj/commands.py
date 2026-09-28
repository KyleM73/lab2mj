"""Command generators for the MuJoCo runtime, mirroring IsaacLab's command manager.

Single-env, numpy-only ports of the IsaacLab v2.3.0 ``CommandTerm`` compute flow:

* ``reset(rng, robot_state)`` zeroes the command counter and resamples both the command
  and its expiry clock (``CommandTerm.reset`` -> ``_resample``).
* ``step(dt, robot_state, rng)`` mirrors ``CommandTerm.compute``: decrement ``time_left``
  by the policy dt, resample when it hits zero, then post-process the command every step
  (``_update_command``).

A fixed command (required by strict sim2sim eval, optional otherwise) is pinned at
reset: no resample clock, no standing-env zeroing, no heading-error control. Pose
commands pin the *world-frame* goal and still re-target the body-frame command from the
current root state each step.

All quaternions are wxyz. Frame suffixes: ``_b`` body, ``_w`` world.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lab2mj.ir import CommandIR
from lab2mj.quat import quat_apply_inverse, wrap_to_pi, yaw_quat

# ---------------------------------------------------------------------------------------
# Robot state handed to the generators each step.
# ---------------------------------------------------------------------------------------


@dataclass
class RobotState:
    """Minimal root state the command generators need (provided by the runtime)."""

    root_pos_w: np.ndarray
    root_quat_w_wxyz: np.ndarray
    heading_w: float
    env_origin_w: np.ndarray

    def __post_init__(self) -> None:
        self.root_pos_w = np.asarray(self.root_pos_w, dtype=np.float64)
        self.root_quat_w_wxyz = np.asarray(self.root_quat_w_wxyz, dtype=np.float64)
        self.heading_w = float(self.heading_w)
        self.env_origin_w = np.asarray(self.env_origin_w, dtype=np.float64)


def _range(pair: Any) -> tuple[float, float]:
    return (float(pair[0]), float(pair[1]))


# ---------------------------------------------------------------------------------------
# Base generator: IsaacLab CommandTerm resample clock.
# ---------------------------------------------------------------------------------------


class CommandGenerator(ABC):
    """Single-env port of ``isaaclab.managers.command_manager.CommandTerm``."""

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        if strict and fixed_command is None:
            raise ValueError("strict mode requires a fixed_command to pin")
        self.resampling_time_range = _range(resampling_time_range)
        self.time_left: float = 0.0
        self.command_counter: int = 0
        self._strict = bool(strict)
        self._fixed_command = None if fixed_command is None else np.asarray(fixed_command, dtype=np.float64)

    @property
    @abstractmethod
    def command(self) -> np.ndarray:
        """Current command vector."""

    def reset(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.command_counter = 0
        if self._fixed_command is not None:
            self.time_left = math.inf
            self._pin_command(self._fixed_command)
        else:
            self._resample(rng, robot_state)

    def step(self, dt: float, robot_state: RobotState, rng: np.random.Generator) -> None:
        if self._fixed_command is None:
            self.time_left -= dt
            if self.time_left <= 0.0:
                self._resample(rng, robot_state)
        self._update_command(robot_state)

    def retarget(self, robot_state: RobotState) -> None:
        """Re-run the per-step command post-processing without ticking the resample clock.

        Continuing a settled episode (reference-dump init with settle steps):
        Isaac ran ``_update_command`` on every settle step, so the state-dependent
        command parts (the pose command's body-frame goal) are live at t0 rather
        than the post-reset zeros.
        """
        self._update_command(robot_state)

    def _resample(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.time_left = float(rng.uniform(*self.resampling_time_range))
        self._resample_command(rng, robot_state)
        self.command_counter += 1

    @abstractmethod
    def _resample_command(self, rng: np.random.Generator, robot_state: RobotState) -> None: ...

    @abstractmethod
    def _update_command(self, robot_state: RobotState) -> None: ...

    @abstractmethod
    def _pin_command(self, fixed: np.ndarray) -> None: ...


class UniformVelocityCommand(CommandGenerator):
    """SE(2) velocity command ``[vx_b, vy_b, wz_b]`` (UniformVelocityCommand in IsaacLab).

    With ``heading_command=True``, heading-controlled envs replace the sampled yaw rate by
    ``clip(heading_control_stiffness * wrap_to_pi(heading_target - heading_w), *ang_vel_z)``.
    Standing envs (prob ``rel_standing_envs``) have the whole command zeroed every step.
    """

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        lin_vel_x: tuple[float, float],
        lin_vel_y: tuple[float, float],
        ang_vel_z: tuple[float, float],
        heading: tuple[float, float] | None = None,
        heading_command: bool = False,
        heading_control_stiffness: float = 1.0,
        rel_standing_envs: float = 0.0,
        rel_heading_envs: float = 1.0,
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        super().__init__(resampling_time_range, strict=strict, fixed_command=fixed_command)
        if heading_command and heading is None:
            raise ValueError("heading_command=True requires a heading range")
        self.lin_vel_x = _range(lin_vel_x)
        self.lin_vel_y = _range(lin_vel_y)
        self.ang_vel_z = _range(ang_vel_z)
        self.heading = None if heading is None else _range(heading)
        self.heading_command = bool(heading_command)
        self.heading_control_stiffness = float(heading_control_stiffness)
        self.rel_standing_envs = float(rel_standing_envs)
        self.rel_heading_envs = float(rel_heading_envs)

        self.vel_command_b = np.zeros(3, dtype=np.float64)
        self.heading_target = 0.0
        self.is_heading_env = False
        self.is_standing_env = False

    @property
    def command(self) -> np.ndarray:
        return self.vel_command_b

    def _pin_command(self, fixed: np.ndarray) -> None:
        if fixed.shape != (3,):
            raise ValueError(f"velocity fixed_command must have shape (3,), got {fixed.shape}")
        self.vel_command_b = fixed.copy()
        self.is_heading_env = False
        self.is_standing_env = False

    def _resample_command(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.vel_command_b[0] = rng.uniform(*self.lin_vel_x)
        self.vel_command_b[1] = rng.uniform(*self.lin_vel_y)
        self.vel_command_b[2] = rng.uniform(*self.ang_vel_z)
        if self.heading_command:
            assert self.heading is not None
            self.heading_target = float(rng.uniform(*self.heading))
            self.is_heading_env = bool(rng.uniform(0.0, 1.0) <= self.rel_heading_envs)
        self.is_standing_env = bool(rng.uniform(0.0, 1.0) <= self.rel_standing_envs)

    def _update_command(self, robot_state: RobotState) -> None:
        if self.heading_command and self.is_heading_env:
            heading_error = wrap_to_pi(self.heading_target - robot_state.heading_w)
            self.vel_command_b[2] = float(
                np.clip(self.heading_control_stiffness * heading_error, self.ang_vel_z[0], self.ang_vel_z[1])
            )
        if self.is_standing_env:
            self.vel_command_b[:] = 0.0


class UniformPose2dCommand(CommandGenerator):
    """2D pose command ``[x_b, y_b, z_b, heading_err]`` (UniformPose2dCommand in IsaacLab).

    Goals are sampled relative to the env origin; z is the default root height. The
    body-frame command is re-targeted every step through the yaw-only root rotation.
    """

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        pos_x: tuple[float, float],
        pos_y: tuple[float, float],
        heading: tuple[float, float],
        simple_heading: bool = False,
        default_root_height: float = 0.0,
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        super().__init__(resampling_time_range, strict=strict, fixed_command=fixed_command)
        self.pos_x = _range(pos_x)
        self.pos_y = _range(pos_y)
        self.heading = _range(heading)
        self.simple_heading = bool(simple_heading)
        self.default_root_height = float(default_root_height)

        self.pos_command_w = np.zeros(3, dtype=np.float64)
        self.heading_command_w = 0.0
        self.pos_command_b = np.zeros(3, dtype=np.float64)
        self.heading_command_b = 0.0

    @property
    def command(self) -> np.ndarray:
        return np.concatenate([self.pos_command_b, [self.heading_command_b]])

    def reset(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        # The body-frame command is written only by _update_command (per step) —
        # neither resampling nor pinning touches it. Clear it so a reused env's
        # first observation matches a fresh Isaac env's zeros instead of serving
        # the previous episode's goal; settled-dump continuation re-targets it
        # via retarget().
        self.pos_command_b = np.zeros(3, dtype=np.float64)
        self.heading_command_b = 0.0
        super().reset(rng, robot_state)

    def _pin_command(self, fixed: np.ndarray) -> None:
        # Strict mode pins the world-frame goal [x_w, y_w, z_w, heading_w]; the body-frame
        # command stays state-dependent by construction.
        if fixed.shape != (4,):
            raise ValueError(f"pose fixed_command must have shape (4,), got {fixed.shape}")
        self.pos_command_w = fixed[:3].copy()
        self.heading_command_w = float(fixed[3])

    def _sample_goal_pos_w(self, rng: np.random.Generator, robot_state: RobotState) -> np.ndarray:
        pos = robot_state.env_origin_w.copy()
        pos[0] += rng.uniform(*self.pos_x)
        pos[1] += rng.uniform(*self.pos_y)
        pos[2] += self.default_root_height
        return pos

    def _resample_command(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.pos_command_w = self._sample_goal_pos_w(rng, robot_state)
        if self.simple_heading:
            # Point towards the target, picking the direction (or its flip) closest to the
            # current heading to avoid the -pi/pi discontinuity.
            target_vec = self.pos_command_w - robot_state.root_pos_w
            target_direction = math.atan2(target_vec[1], target_vec[0])
            flipped_target_direction = wrap_to_pi(target_direction + math.pi)
            curr_to_target = abs(wrap_to_pi(target_direction - robot_state.heading_w))
            curr_to_flipped_target = abs(wrap_to_pi(flipped_target_direction - robot_state.heading_w))
            if curr_to_target < curr_to_flipped_target:
                self.heading_command_w = float(target_direction)
            else:
                self.heading_command_w = float(flipped_target_direction)
        else:
            self.heading_command_w = float(rng.uniform(*self.heading))

    def _update_command(self, robot_state: RobotState) -> None:
        target_vec = self.pos_command_w - robot_state.root_pos_w
        self.pos_command_b = quat_apply_inverse(yaw_quat(robot_state.root_quat_w_wxyz), target_vec)
        self.heading_command_b = float(wrap_to_pi(self.heading_command_w - robot_state.heading_w))


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Pose command whose goal position comes from a terrain flat-patch sampler.

    ``patch_sampler(rng)`` returns a world-frame xy (2,) or xyz (3,) patch position; the
    default root height is added to z. Keeps the terrain module decoupled.
    """

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        patch_sampler: Callable[[np.random.Generator], np.ndarray],
        heading: tuple[float, float],
        simple_heading: bool = False,
        default_root_height: float = 0.0,
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        super().__init__(
            resampling_time_range,
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            heading=heading,
            simple_heading=simple_heading,
            default_root_height=default_root_height,
            strict=strict,
            fixed_command=fixed_command,
        )
        self.patch_sampler = patch_sampler

    def _sample_goal_pos_w(self, rng: np.random.Generator, robot_state: RobotState) -> np.ndarray:
        patch = np.asarray(self.patch_sampler(rng), dtype=np.float64)
        pos = np.zeros(3, dtype=np.float64)
        pos[: patch.shape[0]] = patch
        pos[2] += self.default_root_height
        return pos


class UniformScalarCommand(CommandGenerator):
    """Scalar command sampled uniformly from ``range`` (shape ``(1,)``).

    Covers contact_lab's ``VelocityLimitCommand`` (max base xy speed, m/s) and
    ``ContactSafetyThresholdCommand`` (max lateral contact force, N): both resample a
    single uniform scalar and have no per-step post-processing.
    """

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        range: tuple[float, float],
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        super().__init__(resampling_time_range, strict=strict, fixed_command=fixed_command)
        self.range = _range(range)
        self.value = np.ones(1, dtype=np.float64)

    @property
    def command(self) -> np.ndarray:
        return self.value

    def _pin_command(self, fixed: np.ndarray) -> None:
        if fixed.shape != (1,):
            raise ValueError(f"scalar fixed_command must have shape (1,), got {fixed.shape}")
        self.value = fixed.copy()

    def _resample_command(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.value[0] = rng.uniform(*self.range)

    def _update_command(self, robot_state: RobotState) -> None:
        pass


class FrequencyCommand(CommandGenerator):
    """Gait frequency and its integrated phase, ``[frequency, phase]`` (shape ``(2,)``).

    Port of contact_lab's ``FrequencyCommand``: a resample draws the frequency (Hz) from
    ``range`` and a phase uniformly from ``[0, 1)``; every step the phase then advances
    by ``frequency * dt`` and wraps to ``[0, 1)`` (after any resample, like the command
    manager's ``compute``). A pinned command fixes the frequency, and optionally the
    initial phase (``(1,)``: ``[frequency]`` starting at phase 0; ``(2,)``:
    ``[frequency, phase0]``); the phase keeps integrating.
    """

    def __init__(
        self,
        resampling_time_range: tuple[float, float],
        *,
        range: tuple[float, float],
        strict: bool = False,
        fixed_command: Any = None,
    ) -> None:
        if fixed_command is not None and np.asarray(fixed_command).shape == (1,):
            fixed_command = np.array([np.asarray(fixed_command, dtype=np.float64)[0], 0.0])
        super().__init__(resampling_time_range, strict=strict, fixed_command=fixed_command)
        self.range = _range(range)
        if min(self.range) < 0.0:
            raise ValueError(f"FrequencyCommand range must be non-negative, got {self.range}")
        self.value = np.array([1.0, 1.0], dtype=np.float64)

    @property
    def command(self) -> np.ndarray:
        return self.value

    def step(self, dt: float, robot_state: RobotState, rng: np.random.Generator) -> None:
        super().step(dt, robot_state, rng)
        self.value[1] = (self.value[1] + self.value[0] * dt) % 1.0

    def _pin_command(self, fixed: np.ndarray) -> None:
        if fixed.shape != (2,):
            raise ValueError(f"frequency fixed_command must have shape (1,) or (2,), got {fixed.shape}")
        self.value = fixed.copy()
        self.value[1] %= 1.0

    def _resample_command(self, rng: np.random.Generator, robot_state: RobotState) -> None:
        self.value[0] = rng.uniform(*self.range)
        self.value[1] = rng.uniform(0.0, 1.0)

    def _update_command(self, robot_state: RobotState) -> None:
        pass  # the phase integrates in step(), which knows dt; retarget() must not advance it


# ---------------------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------------------

# Single registry: command type -> command vector dimension. ``KNOWN_COMMAND_TYPES`` and
# ``command_dim`` derive from it so the factory below and the dimension table cannot
# silently drift apart (a type added here without a builder branch raises loudly in
# :func:`build_command`, and vice versa).
COMMAND_DIMS: dict[str, int] = {
    "UniformVelocityCommand": 3,
    "UniformPose2dCommand": 4,
    "TerrainBasedPose2dCommand": 4,
    "VelocityLimitCommand": 1,
    "ContactSafetyThresholdCommand": 1,
    "FrequencyCommand": 2,
}

KNOWN_COMMAND_TYPES = tuple(COMMAND_DIMS)


def command_dim(command_type: str) -> int:
    """Command vector dimension for a known command type."""
    if command_type not in COMMAND_DIMS:
        raise ValueError(f"unknown command type '{command_type}' (known: {sorted(COMMAND_DIMS)})")
    return COMMAND_DIMS[command_type]


def _require_param(params: dict[str, Any], key: str, ir: CommandIR) -> Any:
    try:
        return params[key]
    except KeyError as err:
        raise ValueError(
            f"command term '{ir.name}' ({ir.type}) has no '{key}' param — unsupported command cfg layout"
        ) from err


def build_command(
    ir: CommandIR,
    *,
    default_root_height: float = 0.0,
    patch_sampler: Callable[[np.random.Generator], np.ndarray] | None = None,
    strict: bool = False,
    fixed_command: Any = None,
) -> CommandGenerator:
    """Build a command generator from a :class:`CommandIR`.

    Args:
        ir: Parsed command term.
        default_root_height: Env-local default root z (pose commands add it to goal z).
        patch_sampler: Flat-patch sampler, required for ``TerrainBasedPose2dCommand``.
        strict: Require a ``fixed_command`` (sim2sim eval).
        fixed_command: Pinned command (in or out of strict mode); velocity ``(3,)``,
            pose world goal ``(4,)``, scalar ``(1,)``, gait frequency ``(1,)`` or
            ``(2,)`` (with the initial phase). ``None`` samples normally.
    """
    p = ir.params
    common: dict[str, Any] = {"strict": strict, "fixed_command": fixed_command}
    resampling = _range(_require_param(p, "resampling_time_range", ir))

    if ir.type == "UniformVelocityCommand":
        ranges = _require_param(p, "ranges", ir)
        heading = ranges.get("heading")
        return UniformVelocityCommand(
            resampling,
            lin_vel_x=_range(_require_param(ranges, "lin_vel_x", ir)),
            lin_vel_y=_range(_require_param(ranges, "lin_vel_y", ir)),
            ang_vel_z=_range(_require_param(ranges, "ang_vel_z", ir)),
            heading=None if heading is None else _range(heading),
            heading_command=bool(p.get("heading_command", False)),
            heading_control_stiffness=float(p.get("heading_control_stiffness", 1.0)),
            rel_standing_envs=float(p.get("rel_standing_envs", 0.0)),
            rel_heading_envs=float(p.get("rel_heading_envs", 1.0)),
            **common,
        )
    if ir.type == "UniformPose2dCommand":
        ranges = _require_param(p, "ranges", ir)
        return UniformPose2dCommand(
            resampling,
            pos_x=_range(_require_param(ranges, "pos_x", ir)),
            pos_y=_range(_require_param(ranges, "pos_y", ir)),
            heading=_range(_require_param(ranges, "heading", ir)),
            simple_heading=bool(p.get("simple_heading", False)),
            default_root_height=default_root_height,
            **common,
        )
    if ir.type == "TerrainBasedPose2dCommand":
        if patch_sampler is None:
            raise ValueError("TerrainBasedPose2dCommand requires a patch_sampler")
        ranges = _require_param(p, "ranges", ir)
        return TerrainBasedPose2dCommand(
            resampling,
            patch_sampler=patch_sampler,
            heading=_range(_require_param(ranges, "heading", ir)),
            simple_heading=bool(p.get("simple_heading", False)),
            default_root_height=default_root_height,
            **common,
        )
    if ir.type in ("VelocityLimitCommand", "ContactSafetyThresholdCommand"):
        return UniformScalarCommand(resampling, range=_range(_require_param(p, "range", ir)), **common)
    if ir.type == "FrequencyCommand":
        return FrequencyCommand(resampling, range=_range(_require_param(p, "range", ir)), **common)
    raise ValueError(f"unsupported command type '{ir.type}' (known: {list(KNOWN_COMMAND_TYPES)})")
