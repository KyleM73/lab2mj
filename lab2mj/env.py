"""Single-env MuJoCo runtime for a converted IsaacLab bundle.

Composes the compiled scene model with the manifest-driven actuator set, obs
pipeline, command generators, DR events, and termination set, and mirrors
``ManagerBasedRLEnv.step`` ordering exactly: process action -> ``decimation``
physics steps (each integrated as ``physics_substeps`` MuJoCo steps at
``physics_dt / physics_substeps``; explicit actuator groups — PD models and
actuator nets alike — get their torque computed once per physics step and held
across the substeps, while ``implicit_pd`` groups have their drive ctrl
recomputed each substep against the updated joint state, mirroring PhysX's
per-iteration implicit drive) -> terminations -> auto-reset (non-strict) ->
commands -> interval events -> observations. Observations are computed after
resets, so a freshly reset episode observes its post-reset state (IsaacLab
semantics).

``torch`` is imported lazily and only when a policy is actually loaded or the
bundle has an actuator-net group (whose TorchScript net needs ``torch.jit``).

Frame conventions (repo-wide): ``_b`` body frame, ``_com`` CoM, ``_w`` world.
MuJoCo free-joint ``qvel`` stores the **world-frame velocity of the body-frame
origin** (linear) and the **body-local angular velocity**; Isaac's root
velocities are CoM/world quantities, so every write goes through
:func:`free_joint_qvel` / :func:`lab2mj.events.write_root_state`.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from lab2mj import bundle
from lab2mj.actuators import ActuatorSet, resolve_matching_names, resolve_matching_names_values
from lab2mj.commands import CommandGenerator, RobotState, build_command, command_dim
from lab2mj.env_yaml import class_name
from lab2mj.events import EventSet, RobotMap, read_root_com_vel_w
from lab2mj.heightscan import HeightScanProvider, height_scan_extras_dims
from lab2mj.ir import (
    ActuatorGroupIR,
    CommandIR,
    EventIR,
    HeightScannerIR,
    ObsGroupIR,
    TerminationIR,
    TerrainIR,
)
from lab2mj.obs import ObsContext, ObsPipeline
from lab2mj.quat import heading_w_from_quat, quat_to_rotmat
from lab2mj.stiction import JointStiction
from lab2mj.terminations import TerminationSet, map_size_from_generator

# PreTrainedPolicyAction.action_dim is fixed in IsaacLab: the high-level action is
# the [vx, vy, wz] velocity command fed into the low-level policy's obs group.
PRE_TRAINED_POLICY_ACTION_DIM = 3

# Name of the synthetic command slot that serves the held high-level action to the
# low-level obs group's remapped ``velocity_commands`` term. Must not collide with a
# real command term of the high-level env.
HELD_ACTION_COMMAND = "__pre_trained_policy_raw_action__"

_XML_RTOL = bundle.XML_RTOL


def stamp_exact_inertia(model: mujoco.MjModel, data: mujoco.MjData, manifest: dict[str, Any]) -> None:
    """Restore exact principal inertias the compiler could not accept.

    ``manifest["robot"]["body_inertia_exact"]`` carries the exact moments of links
    whose authored inertia violates the rigid-body triangle inequality: PhysX
    simulates the authored tensor, MuJoCo's compiler rejects it, so scene.xml holds
    a boundary-projected value. The runtime integrates the exact tensor fine —
    stamp it and refresh the derived model constants.
    """
    exact = manifest["robot"].get("body_inertia_exact") or {}
    if not exact:
        return
    for body_name, moments in exact.items():
        model.body_inertia[model.body(body_name).id] = np.asarray(moments, dtype=np.float64)
    mujoco.mj_setConst(model, data)


def free_joint_qvel(
    root_quat_wxyz: np.ndarray, root_link_lin_vel_w: np.ndarray, root_ang_vel_w: np.ndarray
) -> np.ndarray:
    """MuJoCo free-joint qvel (6,) from a world-frame root **link-origin** state.

    The linear part of a free joint's qvel is the world-frame velocity of the
    body-frame origin, so ``root_link_lin_vel_w`` passes through unchanged (do
    NOT feed the CoM velocity here); the angular part is body-local:
    ``ang_vel_b = R_wb^T @ ang_vel_w``.
    """
    rot_wb = quat_to_rotmat(np.asarray(root_quat_wxyz, dtype=np.float64))
    out = np.empty(6, dtype=np.float64)
    out[0:3] = np.asarray(root_link_lin_vel_w, dtype=np.float64)
    out[3:6] = rot_wb.T @ np.asarray(root_ang_vel_w, dtype=np.float64)
    return out


def dump_actuator_lags(dump: Any) -> dict[str, int] | None:
    """Per-group DelayedPD lags recorded in a reference dump.

    Reads the ``init_actuator_lag_<group>`` keys (from either an npz or a plain
    dict dump); ``None`` when the dump predates them, in which case
    :meth:`MjEnv.reset_from_state` keeps the strict pinned-zero lags.
    """
    prefix = "init_actuator_lag_"
    keys = dump.files if hasattr(dump, "files") else dump.keys()
    lags = {key[len(prefix) :]: int(dump[key]) for key in keys if key.startswith(prefix)}
    return lags or None


def _load_jit_policy(path: str | Path) -> Any:
    import torch

    policy = torch.jit.load(str(path), map_location="cpu")
    policy.eval()
    return policy


def _jit_policy_action(policy: Any, obs: np.ndarray) -> np.ndarray:
    """One batched TorchScript forward: float32 obs in, float64 raw action out.

    The ``detach`` matters: frozen TorchScript modules have produced
    grad-requiring outputs even with gradients disabled.
    """
    import torch

    with torch.no_grad():
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).reshape(1, -1)
        action_t = policy(obs_t)
    return action_t[0].detach().cpu().numpy().astype(np.float64)


class ActionProcessor:
    """Isaac ``JointPositionAction`` processing: ``target = scale * raw + offset`` then clip.

    Built from the manifest's resolved action arrays; produces full joint
    position targets in Isaac order (non-action joints hold their default).
    """

    def __init__(
        self,
        *,
        num_joints: int,
        joint_ids_isaac: np.ndarray,
        scale: np.ndarray,
        offset: np.ndarray,
        clip: np.ndarray | None,
        default_joint_pos_isaac: np.ndarray,
    ) -> None:
        self.joint_ids_isaac = np.asarray(joint_ids_isaac, dtype=np.intp)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.offset = np.asarray(offset, dtype=np.float64)
        self.clip = None if clip is None else np.asarray(clip, dtype=np.float64)
        self.action_dim = int(self.joint_ids_isaac.shape[0])
        self._default_targets = np.asarray(default_joint_pos_isaac, dtype=np.float64).copy()
        if self._default_targets.shape != (num_joints,):
            raise ValueError(f"default_joint_pos_isaac must have shape ({num_joints},)")
        for name, arr in (("scale", self.scale), ("offset", self.offset)):
            if arr.shape != (self.action_dim,):
                raise ValueError(f"action {name} must have shape ({self.action_dim},), got {arr.shape}")
        if self.clip is not None and self.clip.shape != (self.action_dim, 2):
            raise ValueError(f"action clip must have shape ({self.action_dim}, 2), got {self.clip.shape}")

    @classmethod
    def from_manifest(cls, action: dict[str, Any], default_joint_pos_isaac: np.ndarray) -> "ActionProcessor":
        return cls(
            num_joints=len(default_joint_pos_isaac),
            joint_ids_isaac=np.asarray(action["joint_ids_isaac"], dtype=np.intp),
            scale=np.asarray(action["scale"], dtype=np.float64),
            offset=np.asarray(action["offset"], dtype=np.float64),
            clip=None if action["clip"] is None else np.asarray(action["clip"], dtype=np.float64),
            default_joint_pos_isaac=default_joint_pos_isaac,
        )

    def processed(self, action_raw: np.ndarray) -> np.ndarray:
        """Processed action (A,): affine transform of the raw action, then clip."""
        raw = np.asarray(action_raw, dtype=np.float64).reshape(self.action_dim)
        out = raw * self.scale + self.offset
        if self.clip is not None:
            np.clip(out, self.clip[:, 0], self.clip[:, 1], out=out)
        return out

    def targets_isaac(self, action_raw: np.ndarray) -> np.ndarray:
        """Full joint position targets (J,) in Isaac order."""
        targets = self._default_targets.copy()
        targets[self.joint_ids_isaac] = self.processed(action_raw)
        return targets


def resolve_action_arrays(
    action_ir: dict[str, Any], isaac_joint_order: list[str], default_joint_pos_isaac: np.ndarray
) -> dict[str, Any]:
    """Resolve an ``ActionIR`` dict onto per-joint arrays for the manifest.

    Returns ``{"ir", "dim", "joint_ids_isaac", "scale", "offset", "clip"}`` with
    IsaacLab ``JointAction`` semantics: scale defaults to 1 (regex dict fills
    matched joints), offset defaults to 0, ``use_default_offset`` replaces the
    offset with the default joint positions, clip resolves regex keys onto
    ``(-inf, inf)`` defaults.
    """
    action_class = class_name(str(action_ir.get("func"))).removesuffix("Cfg")
    if action_class == "PreTrainedPolicyAction":
        low_level = action_ir.get("low_level_action")
        if low_level is None or action_ir.get("low_level_obs") is None:
            raise ValueError("PreTrainedPolicyAction term has no embedded low_level_action/low_level_obs cfg")
        if action_ir.get("low_level_decimation") is None:
            raise ValueError("PreTrainedPolicyAction term has no low_level_decimation")
        return {
            "ir": dict(action_ir),
            "dim": PRE_TRAINED_POLICY_ACTION_DIM,
            "low_level": resolve_action_arrays(low_level, isaac_joint_order, default_joint_pos_isaac),
            "low_level_decimation": int(action_ir["low_level_decimation"]),
        }
    # The runtime executes exactly JointPositionAction semantics (affine transform fed
    # to a position PD); any other JointAction class would be silently misinterpreted.
    if action_class != "JointPositionAction":
        raise ValueError(
            f"unsupported action term class {action_ir.get('func')!r}: only JointPositionAction converts "
            "(the MuJoCo runtime feeds scale * action + offset to a position PD)"
        )
    if action_ir.get("preserve_order"):
        raise ValueError("action terms with preserve_order=True are not supported")
    ids = resolve_matching_names(list(action_ir.get("joint_names_expr") or [".*"]), isaac_joint_order, "action joints")
    joint_names = [isaac_joint_order[i] for i in ids]
    dim = len(ids)

    def resolve(value: Any, default: float, what: str) -> np.ndarray:
        out = np.full(dim, default, dtype=np.float64)
        if isinstance(value, dict):
            indices, values = resolve_matching_names_values(value, joint_names, what=what)
            out[indices] = values
        elif isinstance(value, (int, float)):
            out[:] = float(value)
        else:
            raise TypeError(f"{what}: expected float or dict, got {type(value).__name__}")
        return out

    scale = resolve(action_ir.get("scale", 1.0), 1.0, "action scale")
    offset = resolve(action_ir.get("offset", 0.0), 0.0, "action offset")
    if action_ir.get("use_default_offset"):
        offset = np.asarray(default_joint_pos_isaac, dtype=np.float64)[ids]

    clip_cfg = action_ir.get("clip")
    clip: np.ndarray | None = None
    if clip_cfg is not None:
        if not isinstance(clip_cfg, dict):
            raise TypeError("action clip must be a regex dict of (lo, hi) pairs")
        clip = np.tile(np.array([-np.inf, np.inf]), (dim, 1))
        indices = resolve_matching_names([str(k) for k in clip_cfg], joint_names, "action clip")
        for idx in indices:
            for pattern, bounds in clip_cfg.items():
                if re.fullmatch(pattern, joint_names[idx]):
                    clip[idx] = (float(bounds[0]), float(bounds[1]))
                    break
    return {
        "ir": dict(action_ir),
        "dim": dim,
        "joint_ids_isaac": ids,
        "scale": scale,
        "offset": offset,
        "clip": clip,
    }


def remap_low_level_obs_group(group: ObsGroupIR) -> ObsGroupIR:
    """Mirror ``PreTrainedPolicyAction.__init__``'s low-level obs term remapping.

    IsaacLab replaces the funcs of the terms named exactly ``actions`` (-> the
    wrapped low-level action term's raw output) and ``velocity_commands`` (-> the
    held high-level action) and clears their params; noise/clip/scale/history stay.
    Both terms must exist (Isaac dereferences the attributes unconditionally).
    The returned copy uses the canonical ``last_action`` / ``generated_commands``
    evaluators, with the held action served under :data:`HELD_ACTION_COMMAND`.
    """
    names = {term.name for term in group.terms}
    for required in ("actions", "velocity_commands"):
        if required not in names:
            raise ValueError(
                f"low-level obs group '{group.name}' has no '{required}' term; PreTrainedPolicyAction "
                f"requires it (have {sorted(names)})"
            )
    terms = []
    for term in group.terms:
        if term.name == "actions":
            term = replace(term, func="isaaclab.envs.mdp.observations:last_action", params={})
        elif term.name == "velocity_commands":
            term = replace(
                term,
                func="isaaclab.envs.mdp.observations:generated_commands",
                params={"command_name": HELD_ACTION_COMMAND},
            )
        terms.append(term)
    return replace(group, terms=terms)


def build_low_level_obs_pipeline(
    action_manifest: dict[str, Any],
    *,
    num_joints: int,
    command_dims: dict[str, int],
    command_resample_time_s: dict[str, float],
    height_scanners: list[HeightScannerIR],
    enable_noise: bool | None = None,
    entity: str = "robot",
) -> tuple[ObsGroupIR, ObsPipeline]:
    """Remapped low-level obs group + pipeline for a PreTrainedPolicyAction manifest entry.

    ``command_dims`` are the HIGH-level env's command dims (other
    ``generated_commands`` terms of the low-level group resolve against them);
    the held-action slot is added on top. The pipeline's ``last_action`` dim is
    the LOW-level action dim.
    """
    group = remap_low_level_obs_group(ObsGroupIR.from_dict(action_manifest["ir"]["low_level_obs"]))
    if HELD_ACTION_COMMAND in command_dims:
        raise ValueError(f"command term name {HELD_ACTION_COMMAND!r} collides with the held-action slot")
    pipeline = ObsPipeline(
        group,
        num_joints=num_joints,
        action_dim=int(action_manifest["low_level"]["dim"]),
        command_dims={**command_dims, HELD_ACTION_COMMAND: int(action_manifest["dim"])},
        command_resample_time_s=command_resample_time_s,
        extras_dims=height_scan_extras_dims(group, height_scanners),
        enable_noise=enable_noise,
        entity=entity,
    )
    return group, pipeline


def assert_actuator_order(model: mujoco.MjModel, manifest: dict[str, Any]) -> None:
    """Assert ctrl slot ``k`` drives ``mj_joint_order[k]`` (required by ``ctrl = tau_isaac[isaac_to_mj]``)."""
    mj_order = list(manifest["robot"]["mj_joint_order"])
    if model.nu != len(mj_order):
        raise ValueError(f"model has {model.nu} actuators, expected one per joint ({len(mj_order)})")
    for k, expected in enumerate(mj_order):
        joint_id = int(model.actuator_trnid[k, 0])
        actual = model.joint(joint_id).name
        if actual != expected:
            raise ValueError(f"ctrl slot {k} drives joint '{actual}', expected '{expected}' (mj_joint_order)")


def build_robot_map(model: mujoco.MjModel, manifest: dict[str, Any]) -> RobotMap:
    """Build the runtime :class:`~lab2mj.events.RobotMap` from a manifest."""
    robot = manifest["robot"]
    init = manifest["init"]
    isaac_joint_order: list[str] = list(robot["isaac_joint_order"])

    free_jid = -1
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            free_jid = jid
            break
    if free_jid < 0:
        raise ValueError("model has no free joint (expected a floating-base robot)")

    body_ids: dict[str, list[int]] = {}
    for isaac_name, mj_name in robot["body_map"].items():
        body_ids[isaac_name] = [int(model.body(mj_name).id)]
    geom_ids: dict[str, list[int]] = {}
    for isaac_name, geom_names in robot["geom_map"].items():
        geom_ids[isaac_name] = [int(model.geom(g).id) for g in geom_names]

    qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in isaac_joint_order], dtype=np.int64)
    dof_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in isaac_joint_order], dtype=np.int64)

    default_root_pose_env = np.concatenate(
        [np.asarray(init["root_pos_env"], dtype=np.float64), np.asarray(init["root_quat_wxyz"], dtype=np.float64)]
    )
    default_root_vel_w = np.concatenate(
        [np.asarray(init["root_lin_vel_env"], dtype=np.float64), np.asarray(init["root_ang_vel_env"], dtype=np.float64)]
    )

    return RobotMap(
        body_ids=body_ids,
        geom_ids=geom_ids,
        root_body_id=int(model.jnt_bodyid[free_jid]),
        root_qpos_adr=int(model.jnt_qposadr[free_jid]),
        root_dof_adr=int(model.jnt_dofadr[free_jid]),
        joint_names=isaac_joint_order,
        qpos_adr=qpos_adr,
        dof_adr=dof_adr,
        default_joint_pos=np.asarray(robot["default_joint_pos_isaac"], dtype=np.float64),
        default_joint_vel=np.asarray(robot["default_joint_vel_isaac"], dtype=np.float64),
        joint_pos_limits=np.asarray(robot["joint_pos_limits_isaac"], dtype=np.float64),
        joint_vel_limits=np.asarray(robot["joint_vel_limits_isaac"], dtype=np.float64),
        default_root_pose_env=default_root_pose_env,
        default_root_vel_w=default_root_vel_w,
        env_origin_w=np.asarray(init["env_origin_w"], dtype=np.float64),
        default_body_mass=model.body_mass.copy(),
        default_body_inertia=model.body_inertia.copy(),
        default_body_ipos=model.body_ipos.copy(),
    )


class PreTrainedPolicyRuntime:
    """Runtime mirror of IsaacLab's ``PreTrainedPolicyAction`` (navigation tasks).

    The high-level action — the velocity command for the frozen low-level
    policy — is held for the whole policy step. Every ``low_level_decimation``
    physics steps (before that physics step integrates, matching
    ``ActionManager.apply_action`` ordering) the low-level obs group is
    recomputed with the held action in its ``velocity_commands`` slot and the
    low-level raw output in its ``actions`` slot, the TorchScript policy runs,
    and the wrapped ``JointPositionAction`` turns its output into joint
    position targets that hold until the next low-level fire.

    State semantics mirror Isaac's ``_reset_idx``: the fire counter and the
    low-level obs pipeline state survive mid-rollout auto-resets
    (``PreTrainedPolicyAction`` never resets its counter, and its standalone
    low-level ``ObservationManager`` is not registered for resets). Explicit
    resets restore the constructor state via :meth:`reset_cold` instead —
    reference dumps are recorded from freshly created Isaac envs, and gate
    metrics on a shared env must not depend on which gates ran before. Isaac's
    remapped ``actions`` obs term zeroes the low-level action buffer whenever
    ``episode_length_buf == 0``, and that buffer only increments after the
    full decimation loop — so every low-level fire of an episode's first
    policy step observes zeros in the ``actions`` slot, while the targets
    always come from the fresh policy output.
    """

    def __init__(self, env: "MjEnv") -> None:
        action = env.manifest["action"]
        self.action_dim = int(action["dim"])
        self.low_level_decimation = int(action["low_level_decimation"])
        if self.low_level_decimation < 1:
            raise ValueError(f"low_level_decimation must be >= 1, got {self.low_level_decimation}")
        default_joint_pos = np.asarray(env.manifest["robot"]["default_joint_pos_isaac"], dtype=np.float64)
        self.low_level_action = ActionProcessor.from_manifest(action["low_level"], default_joint_pos)

        rel_path = (action["ir"] or {}).get("policy_bundle_path")
        if not rel_path:
            raise ValueError("manifest action has no policy_bundle_path; re-convert the bundle")
        self._policy_path = env.bundle_dir / rel_path
        if not self._policy_path.is_file():
            raise FileNotFoundError(f"low-level policy {self._policy_path} is missing from the bundle")
        self._policy = None

        command_dims = {c["name"]: command_dim(c["type"]) for c in env.manifest["commands"]}
        resample_s = {
            c["name"]: float(c["params"]["resampling_time_range"][1])
            for c in env.manifest["commands"]
            if "resampling_time_range" in c["params"]
        }
        group, self.pipeline = build_low_level_obs_pipeline(
            action,
            num_joints=env.num_joints,
            command_dims=command_dims,
            command_resample_time_s=resample_s,
            height_scanners=env._height_scanner_irs(),
            enable_noise=False if env.strict else None,
            entity=env._robot_entity,
        )
        # Low-level sensors refresh at the low-level cadence, not the policy step.
        self._height_scan_terms = env._build_height_scan_terms(
            group, max_period=self.low_level_decimation * env.physics_dt
        )

        self._default_joint_pos = default_joint_pos
        self._raw = np.zeros(self.action_dim, dtype=np.float64)
        self._ll_last_raw = np.zeros(self.low_level_action.action_dim, dtype=np.float32)
        self._targets_isaac = default_joint_pos.copy()
        self._counter = 0

    def reset_cold(self) -> None:
        """Restore the just-constructed state (explicit resets only).

        Reference dumps are recorded from freshly created Isaac envs, so an
        explicit reset serves the constructor state: fire counter at 0, cold
        obs history, zeroed action buffers, default-pose targets. Mid-rollout
        auto-resets deliberately do NOT come through here — that state
        survives Isaac's ``_reset_idx`` (see class doc).
        """
        self._raw[:] = 0.0
        self._ll_last_raw = np.zeros_like(self._ll_last_raw)
        self._targets_isaac = self._default_joint_pos.copy()
        self._counter = 0
        self.pipeline.reset()

    def processed(self, action_raw: np.ndarray) -> np.ndarray:
        """``PreTrainedPolicyAction.processed_actions`` is the raw high-level action."""
        return np.asarray(action_raw, dtype=np.float64).reshape(self.action_dim).copy()

    def process(self, action_raw: np.ndarray) -> None:
        """Store the high-level action, held for the whole policy step."""
        self._raw[:] = np.asarray(action_raw, dtype=np.float64).reshape(self.action_dim)

    def seed_last_raw(self, low_level_last_raw: np.ndarray) -> None:
        """Seed the low-level action buffer from a settled dump's ``init_low_level_action``.

        Layered on top of :meth:`reset_cold` by ``_finish_reset``; height-scan
        providers are owned and reset by the env, and after a plain reset every
        fire of the first policy step serves zeros in the ``actions`` slot via
        the ``episode_length_buf == 0`` remap in :meth:`physics_step_targets`.
        """
        self._ll_last_raw = np.asarray(low_level_last_raw, dtype=np.float32).reshape(self._ll_last_raw.shape)

    def physics_step_targets(self, env: "MjEnv") -> np.ndarray:
        """Joint position targets (J, Isaac order) for the physics step about to integrate.

        Mirrors ``PreTrainedPolicyAction.apply_actions``: the low-level obs +
        policy re-evaluate when ``counter % low_level_decimation == 0``, the
        counter zeroes on fire and increments every call.
        """
        if self._counter % self.low_level_decimation == 0:
            # Fresh kinematics for the low-level obs read (Isaac's lazily refreshed
            # articulation buffers observe the state after the previous physics step).
            mujoco.mj_forward(env.model, env.data)
            # Isaac's remapped ``actions`` term zeroes the low-level action buffer
            # while episode_length_buf == 0, which increments only after the full
            # decimation loop — every fire of the episode's first policy step
            # observes zeros (env._episode_step mirrors episode_length_buf).
            ll_obs_last_raw = np.zeros_like(self._ll_last_raw) if env._episode_step == 0 else self._ll_last_raw
            ctx = env._make_obs_context(
                last_action_raw=ll_obs_last_raw,
                height_scan_terms=self._height_scan_terms,
                extra_commands={HELD_ACTION_COMMAND: self._raw},
            )
            obs = self.pipeline.compute(ctx, env.rng)
            ll_raw = self._policy_action(obs)
            self._ll_last_raw = ll_raw.astype(np.float32)
            self._targets_isaac = self.low_level_action.targets_isaac(ll_raw)
            self._counter = 0
        self._counter += 1
        return self._targets_isaac

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        """Low-level policy output (raw, pre scale/offset) for one observation."""
        if self._policy is None:
            self._policy = _load_jit_policy(self._policy_path)
        return _jit_policy_action(self._policy, obs)


class MjEnv:
    """Raw-MuJoCo runtime for a converted bundle.

    Args:
        bundle_dir: Bundle directory produced by ``uv run lab2mj-convert``.
        policy_path: Optional TorchScript policy for :meth:`run_policy` /
            :meth:`policy_action` (loaded lazily on first use, CPU, eval mode).
        strict: Deterministic sim2sim mode: no obs noise, no DR events, no
            command resampling (pinned to ``command`` or zeros), actuator delay
            lag pinned to 0 (a dump's recorded ``init_actuator_lag_<group>``
            keys override it via :meth:`reset_from_state`), and no auto-reset
            on termination.
        seed: RNG seed; defaults to the manifest's recorded env seed.
        command: Fixed command, honored in and out of strict mode. A ``(3,)``
            array pins every velocity command term; a dict ``{term_name:
            vector}`` pins terms by name. Unpinned terms sample normally
            (non-strict) or pin to zeros (strict).
    """

    def __init__(
        self,
        bundle_dir: str | Path,
        policy_path: str | Path | None = None,
        *,
        strict: bool = False,
        seed: int | None = None,
        command: Any = None,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.manifest = bundle.read_manifest(self.bundle_dir)
        self.model = mujoco.MjModel.from_xml_path(str(bundle.scene_xml_path(self.bundle_dir)))
        self.data = mujoco.MjData(self.model)
        self.strict = bool(strict)
        self._policy_path = None if policy_path is None else Path(policy_path)
        self._policy = None

        robot = self.manifest["robot"]
        timing = self.manifest["timing"]
        self.isaac_joint_order: list[str] = list(robot["isaac_joint_order"])
        self.num_joints = len(self.isaac_joint_order)
        # Scene key of the robot articulation (SceneEntityCfg target); absent on
        # bundles converted before it was recorded, where the key was 'robot'.
        self._robot_entity = str(robot.get("entity") or "robot")
        self._isaac_to_mj = np.asarray(robot["isaac_to_mj"], dtype=np.intp)
        self._mj_to_isaac = np.asarray(robot["mj_to_isaac"], dtype=np.intp)
        # Torques are written positionally (data.ctrl = ctrl_isaac[isaac_to_mj]) —
        # the one model/manifest contract with no name-resolved fallback. State
        # reads are name-resolved, so a regenerated scene.xml with reordered
        # actuators would pass every other check and silently scramble torques.
        assert_actuator_order(self.model, self.manifest)
        self.physics_dt = float(timing["physics_dt"])
        self.decimation = int(timing["decimation"])
        self.policy_dt = self.physics_dt * self.decimation
        self.episode_length_s = float(timing["episode_length_s"])
        # physics_substeps > 1 integrates each Isaac physics step as N MuJoCo steps at
        # dt/N (absent on pre-substep bundles, meaning 1). Torque and delay buffers stay
        # on the Isaac physics_dt boundary; only contact/integration runs finer.
        self.physics_substeps = int(timing.get("physics_substeps", 1))
        if self.physics_substeps < 1:
            raise ValueError(f"manifest physics_substeps must be >= 1, got {self.physics_substeps}")
        self.substep_dt = self.physics_dt / self.physics_substeps
        # MjSpec.to_xml() serializes floats to 6 significant digits, so scene.xml values
        # are checked at XML round-trip tolerance and then overwritten with the exact
        # manifest float64 values (the manifest is the source of truth).
        if abs(self.model.opt.timestep - self.substep_dt) > _XML_RTOL * self.substep_dt:
            raise ValueError(
                f"scene.xml timestep {self.model.opt.timestep} != manifest physics_dt / physics_substeps "
                f"({self.physics_dt} / {self.physics_substeps} = {self.substep_dt})"
            )
        self.model.opt.timestep = self.substep_dt
        gravity_w = np.asarray(self.manifest["sim"]["gravity_w"], dtype=np.float64)
        if not np.allclose(self.model.opt.gravity, gravity_w, rtol=_XML_RTOL, atol=1e-9):
            raise ValueError(
                f"scene.xml gravity {np.asarray(self.model.opt.gravity)} != manifest sim.gravity_w {gravity_w}; "
                "the bundle was not produced by lab2mj.convert or is stale"
            )
        self.model.opt.gravity[:] = gravity_w
        # Isaac normalizes the sim gravity into GRAVITY_VEC_W for projected-gravity terms.
        gravity_norm = float(np.linalg.norm(gravity_w))
        self._gravity_dir_w = gravity_w / gravity_norm if gravity_norm > 0.0 else np.array([0.0, 0.0, -1.0])

        stamp_exact_inertia(self.model, self.data, self.manifest)
        self.robot_map = build_robot_map(self.model, self.manifest)
        self._default_qpos = np.asarray(robot["default_qpos"], dtype=np.float64)
        self._default_qvel = np.asarray(robot["default_qvel"], dtype=np.float64)
        if self._default_qpos.shape != (self.model.nq,) or self._default_qvel.shape != (self.model.nv,):
            raise ValueError("manifest default qpos/qvel do not match the compiled model dimensions")

        seed_value = seed if seed is not None else self.manifest["init"].get("seed")
        self.rng = np.random.default_rng(0 if seed_value is None else int(seed_value))

        self.actuators = ActuatorSet.from_ir(
            [ActuatorGroupIR.from_dict(g) for g in self.manifest["actuators"]],
            self.isaac_joint_order,
            net_dir=self.bundle_dir,
        )
        # Fail fast if an actuator-net group's weights are missing from the bundle.
        self.actuators.load_networks()
        self._terrain_build_cache = None
        # One provider per scanner, shared across every consuming pipeline (Isaac
        # keeps one RayCaster per sensor cfg): drift and reset draws stay correlated.
        self._height_scan_providers: dict[str, HeightScanProvider] = {}
        if "low_level" in self.manifest["action"]:
            self._pre_trained: PreTrainedPolicyRuntime | None = PreTrainedPolicyRuntime(self)
            self._joint_action: ActionProcessor | None = None
            self.action: ActionProcessor | PreTrainedPolicyRuntime = self._pre_trained
        else:
            self._pre_trained = None
            self._joint_action = ActionProcessor.from_manifest(
                self.manifest["action"], np.asarray(robot["default_joint_pos_isaac"], dtype=np.float64)
            )
            self.action = self._joint_action
        self.action_dim = self.action.action_dim

        # PhysX static joint friction (breakaway/capture on stationary joints) has no
        # compiled-model counterpart; a per-step switch on the friction bound supplies it.
        self._stiction = JointStiction(
            self.model,
            self.robot_map.dof_adr,
            self.actuators.static_friction,
            self.actuators.dynamic_friction,
            self.physics_dt,
        )
        self._commands = self._build_commands(command)
        self._events = self._build_events()
        self._terminations = self._build_terminations()
        self._obs_pipeline = self._build_obs_pipeline()
        self._height_scan_terms = self._build_height_scan_terms(self._policy_group_ir(), max_period=self.policy_dt)
        self.obs_dim = self._obs_pipeline.obs_dim
        self._implicit_joint_ids, self._implicit_kd = self._check_implicit_damping()
        self._implicit_kp = self.actuators.kp[self._implicit_joint_ids]
        self._implicit_effort = self.actuators.effort_limit[self._implicit_joint_ids]
        self._implicit_qpos_adr = self.robot_map.qpos_adr[self._implicit_joint_ids]
        self._implicit_dof_adr = self.robot_map.dof_adr[self._implicit_joint_ids]
        # data.ctrl is written as ctrl_isaac[isaac_to_mj]; invert that gather to
        # address individual implicit_pd joints' ctrl slots during substeps.
        mj_slot_of_isaac = np.empty(self.num_joints, dtype=np.intp)
        mj_slot_of_isaac[self._isaac_to_mj] = np.arange(self.num_joints, dtype=np.intp)
        self._implicit_ctrl_adr = mj_slot_of_isaac[self._implicit_joint_ids]

        self._last_action_raw = np.zeros(self.action_dim, dtype=np.float32)
        self._applied_torque_isaac = np.zeros(self.num_joints, dtype=np.float64)
        self._last_obs: np.ndarray | None = None
        self._episode_step = 0
        self._startup_done = False

    """
    Construction helpers.
    """

    def _build_commands(self, command: Any) -> dict[str, CommandGenerator]:
        generators: dict[str, CommandGenerator] = {}
        fixed_by_name: dict[str, np.ndarray] = {}
        vel_fixed: np.ndarray | None = None
        if isinstance(command, dict):
            fixed_by_name = {k: np.asarray(v, dtype=np.float64) for k, v in command.items()}
            unknown = set(fixed_by_name) - {entry["name"] for entry in self.manifest["commands"]}
            if unknown:
                # A typo'd term name would otherwise be silently dropped and the real
                # term pinned to zeros in strict mode.
                raise ValueError(
                    f"fixed command names {sorted(unknown)} match no command term "
                    f"(have {sorted(entry['name'] for entry in self.manifest['commands'])})"
                )
        elif command is not None:
            vel_fixed = np.asarray(command, dtype=np.float64).reshape(3)
            if not any(entry["type"] == "UniformVelocityCommand" for entry in self.manifest["commands"]):
                # The array form pins velocity terms only; on a pose/scalar-command
                # bundle it would be silently dropped — and strict mode would pin the
                # real terms to zeros — while the caller believes it was applied.
                have = sorted((entry["name"], entry["type"]) for entry in self.manifest["commands"])
                raise ValueError(
                    f"array-form command pins UniformVelocityCommand terms, but this bundle has none "
                    f"(command terms: {have}); pass a dict {{term_name: value}} instead"
                )
        for entry in self.manifest["commands"]:
            ir = CommandIR.from_dict(entry)
            fixed = fixed_by_name.get(ir.name)
            if fixed is None and ir.type == "UniformVelocityCommand" and vel_fixed is not None:
                fixed = vel_fixed
            if fixed is None and self.strict:
                fixed = np.zeros(command_dim(ir.type), dtype=np.float64)
            generators[ir.name] = build_command(
                ir,
                default_root_height=float(self.manifest["init"]["root_pos_env"][2]),
                patch_sampler=self._make_patch_sampler() if ir.type == "TerrainBasedPose2dCommand" else None,
                strict=self.strict,
                fixed_command=fixed,
            )
        return generators

    def _terrain_build(self):
        """Deterministic terrain rebuild from the manifest, built once on demand.

        Same manifest seed and generator cfg the converter used, so the tiles
        and surfaces match ``scene.xml``. A measured bundle instead reloads the
        measured-terrain record (``terrain.measured.file``), so ``height_at``
        raycasts the exact Isaac terrain mesh the reference walked on.
        """
        if self._terrain_build_cache is None:
            from lab2mj.terrain import MeasuredTerrainData, build_terrain

            terrain_cfg = self.manifest["terrain"]
            terrain_ir = None if terrain_cfg is None else TerrainIR.from_dict(terrain_cfg)
            measured = None
            if terrain_cfg is not None and terrain_cfg.get("source") == "measured-from-dump":
                measured_file = self.bundle_dir / str(terrain_cfg["measured"]["file"])
                measured = MeasuredTerrainData.from_npz(np.load(measured_file))
                if measured is None:
                    raise ValueError(f"{measured_file} carries no measured-terrain record; the bundle is corrupt")
            seed = self.manifest["init"].get("seed")
            terrain_rng = np.random.default_rng(0 if seed is None else int(seed))
            self._terrain_build_cache = build_terrain(terrain_ir, rng=terrain_rng, num_envs=1, measured=measured)
        return self._terrain_build_cache

    def _make_patch_sampler(self):
        """Flat-patch sampler for ``TerrainBasedPose2dCommand``, lazy-built from the manifest.

        Rebuilds the terrain layout on the first draw and samples the ``target``
        patch set — the set IsaacLab's command term reads. Single-env
        approximation: IsaacLab draws from the patches of the env's current
        tile; here every configured tile's patches are pooled.
        """
        cache: dict[str, np.ndarray] = {}

        def sample(rng: np.random.Generator) -> np.ndarray:
            if "patches_w" not in cache:
                build = self._terrain_build()
                sampler = build.flat_patches.get("target")
                if sampler is None:
                    raise ValueError(
                        "TerrainBasedPose2dCommand needs a 'target' flat-patch set, but the terrain "
                        f"generator cfg defines none (have {sorted(build.flat_patches)})"
                    )
                cache["patches_w"] = sampler()
            patches_w = cache["patches_w"]
            return patches_w[int(rng.integers(0, patches_w.shape[0]))]

        return sample

    def _build_events(self) -> EventSet:
        terrain = self.manifest["terrain"] or {}
        material = terrain.get("physics_material") or {}
        return EventSet(
            [EventIR.from_dict(e) for e in self.manifest["events"]],
            terrain_static_friction=float(material.get("static_friction", 1.0)),
            friction_combine_mode=str(material.get("friction_combine_mode", "average")),
            strict=self.strict,
            entity=self._robot_entity,
        )

    def _build_terminations(self) -> TerminationSet:
        terrain = self.manifest["terrain"] or {}
        terrain_size_xy = None
        if terrain.get("terrain_type") == "generator" and terrain.get("generator") is not None:
            terrain_size_xy = map_size_from_generator(terrain["generator"])
        return TerminationSet(
            [TerminationIR.from_dict(t) for t in self.manifest["terminations"]],
            step_dt=self.policy_dt,
            episode_length_s=self.episode_length_s,
            body_map=self.robot_map.body_ids,
            terrain_size_xy=terrain_size_xy,
            contact_sensors=self.manifest["contact_sensors"],
            gravity_dir_w=self._gravity_dir_w,
            entity=self._robot_entity,
        )

    def _check_implicit_damping(self) -> tuple[np.ndarray, np.ndarray]:
        """Joints whose PD damping lives in the model (implicit_pd groups).

        The converter authors ``dof_damping = kd`` for these joints (implicit
        integration keeps PhysX-style PD stable; see the implicit_pd authoring
        in ``convert.convert_run``). The runtime adds ``+kd*qd`` back into
        ctrl so the effort clamp still acts on the full PD torque; the net
        applied torque matches Isaac's clamped PD.
        """
        ids: list[int] = []
        for group in self.actuators.groups:
            if group.model == "implicit_pd":
                ids.extend(int(i) for i in group.joint_ids)
        joint_ids = np.asarray(sorted(ids), dtype=np.intp)
        kd = self.actuators.kd[joint_ids]
        dof = self.robot_map.dof_adr[joint_ids]
        # Model damping = implicit_pd kd + viscous joint friction (both integrate
        # implicitly); only the kd part is drive and gets added back into ctrl.
        expected = kd + self.actuators.viscous_friction[joint_ids]
        if not np.allclose(self.model.dof_damping[dof], expected, rtol=_XML_RTOL, atol=1e-12):
            raise ValueError(
                "scene.xml dof_damping does not carry the implicit_pd group kd (+ viscous friction) values; "
                "the bundle was not produced by lab2mj.convert or is stale"
            )
        self.model.dof_damping[dof] = expected  # exact float64 over the 6-digit XML round-trip
        return joint_ids, kd

    def _policy_group_ir(self) -> ObsGroupIR:
        obs = self.manifest["obs"]
        return ObsGroupIR.from_dict(obs["groups"][obs["policy_group"]])

    def _height_scanner_irs(self) -> list[HeightScannerIR]:
        return [HeightScannerIR.from_dict(s) for s in self.manifest.get("height_scanners") or []]

    def _build_obs_pipeline(self) -> ObsPipeline:
        group = self._policy_group_ir()
        command_dims = {c["name"]: command_dim(c["type"]) for c in self.manifest["commands"]}
        resample_s = {
            c["name"]: float(c["params"]["resampling_time_range"][1])
            for c in self.manifest["commands"]
            if "resampling_time_range" in c["params"]
        }
        return ObsPipeline(
            group,
            num_joints=self.num_joints,
            action_dim=self.action_dim,
            command_dims=command_dims,
            command_resample_time_s=resample_s,
            extras_dims=height_scan_extras_dims(group, self._height_scanner_irs()),
            enable_noise=False if self.strict else None,
            entity=self._robot_entity,
        )

    def _build_height_scan_terms(
        self, group: ObsGroupIR, *, max_period: float
    ) -> list[tuple[str, HeightScanProvider, float, int]]:
        """(term_name, provider, term offset, attach mj body id) per height_scan term of ``group``.

        The provider casts against the bundle terrain's ``height_at`` lookup and
        reads the sensor pose from the attach body's MuJoCo frame each time the
        group's pipeline computes. ``max_period`` is that pipeline's refresh
        period (the policy step for the policy group, the low-level cadence for
        a PreTrainedPolicyAction group); a scanner with a slower cfg update
        period would observe stale data between refreshes, which the runtime
        does not model.
        """
        scanners = {scanner.name: scanner for scanner in self._height_scanner_irs()}
        entries: list[tuple[str, HeightScanProvider, float, int]] = []
        for term in group.terms:
            if class_name(term.func) != "height_scan":
                continue
            sensor_name = (term.params.get("sensor_cfg") or {}).get("name")
            scanner = scanners.get(sensor_name)
            if scanner is None:
                raise ValueError(
                    f"obs term '{term.name}' reads height scanner '{sensor_name}', which the manifest does not "
                    f"record (have {sorted(scanners)})"
                )
            if scanner.update_period > max_period + 1e-9:
                raise ValueError(
                    f"height scanner '{scanner.name}' has update_period {scanner.update_period} s > the consuming "
                    f"pipeline's refresh period {max_period} s; the runtime refreshes the scan on every pipeline "
                    "compute and cannot model a slower sensor"
                )
            attach = scanner.attach_body_name
            mj_body = self.manifest["robot"]["body_map"].get(attach)
            if mj_body is None:
                raise ValueError(f"height scanner '{scanner.name}' attaches to unknown body '{attach}'")
            if mj_body != attach:
                raise ValueError(
                    f"height scanner '{scanner.name}' attaches to '{attach}', which was welded into "
                    f"'{mj_body}'; the welded frame is not tracked at runtime"
                )
            build = self._terrain_build()
            offset = float(term.params.get("offset", 0.5))
            body_id = int(self.model.body(mj_body).id)
            provider = self._height_scan_providers.get(scanner.name)
            if provider is None:
                provider = HeightScanProvider(scanner, build.height_at)
                self._height_scan_providers[scanner.name] = provider
            entries.append((term.name, provider, offset, body_id))
        return entries

    """
    Reset / step.
    """

    def reset(self) -> np.ndarray:
        """Reset to the default episode start; returns the first observation.

        Strict mode restores the manifest's exact default ``qpos``/``qvel`` and
        pins the commands; non-strict mode additionally applies startup events
        (once) and the configured reset events, and resamples unpinned commands.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._default_qpos
        self.data.qvel[:] = self._default_qvel
        if not self.strict and not self._startup_done:
            self._events.apply_startup(self.model, self.data, self.robot_map, self.rng)
            self._startup_done = True
        if not self.strict:
            self._events.apply_reset(self.model, self.data, self.robot_map, self.rng)
        return self._finish_reset()

    def reset_from_state(
        self,
        root_pos_w: np.ndarray,
        root_quat_wxyz: np.ndarray,
        root_link_lin_vel_w: np.ndarray,
        root_ang_vel_w: np.ndarray,
        joint_pos_isaac: np.ndarray,
        joint_vel_isaac: np.ndarray,
        last_action_raw: np.ndarray | None = None,
        obs0: np.ndarray | None = None,
        episode_step: int = 0,
        low_level_last_raw: np.ndarray | None = None,
        actuator_lags: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Reset to an exact recorded state (reference-dump initialization).

        ``actuator_lags`` pins each named DelayedPD group to the lag Isaac
        sampled at the dumped env's reset (the dump's
        ``init_actuator_lag_<group>`` keys); without it, strict mode pins
        every lag to 0 and the replayed PD targets lead Isaac's by the
        sampled number of physics steps.

        ``root_link_lin_vel_w`` is the world-frame velocity of the root
        **link origin** (the dump's ``root_link_lin_vel_w`` key, not the CoM
        velocity) and maps straight onto the free joint's linear qvel;
        ``root_ang_vel_w`` is rotated into the body-local frame.

        ``last_action_raw`` (the dump's action from the settle phase preceding
        t0) seeds the ``last_action`` observation term and, for MLP
        actuator-net groups, warm-starts the net's history buffers to match
        Isaac's settled nets (see :meth:`ActuatorSet.warm_start`); without it
        the nets start cold, as after a plain :meth:`reset`.

        Settled-dump continuation state: ``obs0`` (the dump's first recorded
        observation) seeds the policy group's history buffers with the settle
        phase's rolling frames; ``episode_step`` mirrors Isaac's
        ``episode_length_buf`` at recording start (the dump's ``settle_steps``)
        so the timeout clock and the low-level ``actions``-slot zeroing match,
        and a positive value re-targets state-dependent commands (the pose
        command's body-frame goal) from the restored root state;
        ``low_level_last_raw`` seeds the PreTrainedPolicyAction low-level
        action buffer (the dump's ``init_low_level_action``). The LOW-level
        obs group's own history buffers stay cold — the dump does not record
        low-level observations.
        """
        mujoco.mj_resetData(self.model, self.data)
        # No reset events run on this path: drop any held DR wrench from the
        # previous episode (Isaac zeroes held wrenches in every _reset_idx);
        # mj_resetData already zeroed xfrc_applied.
        self._events.clear_external_wrenches()
        qa, da = self.robot_map.root_qpos_adr, self.robot_map.root_dof_adr
        self.data.qpos[qa : qa + 3] = np.asarray(root_pos_w, dtype=np.float64)
        self.data.qpos[qa + 3 : qa + 7] = np.asarray(root_quat_wxyz, dtype=np.float64)
        self.data.qvel[da : da + 6] = free_joint_qvel(root_quat_wxyz, root_link_lin_vel_w, root_ang_vel_w)
        self.data.qpos[self.robot_map.qpos_adr] = np.asarray(joint_pos_isaac, dtype=np.float64)
        self.data.qvel[self.robot_map.dof_adr] = np.asarray(joint_vel_isaac, dtype=np.float64)
        return self._finish_reset(
            last_action_raw=last_action_raw,
            obs0=obs0,
            episode_step=episode_step,
            low_level_last_raw=low_level_last_raw,
            actuator_lags=actuator_lags,
        )

    def step(self, action_raw: np.ndarray) -> tuple[np.ndarray, bool, bool, dict[str, Any]]:
        """Advance one policy step; returns ``(obs, terminated, truncated, info)``.

        Mirrors ``ManagerBasedRLEnv.step``: the raw action is stored for the
        ``last_action`` observation, the processed position targets drive the
        actuator model for ``decimation`` physics steps, then terminations are
        checked, terminated episodes auto-reset (non-strict only), commands
        tick, interval events fire, and the observation is computed last.
        """
        action_raw = np.asarray(action_raw, dtype=np.float64).reshape(self.action_dim)
        self._last_action_raw = action_raw.astype(np.float32)
        if self._joint_action is not None:
            q_target_isaac = self._joint_action.targets_isaac(action_raw)
        else:
            # The high-level action is held; targets come from the low-level policy at
            # its own cadence inside the decimation loop.
            assert self._pre_trained is not None
            self._pre_trained.process(action_raw)
            q_target_isaac = self._pre_trained._targets_isaac

        for _ in range(self.decimation):
            if self._pre_trained is not None:
                q_target_isaac = self._pre_trained.physics_step_targets(self)
            # Isaac physics-step boundary (physics_dt): delay buffers advance once and
            # the explicit PD torque is computed once, mirroring Isaac's explicit
            # actuators (one torque per physics step).
            q_des, qd_des = self.actuators.step_delay_buffers(q_target_isaac)
            qd_isaac = self.data.qvel[self.robot_map.dof_adr].copy()
            tau_isaac = self.actuators.compute_torques(self.data.qpos[self.robot_map.qpos_adr], qd_isaac, q_des, qd_des)
            # implicit_pd damping lives in the model (dof_damping); add it back into
            # ctrl so the net torque is the clamped total PD (see _check_implicit_damping).
            ctrl_isaac = tau_isaac.copy()
            ctrl_isaac[self._implicit_joint_ids] += self._implicit_kd * qd_isaac[self._implicit_joint_ids]
            self.data.ctrl[:] = ctrl_isaac[self._isaac_to_mj]
            # Explicit groups (ideal/delayed/remotized) hold this ctrl across the
            # substeps — Isaac computes their torque once per physics step. implicit_pd
            # groups are refreshed each substep instead (PhysX re-evaluates the implicit
            # drive against updated state at every solver iteration); dof_damping
            # (implicit_pd kd) keeps integrating against the live joint velocity.
            for substep in range(self.physics_substeps):
                if substep:
                    self._refresh_implicit_ctrl(q_des, qd_des)
                # Held DR wrenches rotate with their link (PhysX re-applies Isaac's
                # link-frame wrench at the current orientation every step).
                self._events.refresh_external_wrenches(self.model, self.data)
                self._stiction.apply(self.model, self.data)
                mujoco.mj_step(self.model, self.data)
            self._terminations.push_contact_forces(self.model, self.data)
            # Recorded applied torque stays the physics-step-boundary value, matching
            # Isaac's once-per-physics-step applied_torque bookkeeping.
            self._applied_torque_isaac = tau_isaac

        mujoco.mj_forward(self.model, self.data)  # fresh kinematics for terminations/obs (scene.update equivalent)
        self._episode_step += 1
        state = self._robot_state()
        terminated, truncated, reasons = self._terminations.check(
            self.model, self.data, state, self._episode_step * self.policy_dt
        )

        did_reset = False
        if (terminated or truncated) and not self.strict:
            self._auto_reset()
            did_reset = True
            state = self._robot_state()

        for name, generator in self._commands.items():
            generator.step(self.policy_dt, state, self.rng)
        if not self.strict:
            if self._events.step_interval(self.policy_dt, self.model, self.data, self.robot_map, self.rng):
                mujoco.mj_forward(self.model, self.data)

        obs = self._compute_obs()
        return obs, terminated, truncated, {"reasons": reasons, "did_reset": did_reset}

    def _refresh_implicit_ctrl(self, q_des_isaac: np.ndarray, qd_des_isaac: np.ndarray) -> None:
        """Recompute the implicit_pd groups' ctrl from the current substep state.

        PhysX's implicit joint drive re-evaluates ``kp*(q_des - q)`` (and the
        effort clamp) against the updated joint state at every solver
        iteration; holding the physics-step-boundary torque is only correct
        for explicit actuator groups. The clamp semantics match
        :meth:`ActuatorSet.compute_torques` on the **total** PD torque
        ``kp*(q_des - q) + kd*(qd_des - qd)``; the ``+kd*qd`` is then added
        back into ctrl because the kd term is realized as ``dof_damping``
        (integrated implicitly by MuJoCo, see :meth:`_check_implicit_damping`).
        """
        ids = self._implicit_joint_ids
        if ids.size == 0:
            return
        qd = self.data.qvel[self._implicit_dof_adr]
        tau = self._implicit_kp * (q_des_isaac[ids] - self.data.qpos[self._implicit_qpos_adr]) + self._implicit_kd * (
            qd_des_isaac[ids] - qd
        )
        np.clip(tau, -self._implicit_effort, self._implicit_effort, out=tau)
        self.data.ctrl[self._implicit_ctrl_adr] = tau + self._implicit_kd * qd

    def obs(self) -> np.ndarray:
        """The last computed observation (reset first)."""
        if self._last_obs is None:
            raise RuntimeError("no observation available yet; call reset() or reset_from_state() first")
        return self._last_obs

    """
    Policy execution.
    """

    def policy_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic policy action for one observation (loads the policy lazily)."""
        return _jit_policy_action(self._load_policy(), obs)

    def run_policy(self, n_steps: int) -> dict[str, np.ndarray]:
        """Closed-loop rollout for ``n_steps`` policy steps from the current state.

        Returns arrays mirroring the reference-dump schema: ``obs``,
        ``action_raw``, ``processed_action``, ``root_pos_w``, ``root_quat_w``,
        ``root_lin_vel_w`` (CoM), ``root_link_lin_vel_w``, ``root_ang_vel_w``,
        ``joint_pos``, ``joint_vel``, ``applied_torque``, ``terminated``,
        ``truncated`` — state at index ``t`` is the state after applying
        ``action_raw[t]``, and ``obs[t]`` is the observation that produced it.
        """
        rec: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "obs",
                "action_raw",
                "processed_action",
                "root_pos_w",
                "root_quat_w",
                "root_lin_vel_w",
                "root_link_lin_vel_w",
                "root_ang_vel_w",
                "joint_pos",
                "joint_vel",
                "applied_torque",
                "terminated",
                "truncated",
            )
        }
        obs = self.obs()
        for _ in range(int(n_steps)):
            action_raw = self.policy_action(obs)
            rec["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            rec["action_raw"].append(action_raw.copy())
            rec["processed_action"].append(self.action.processed(action_raw))
            obs, terminated, truncated, _ = self.step(action_raw)
            state = self.record_state()
            for key, value in state.items():
                rec[key].append(value)
            rec["terminated"].append(np.array(terminated))
            rec["truncated"].append(np.array(truncated))
        return {key: np.stack(values) for key, values in rec.items()}

    def record_state(self) -> dict[str, np.ndarray]:
        """Current robot state in the reference-dump's key/frame conventions."""
        qa, da = self.robot_map.root_qpos_adr, self.robot_map.root_dof_adr
        vel_com_w = read_root_com_vel_w(self.model, self.data, self.robot_map)
        return {
            "root_pos_w": self.data.qpos[qa : qa + 3].copy(),
            "root_quat_w": self.data.qpos[qa + 3 : qa + 7].copy(),
            "root_lin_vel_w": vel_com_w[0:3],
            "root_link_lin_vel_w": self.data.qvel[da : da + 3].copy(),
            "root_ang_vel_w": vel_com_w[3:6],
            "joint_pos": self.data.qpos[self.robot_map.qpos_adr].copy(),
            "joint_vel": self.data.qvel[self.robot_map.dof_adr].copy(),
            "applied_torque": self._applied_torque_isaac.copy(),
        }

    """
    Internals.
    """

    def _load_policy(self):
        if self._policy is None:
            if self._policy_path is None:
                raise RuntimeError("no policy_path was given to MjEnv")
            self._policy = _load_jit_policy(self._policy_path)
        return self._policy

    def _finish_reset(
        self,
        last_action_raw: np.ndarray | None = None,
        obs0: np.ndarray | None = None,
        episode_step: int = 0,
        low_level_last_raw: np.ndarray | None = None,
        actuator_lags: dict[str, int] | None = None,
    ) -> np.ndarray:
        # No xfrc_applied wipe here: both reset paths start from mj_resetData (which
        # zeroes it). reset() runs its reset events BEFORE this — those wrenches must
        # persist until the next reset, matching Isaac — and reset_from_state clears
        # the held wrenches itself (it runs no reset events).
        mujoco.mj_forward(self.model, self.data)
        # Mirrors Isaac's episode_length_buf: 0 for a fresh episode, the dump's
        # settle_steps when continuing a settled reference (timeout clock and the
        # low-level actions-slot zeroing both key off it).
        self._episode_step = int(episode_step)
        # Dumps recorded after a settle phase carry the pre-reset policy action; seeding it
        # keeps the last_action obs term aligned with Isaac at the first recorded step.
        if last_action_raw is None:
            self._last_action_raw = np.zeros(self.action_dim, dtype=np.float32)
        else:
            self._last_action_raw = np.asarray(last_action_raw, dtype=np.float32).reshape(self.action_dim)
        self._reset_runtime()
        # Explicit resets restore a fresh env: the low-level runtime's fire
        # counter, obs history, and action buffers match the just-created Isaac
        # env the dumps are recorded from. Mid-rollout auto-resets keep Isaac's
        # survival semantics instead (see PreTrainedPolicyRuntime).
        if self._pre_trained is not None:
            self._pre_trained.reset_cold()
        # Continuing a settled episode: Isaac ran the command post-processing on
        # every settle step, so state-dependent command parts (the pose command's
        # body-frame goal) are live at t0; a fresh episode serves the reset zeros.
        if self._episode_step > 0:
            state = self._robot_state()
            for generator in self._commands.values():
                generator.retarget(state)
        # Recorded per-group delay lags override the reset's lags (strict pins 0,
        # non-strict samples); must land before warm_start fills the buffers.
        if actuator_lags:
            self.actuators.set_lags(actuator_lags)
        # A settled dump's init also pairs with WARM actuator state (Isaac's nets and
        # delay buffers ran through the settle phase); reconstruct the MLP histories and
        # delay-buffer contents from the t0-consistent constant input: position targets
        # from the recorded pre-recording action, state from the dump (LSTM groups stay
        # cold — see ActuatorSet.warm_start). Plain resets stay cold too: Isaac's
        # ActuatorNetLSTM/ActuatorNetMLP reset() zero their state on env reset. The
        # PreTrainedPolicyAction path has no direct raw-action -> joint-target map (its
        # raw action is the held velocity command), so it keeps cold state.
        if (
            last_action_raw is not None
            and self._joint_action is not None
            and (self.actuators.has_net or self.actuators.has_delay)
        ):
            self.actuators.warm_start(
                self.data.qpos[self.robot_map.qpos_adr],
                self.data.qvel[self.robot_map.dof_adr],
                self._joint_action.targets_isaac(np.asarray(last_action_raw, dtype=np.float64)),
            )
        if obs0 is not None:
            self._obs_pipeline.seed_history_from_flat(obs0)
        if self._pre_trained is not None and low_level_last_raw is not None:
            self._pre_trained.seed_last_raw(low_level_last_raw)
        return self._compute_obs()

    def _reset_runtime(self) -> None:
        """Zero the runtime buffers and reset every stateful component.

        The RNG consumption order — actuators, then height-scan providers, then
        commands — is part of the reproducibility contract; both reset paths
        must draw identically. Dump-seeding (lags, warm start, obs history) is
        RNG-free and layered on top by ``_finish_reset``.
        """
        self._applied_torque_isaac = np.zeros(self.num_joints, dtype=np.float64)
        self.actuators.reset(self.rng, strict=self.strict)
        self._stiction.reset()
        self._terminations.reset()
        self._obs_pipeline.reset()
        # One reset per scanner, matching Isaac's one shared RayCaster per sensor
        # (providers are shared across the policy and low-level pipelines).
        for provider in self._height_scan_providers.values():
            provider.reset(self.rng, strict=self.strict)
        state = self._robot_state()
        for generator in self._commands.values():
            generator.reset(self.rng, state)

    def _auto_reset(self) -> None:
        """Mid-rollout episode reset (non-strict), mirroring ``_reset_idx``.

        Reset events write absolute state on top of the defaults; joints/bodies
        no event covers keep their current state, exactly like IsaacLab.
        """
        self.data.xfrc_applied[:] = 0.0
        self._events.apply_reset(self.model, self.data, self.robot_map, self.rng)
        mujoco.mj_forward(self.model, self.data)
        self._episode_step = 0
        self._last_action_raw = np.zeros(self.action_dim, dtype=np.float32)
        self._reset_runtime()

    def _robot_state(self) -> RobotState:
        qa = self.robot_map.root_qpos_adr
        quat = self.data.qpos[qa + 3 : qa + 7].copy()
        return RobotState(
            root_pos_w=self.data.qpos[qa : qa + 3].copy(),
            root_quat_w_wxyz=quat,
            heading_w=heading_w_from_quat(quat),
            env_origin_w=self.robot_map.env_origin_w,
        )

    def _body_frame_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(root_lin_vel_b, root_ang_vel_b, projected_gravity_b) matching Isaac obs.

        Isaac's ``root_lin_vel_b`` is the root **CoM** velocity in the body
        frame; ``mj_objectVelocity`` with ``mjOBJ_XBODY`` and ``flg_local=1``
        gives the body-origin velocity ``[ang_b, lin_b]``, shifted to the CoM
        via ``lin_vel_b += ang_vel_b x body_ipos``. ``mjOBJ_BODY`` would use
        the inertial frame instead and must not be used here.
        """
        bid = self.robot_map.root_body_id
        vel_buf = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, int(mujoco.mjtObj.mjOBJ_XBODY), bid, vel_buf, 1)
        ang_vel_b = vel_buf[0:3].copy()
        lin_vel_b = vel_buf[3:6] + np.cross(ang_vel_b, self.model.body_ipos[bid])
        rot_wb = self.data.xmat[bid].reshape(3, 3)
        projected_gravity_b = rot_wb.T @ self._gravity_dir_w
        return lin_vel_b, ang_vel_b, projected_gravity_b

    def _make_obs_context(
        self,
        *,
        last_action_raw: np.ndarray,
        height_scan_terms: list[tuple[str, HeightScanProvider, float, int]],
        extra_commands: dict[str, np.ndarray] | None = None,
    ) -> ObsContext:
        """Observation context from the current (forwarded) sim state.

        ``extra_commands`` overlay the live command generators' vectors (the
        low-level pipeline's held-action slot).
        """
        lin_vel_b, ang_vel_b, gravity_b = self._body_frame_state()
        extras: dict[str, np.ndarray] = {}
        for term_name, provider, offset, body_id in height_scan_terms:
            extras[term_name] = provider.height_scan(self.data.xpos[body_id], self.data.xquat[body_id], offset=offset)
        commands = {name: gen.command.copy() for name, gen in self._commands.items()}
        if extra_commands:
            commands.update({name: np.asarray(vec, dtype=np.float64).copy() for name, vec in extra_commands.items()})
        return ObsContext(
            root_lin_vel_b=lin_vel_b,
            root_ang_vel_b=ang_vel_b,
            projected_gravity_b=gravity_b,
            joint_pos_isaac=self.data.qpos[self.robot_map.qpos_adr].copy(),
            joint_vel_isaac=self.data.qvel[self.robot_map.dof_adr].copy(),
            default_joint_pos_isaac=self.robot_map.default_joint_pos,
            default_joint_vel_isaac=self.robot_map.default_joint_vel,
            last_action_raw=last_action_raw,
            commands=commands,
            command_time_left={name: float(gen.time_left) for name, gen in self._commands.items()},
            extras=extras,
        )

    def _compute_obs(self) -> np.ndarray:
        ctx = self._make_obs_context(last_action_raw=self._last_action_raw, height_scan_terms=self._height_scan_terms)
        self._last_obs = self._obs_pipeline.compute(ctx, self.rng)
        return self._last_obs
