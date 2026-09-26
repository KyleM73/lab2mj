"""Dump a fully-resolved ``env.yaml`` for a registered task without Isaac Sim.

Installs :mod:`lab2mj.isaac_shims` so the isaaclab config layer imports on a
machine without the Omniverse runtime, loads the task's env config from the gym
registry exactly like ``isaaclab_tasks.utils.parse_env_cfg``, applies the config
mutations the training pipeline would apply before dumping (scene-entity prim-path
resolution, terrain num_envs/env_spacing, optional seed and PhysX buffer scaling),
and writes the config through the real ``isaaclab.utils.io.dump_yaml``.

The output is structurally identical to the ``params/env.yaml`` a training run dumps
on a GPU box, except for the run-specific keys ``log_dir`` and
``io_descriptors_output_dir`` (and ``seed``/``sim.device`` when not matched via CLI).

Example::

    uv run lab2mj-dump-env-cfg --task Isaac-Velocity-Flat-Anymal-D-v0 \\
        --out /tmp/anymal_d_flat/params/env.yaml --seed 42

Rough-terrain a1/go1/go2 configs mutate a shared module-level terrain config in
``__post_init__``, so dump one task per process (same caveat as the real dumper).
"""

from __future__ import annotations

import argparse


def resolve_scene_entities(env_cfg) -> None:
    """Apply the scene-config mutations ``InteractiveScene._add_entities_from_cfg`` performs.

    The real ``params/env.yaml`` is dumped after env construction, by which time the
    scene has resolved ``{ENV_REGEX_NS}`` placeholders in prim paths and copied
    ``num_envs``/``env_spacing`` onto the terrain-importer config. Replicating those
    mutations here keeps local dumps byte-comparable with on-box dumps.
    """
    from isaaclab.assets import RigidObjectCollectionCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.terrains.height_field import HfTerrainBaseCfg

    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None:
        return
    env_regex_ns = "/World/envs/env_.*"
    for entity_name, entity_cfg in scene_cfg.__dict__.items():
        if entity_name in InteractiveSceneCfg.__dataclass_fields__ or entity_cfg is None:
            continue
        if hasattr(entity_cfg, "prim_path"):
            entity_cfg.prim_path = entity_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
        if isinstance(entity_cfg, TerrainImporterCfg):
            entity_cfg.num_envs = scene_cfg.num_envs
            entity_cfg.env_spacing = scene_cfg.env_spacing
            # TerrainGenerator.__init__ pushes generator-level values onto every
            # sub-terrain config before generation.
            generator_cfg = entity_cfg.terrain_generator
            if entity_cfg.terrain_type == "generator" and generator_cfg is not None:
                for sub_cfg in generator_cfg.sub_terrains.values():
                    sub_cfg.size = generator_cfg.size
                    if isinstance(sub_cfg, HfTerrainBaseCfg):
                        sub_cfg.horizontal_scale = generator_cfg.horizontal_scale
                        sub_cfg.vertical_scale = generator_cfg.vertical_scale
                        sub_cfg.slope_threshold = generator_cfg.slope_threshold
        elif isinstance(entity_cfg, RigidObjectCollectionCfg):
            for rigid_object_cfg in entity_cfg.rigid_objects.values():
                rigid_object_cfg.prim_path = rigid_object_cfg.prim_path.format(ENV_REGEX_NS=env_regex_ns)
        elif isinstance(entity_cfg, FrameTransformerCfg):
            for target_frame in entity_cfg.target_frames:
                target_frame.prim_path = target_frame.prim_path.format(ENV_REGEX_NS=env_regex_ns)
        elif isinstance(entity_cfg, ContactSensorCfg):
            entity_cfg.filter_prim_paths_expr = [
                expr.format(ENV_REGEX_NS=env_regex_ns) for expr in entity_cfg.filter_prim_paths_expr
            ]


def hydra_roundtrip(env_cfg):
    """Round-trip the config through OmegaConf exactly like ``hydra_task_config`` does.

    Both stock IsaacLab's rsl_rl ``train.py`` and contact_lab's ``scripts/train.py``
    load configs through ``isaaclab_tasks.utils.hydra``, which passes the config dict
    through OmegaConf and back via ``from_dict``. OmegaConf represents tuples as
    lists; ``from_dict`` restores tuple-typed attributes but keeps lists that sit
    inside plain dict values (e.g. reward-term params), so the round-trip leaves a
    deterministic tuple-to-list residue that on-box dumps carry.
    """
    from typing import Any, cast

    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_slices_with_strings, replace_strings_with_slices
    from omegaconf import OmegaConf

    env_cfg: Any = replace_env_cfg_spaces_with_strings(env_cfg)
    cfg_dict = replace_slices_with_strings(env_cfg.to_dict())
    cfg_dict = cast(dict, OmegaConf.to_container(OmegaConf.create(cfg_dict), resolve=True))
    cfg_dict = replace_strings_with_slices(cfg_dict)
    env_cfg.from_dict(cfg_dict)
    return replace_strings_with_env_cfg_spaces(env_cfg)


def scale_physx_buffers(env_cfg) -> None:
    """Apply the PhysX GPU buffer scaling ``scripts/train.py`` applies before dumping.

    Only contact_lab's training entry point does this; stock IsaacLab rsl_rl runs do
    not. Enable via ``--train_physx_buffers`` when matching a contact_lab run.
    """
    if not (hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx")):
        return
    n = env_cfg.scene.num_envs
    physx = env_cfg.sim.physx
    physx.gpu_max_rigid_patch_count = max(physx.gpu_max_rigid_patch_count, 20 * n)
    physx.gpu_max_rigid_contact_count = max(physx.gpu_max_rigid_contact_count, 64 * n)
    physx.gpu_found_lost_pairs_capacity = max(physx.gpu_found_lost_pairs_capacity, 64 * n)
    physx.gpu_found_lost_aggregate_pairs_capacity = max(physx.gpu_found_lost_aggregate_pairs_capacity, 128 * n)
    physx.gpu_total_aggregate_pairs_capacity = max(physx.gpu_total_aggregate_pairs_capacity, 64 * n)
    physx.gpu_collision_stack_size = max(physx.gpu_collision_stack_size, 1 << 27)
    physx.gpu_heap_capacity = max(physx.gpu_heap_capacity, 1 << 27)
    physx.gpu_temp_buffer_capacity = max(physx.gpu_temp_buffer_capacity, 1 << 25)


def dump_env_cfg(
    task: str,
    out: str,
    num_envs: int | None = None,
    device: str = "cuda:0",
    seed: int | None = None,
    train_physx_buffers: bool = False,
    use_hydra_roundtrip: bool = True,
) -> None:
    """Resolve the env config for ``task`` under shims and dump it to ``out``."""
    from lab2mj.isaac_shims import install

    install()

    # Register stock IsaacLab tasks, then contact_lab tasks (guarded: the contact_lab
    # stack is optional for dumping stock tasks).
    import isaaclab_tasks  # noqa: F401

    try:
        import contact_lab.tasks  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"[WARN] contact_lab task registration failed ({exc}); stock tasks only")

    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if use_hydra_roundtrip:
        env_cfg = hydra_roundtrip(env_cfg)
    if seed is not None:
        env_cfg.seed = seed
    resolve_scene_entities(env_cfg)
    if train_physx_buffers:
        scale_physx_buffers(env_cfg)
    dump_yaml(out, env_cfg)
    print(f"[INFO] dumped {task} -> {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="Registered task id (stock IsaacLab or contact_lab).")
    parser.add_argument("--out", required=True, help="Output yaml path.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override scene.num_envs (default: task default).")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Value written to sim.device; only serialized, nothing runs on it (default matches on-box dumps).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Value written to the seed key (train scripts write the agent seed)."
    )
    parser.add_argument(
        "--train_physx_buffers",
        action="store_true",
        help="Replicate contact_lab train.py's PhysX GPU buffer scaling (contact_lab runs only).",
    )
    parser.add_argument(
        "--no_hydra_roundtrip",
        action="store_true",
        help="Skip the OmegaConf round-trip that hydra-based train scripts apply to the config.",
    )
    args = parser.parse_args(argv)
    dump_env_cfg(
        task=args.task,
        out=args.out,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        train_physx_buffers=args.train_physx_buffers,
        use_hydra_roundtrip=not args.no_hydra_roundtrip,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
