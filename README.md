# lab2mj

Isaac Lab -> MuJoCo sim2sim converter, runtime, and validation harness.

`lab2mj` converts any IsaacLab manager-based training run (its `params/env.yaml` +
robot USD) into a self-contained MuJoCo bundle that runs on a CPU-only machine —
no Isaac Sim needed for conversion or evaluation. The bundle carries the robot
MJCF, terrain, actuator model (implicit PD, DC motor, remotized PD, TorchScript
actuator nets), observation pipeline, commands, events, and terminations, so a
policy trained in Isaac Lab can be replayed and validated against an Isaac
reference dump.

The library depends only on numpy, mujoco, scipy, and PyYAML. `usd-core` is
needed for conversion, `torch` for TorchScript policies / actuator nets, and
`matplotlib` + `pillow` for plots and ghost renders — all optional extras.

## Install

```bash
# As a dependency of another project (e.g. the venv you train in).
uv pip install "lab2mj[all] @ git+https://github.com/KyleM73/lab2mj.git"
uv add "lab2mj[all] @ git+https://github.com/KyleM73/lab2mj.git"

# Standalone checkout for development (Python 3.11+, uv).
uv venv .venv --python 3.11 && source .venv/bin/activate
make sync                      # uv sync --group dev --group isaac + scripts/fix_isaaclab_stubs.py
```

The `lab2mj-*` commands below are console entry points: run them via `uv run`,
or bare inside an activated venv.

## Usage

```bash
# Convert a training run into a bundle (USD / actuator nets auto-download into data/).
# The robot USD must be Z-up (stage upAxis = Z); Y-up stages are rejected.
uv run lab2mj-convert --run <run_dir> [--dump reference.npz] [--substeps N] \
    [--contact_profile auto|default|engagement] [--terrain_exact/--no-terrain_exact]

# Run a TorchScript policy in the bundle (optionally in the viewer).
uv run lab2mj-play --bundle <bundle_dir> --policy policy.pt --command "0.7,0,0" [--viewer]

# Validate against an Isaac reference dump (gates: obs, open_loop, closed_loop, behavior).
uv run lab2mj-validate --bundle <bundle_dir> --dump reference.npz --out logs/validate/<run>

# Dump a fully-resolved env.yaml WITHOUT Isaac Sim (curated Omniverse shims in lab2mj/isaac_shims.py).
uv run lab2mj-dump-env-cfg --task <task-id> --out env.yaml
```

Bundles live under `data/mj_bundles/`, reference dumps under `data/sim2sim_dumps/`,
and downloaded USDs / actuator nets / low-level policies under `data/usd_cache/`,
`data/actuator_nets/`, `data/policy_cache/` (all gitignored). Set
`LAB2MJ_DATA_DIR` to relocate that root — required when lab2mj is installed into
another project's venv rather than run from a checkout. Dumps are
self-describing: task, policy path, seed, and commands reconstruct the exact dump
invocation.

## Protocol scripts

Isaac-side dumpers live in the package (`lab2mj.isaac`) and run as modules from
any venv that has lab2mj installed on a GPU box with Isaac Sim + Isaac Lab; for
`contact_lab` tasks run them from a `contact_lab` checkout so its task registry
imports. Everything else is under `scripts/` and runs on a CPU-only machine.

| Entry point | Side | Purpose |
| --- | --- | --- |
| `python -m lab2mj.isaac.dump_reference` | Isaac | Task rollout ground truth (the input to `lab2mj-validate`) |
| `python -m lab2mj.isaac.dump_freespace` | Isaac | Plant parity, no contact |
| `python -m lab2mj.isaac.dump_contact` | Isaac | Contact parity: passive + stiff drops |
| `scripts/replay_freespace_mujoco.py` | MuJoCo | Counterpart of the free-space dump |
| `scripts/replay_contact_mujoco.py` | MuJoCo | Counterpart of the contact dump |
| `scripts/render_ghost_compare.py` | MuJoCo | MuJoCo rollout vs Isaac ghost video |
| `scripts/aggregate_sim2sim_matrix.py` | MuJoCo | Matrix table + per-bundle distribution rollup |
| `scripts/calibrate_contact.py` | MuJoCo | Offline contact-parameter sweep against a dump |
| `scripts/one_step_parity.py` | MuJoCo | Per-step model error at matched states |
| `scripts/compare_env_cfg_dumps.py` | either | Deep-compare shimmed env.yaml dumps against real training dumps (`--logs_root`) |

```bash
python -m lab2mj.isaac.dump_reference --task Isaac-Velocity-Flat-G1-v0 --policy policy.pt \
    --command 0.7,0,0 --settle_steps 100 --record_physics_steps --seed 42 --out data/sim2sim_dumps/g1.npz
```

## Robots

The converter is robot-agnostic: any Isaac Lab manager-based task whose robot
USD is Z-up and whose actuators are implicit PD, DC motor, delayed PD, remotized
PD, or a TorchScript actuator net (LSTM / MLP) converts from its `env.yaml`. No
robot assets ship in this repository — USDs and actuator nets download from the
Isaac Lab asset server into `data/` on first conversion.

Validated so far (reference dumps, free-space and contact parity):
Unitree A1 / Go1 / Go2 / G1 / H1, ANYmal-B / C / D (incl. the ANYmal-C
navigation task), Agility Cassie, and Boston Dynamics Spot (stock Isaac Lab
velocity task and the `contact_lab` Spot tasks). The task-based reference dumper
takes any gym task id; the robot-based parity dumpers (`dump_freespace`,
`dump_contact`) use a small articulation-config registry in
`lab2mj/isaac/dump_common.py` (Spot, G1, H1, A1, Go1, Go2, ANYmal-B/C/D) that a
new robot is added to. Test fixtures are real
`env.yaml` dumps for Spot, G1 (flat + rough deltas), and an ANYmal-C nav-like
task.

## Tests

```bash
make test        # == uv run pytest -q   (~40 s with cached assets, offline, no Isaac Sim)
make lint        # ruff check + format --check
make typecheck   # uvx ty check .
```

Tests that need downloaded USDs, bundles, dumps, or policies skip when the
asset is absent from `data/`; the pure-numpy core (~250 tests) always runs.

## Layout

```
lab2mj/            library (numpy + mujoco + scipy + yaml; torch / pxr lazily)
  usd2mjcf/        USD -> MJCF parser, builder, verifier
  isaac/           Isaac-side dumpers (`python -m lab2mj.isaac.dump_*`)
  tests/           offline pytest suite + env.yaml fixtures
scripts/           MuJoCo-side replay / analysis protocols, Isaac Lab stub fixup
licenses/          third-party notices
.claude/skills/    sim2sim-new-task, sim2sim-diagnose (Claude Code workflows)
```

## License

MIT (see `LICENSE`). `lab2mj/heightscan.py`, `lab2mj/terrain.py`, and the command
terms in `lab2mj/commands.py` are numpy ports of Isaac Lab v2.3 semantics; Isaac
Lab's BSD-3-Clause notice is reproduced in `licenses/isaaclab.txt`. Robot assets
are not redistributed here; they download from NVIDIA's asset server under their
own terms.
