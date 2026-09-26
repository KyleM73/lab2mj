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
# Standalone checkout (Python 3.11+, uv).
uv venv .venv --python 3.11 && source .venv/bin/activate
make sync                      # uv sync --group dev --group isaac + scripts/fix_isaaclab_stubs.py

# As a dependency of another project.
uv add "lab2mj[all] @ git+ssh://git@github.com/KyleM73/lab2mj.git"
uv pip install "lab2mj[all] @ git+ssh://git@github.com/KyleM73/lab2mj.git"
uv pip install -e ../lab2mj    # local development copy
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

## Protocol scripts (`scripts/`)

Isaac-side dumpers need a GPU box with Isaac Sim + Isaac Lab (and, for
`contact_lab` tasks, that package on the path); replays and analysis run on a
CPU-only machine.

| Script | Side | Purpose |
| --- | --- | --- |
| `dump_isaac_reference.py` | Isaac | Task rollout ground truth (the input to `lab2mj-validate`) |
| `dump_isaac_freespace.py` | Isaac | Plant parity, no contact |
| `dump_isaac_contact.py` | Isaac | Contact parity: passive + stiff drops |
| `replay_freespace_mujoco.py` | MuJoCo | Counterpart of the free-space dump |
| `replay_contact_mujoco.py` | MuJoCo | Counterpart of the contact dump |
| `render_ghost_compare.py` | MuJoCo | MuJoCo rollout vs Isaac ghost video |
| `aggregate_sim2sim_matrix.py` | MuJoCo | Matrix table + per-bundle distribution rollup |
| `calibrate_contact.py` | MuJoCo | Offline contact-parameter sweep against a dump |
| `one_step_parity.py` | MuJoCo | Per-step model error at matched states |
| `compare_env_cfg_dumps.py` | either | Deep-compare shimmed env.yaml dumps against real on-box training dumps |

Run the Isaac-side dumpers from the training project's venv with lab2mj
installed, e.g. from a `contact_lab` checkout:

```bash
python ../lab2mj/scripts/dump_isaac_reference.py --task spot-velocity-v0 ... --out data/sim2sim_dumps/x.npz
```

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
  tests/           offline pytest suite + env.yaml fixtures
scripts/           Isaac-side dumpers and MuJoCo-side replay / analysis protocols
.claude/skills/    sim2sim-new-task, sim2sim-diagnose (Claude Code workflows)
```
