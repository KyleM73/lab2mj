# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What This Is

`lab2mj` converts Isaac Lab manager-based RL training runs (env.yaml + robot USD)
into self-contained MuJoCo bundles and validates them against Isaac reference
dumps. It was split out of `contact_lab` (the Spot / G1 contact-rich locomotion
training repo); the two repos import nothing from each other except that the
Isaac-side dumper scripts and `lab2mj.dump_env_cfg` register `contact_lab.tasks`
when that package is importable, and one test cross-checks against
`contact_lab.utils.mj.isaac_parity` (skips otherwise).

## Setup

Requires Python 3.11+ and uv.

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
make sync            # uv sync --group dev --group isaac, then scripts/fix_isaaclab_stubs.py
```

Library deps are numpy + mujoco + scipy + pyyaml only. `torch` and `pxr` are
imported lazily inside the functions that need them; keep it that way so the
runtime stays usable on CPU-only machines without those packages.

## Common Commands

```bash
uv run lab2mj-convert --run <run_dir> [--dump reference.npz]
uv run lab2mj-play --bundle <bundle_dir> --policy policy.pt [--viewer]
uv run lab2mj-validate --bundle <bundle_dir> --dump reference.npz --out logs/validate/<run>
uv run lab2mj-dump-env-cfg --task <id> --out env.yaml   # no Isaac Sim (isaac_shims)

make test            # uv run pytest -q, ~40 s with cached assets; asset-dependent tests skip
make lint            # ruff check . && ruff format --check .
make typecheck       # uvx ty check .
```

Ruff: line length 120, Python 3.11, rules E/F/W/I; `E402` is ignored because the
Isaac-side scripts must launch the Kit app before importing `isaaclab`. Use
`ruff` + `ty` (seconds each). Do NOT run `pyright`.

## Data and caches

Everything under `data/` and `logs/` is gitignored:

- `data/usd_cache/`, `data/actuator_nets/`, `data/policy_cache/` — auto-downloaded by
  `lab2mj-convert`; root is `<repo>/data` from a checkout, `./data` for an installed
  copy, or `$LAB2MJ_DATA_DIR` when set (`lab2mj.convert.DATA_DIR`).
- `data/mj_bundles/` — converted bundles. `data/sim2sim_dumps/` — Isaac reference
  dumps (self-describing: task, policy path, seed, commands).

## Protocol scripts

`scripts/` holds the Isaac-side dumpers (`dump_isaac_reference.py`,
`dump_isaac_freespace.py`, `dump_isaac_contact.py`, shared helpers in
`isaac_dump_common.py`) and their MuJoCo-side counterparts / analysis tools.
The dumpers need a GPU box with Isaac Sim; run them from the training project's
venv with lab2mj installed (for `contact_lab` tasks, from a `contact_lab`
checkout: `python ../lab2mj/scripts/dump_isaac_reference.py ...`). Everything
else runs on mac CPU.

Isaac-side scripts follow the Isaac import order: stdlib / third-party, then the
Kit app launch, then `isaaclab` imports, then task-package imports.

## Skills

`.claude/skills/sim2sim-new-task` (onboard a new robot/task: registries, dumps,
gates, pitfalls) and `.claude/skills/sim2sim-diagnose` (layered gate-failure
diagnosis and measured dead ends).
