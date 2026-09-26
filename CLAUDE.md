# CLAUDE.md

Isaac Lab -> MuJoCo sim2sim converter, runtime, and validation harness. Split out of
`contact_lab`; neither imports the other, except that the Isaac-side dumpers and
`dump_env_cfg` register `contact_lab.tasks` when importable.

## Commands

```bash
make sync        # uv sync --group dev --group isaac + scripts/fix_isaaclab_stubs.py
make test        # pytest lab2mj/tests (~40 s); asset-dependent tests skip
make lint        # ruff check + format --check (line length 120, E/F/W/I; E402 ignored for Isaac scripts)
make typecheck   # uvx ty check .   -- never pyright
```

## Rules

- Library deps are numpy + mujoco + scipy + pyyaml. `torch` and `pxr` import lazily
  inside the functions that need them; `isaaclab` only in `lab2mj/isaac/` and
  `dump_env_cfg`.
- MuJoCo is pinned `<3.8`: newer releases drop `mjData.qM` and change enum/int
  comparisons (see `stiction.py`, `convert._mj_joint_order`).
- Isaac-side dumpers run as `python -m lab2mj.isaac.dump_*` from any venv with lab2mj
  installed. Never assume where a checkout lives.
- `data/` and `logs/` are gitignored. Never commit robot assets; they download from
  NVIDIA's asset server. `LAB2MJ_DATA_DIR` relocates the cache root.
- Ports of Isaac Lab code (`heightscan.py`, `terrain.py`, command terms) keep the
  BSD-3 notice in `licenses/isaaclab.txt` current.
- Comments and docs stay short. Skills: `.claude/skills/sim2sim-new-task`,
  `.claude/skills/sim2sim-diagnose`.
