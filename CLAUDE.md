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
- MuJoCo >=3.7 (validated through 3.14; `stiction.py` and `_mj_joint_order` handle the
  3.8+ CSR inertia and int-typed enums). 3.13 rewrote the plane-mesh collider: bundles
  with mesh feet (g1, h1) get a fuller contact manifold and sit closer to Isaac than on
  3.7; primitive-foot bundles (spot, anymal, unitree) are bit-identical. The Spot oracle
  (obs 2.98e-08, open_loop 0.0302) holds on every version.
- Isaac-side dumpers run as `python -m lab2mj.isaac.dump_*` from any venv with lab2mj
  installed. Never assume where a checkout lives.
- `data/` and `logs/` are gitignored. Never commit robot assets; they download from
  NVIDIA's asset server. `LAB2MJ_DATA_DIR` relocates the cache root.
- Ports of Isaac Lab code (`heightscan.py`, `terrain.py`, command terms) keep the
  BSD-3 notice in `licenses/isaaclab.txt` current.
- Comments and docs stay short. Skills: `.claude/skills/sim2sim-new-task`,
  `.claude/skills/sim2sim-diagnose`.
