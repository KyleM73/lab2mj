# lab2mj

Isaac Lab -> MuJoCo sim2sim. Converts a manager-based training run (`params/env.yaml`
+ robot USD) into a self-contained MuJoCo bundle, runs the trained policy in it, and
validates the bundle against an Isaac reference dump. Robot-agnostic; no assets ship
here (USDs and actuator nets download into `data/` on first conversion).

Core deps: numpy, mujoco (<3.8), scipy, pyyaml. Extras: `usd` (conversion), `torch`
(policies / actuator nets), `plot`, `all`.

## Install

```bash
uv pip install "lab2mj[all] @ git+https://github.com/KyleM73/lab2mj.git"

# development checkout
uv venv .venv --python 3.11 && source .venv/bin/activate && make sync
```

## Use

```bash
lab2mj-convert  --run <run_dir> [--dump reference.npz]                 # run -> bundle (USD must be Z-up)
lab2mj-play     --bundle <bundle> --policy policy.pt [--viewer]
lab2mj-validate --bundle <bundle> --dump reference.npz --out <dir>     # gates: obs, open_loop, closed_loop, behavior
lab2mj-dump-env-cfg --task <id> --out env.yaml                         # resolved env.yaml without Isaac Sim
```

Reference dumps come from a GPU box with Isaac Sim, from whatever venv has lab2mj
installed (a `contact_lab` checkout for its tasks):

```bash
python -m lab2mj.isaac.dump_reference --task <id> --policy policy.pt --command 0.7,0,0 --out ref.npz
python -m lab2mj.isaac.dump_freespace --robot <name> --out freespace.npz   # plant parity
python -m lab2mj.isaac.dump_contact   --robot <name> --out contact.npz     # contact parity
```

`scripts/` holds the MuJoCo-side replay, calibration, and aggregation counterparts.
Bundles, dumps, and caches live under `data/` (gitignored; `LAB2MJ_DATA_DIR` overrides).

## Develop

```bash
make test        # pytest, ~40 s; asset-dependent tests skip when data/ is empty
make lint        # ruff
make typecheck   # ty
```
