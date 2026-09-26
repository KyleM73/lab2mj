---
name: sim2sim-new-task
description: Onboard a new robot or task into the lab2mj sim2sim fleet — registry entries, reference/parity dumps, conversion, validation gates, and the acceptance bar.
---

# Add a new robot/task to the sim2sim fleet

Machine split: Isaac-side dumpers need the GPU box (`ssh a5090`; run them from
the `~/contact_lab` venv with this repo checked out alongside at `~/lab2mj` and
installed into that venv); conversion and every replay/validate run on mac CPU. The
converter runs ONLY on mac — the remote venv's `pxr` is Isaac Sim's and needs a
SimulationApp.

## 1. Registry entries (robot-based protocols)

- `scripts/isaac_dump_common.py`: add the robot to `ROBOT_CHOICES` and
  `load_robot_cfg` (import its articulation cfg; some cfgs are not re-exported
  from the `isaaclab_assets` root — import from the submodule).
- `scripts/compare_env_cfg_dumps.py`: add a `CASES` row so the mac-local
  config dump is checked against the on-box `env.yaml`.

Task-based dumps (`dump_isaac_reference.py`) need no registry — any gym task id
works.

## 2. Dumps (GPU box)

```bash
# Task reference (ground truth for the gates). Wrap in `timeout -k 10 420`.
uv run python scripts/dump_isaac_reference.py --task <id> --policy <policy.pt> \
    --command 0.7,0.0,0.0 --settle_steps 100 --record_physics_steps --seed 42 --out <ref.npz>
# Pose tasks: --pose_command=x,y,z,heading (`=` syntax for leading negatives).

# Plant + contact parity (once per robot).
uv run python scripts/dump_isaac_freespace.py --robot <name> --out <freespace.npz>
uv run python scripts/dump_isaac_contact.py --robot <name> --out <contact.npz>
```

Dumps are self-describing (task, policy path, seed, commands) — the exact
invocation is reconstructable from the npz. PhysX GPU rollouts are NOT
run-to-run deterministic: pin dumps between comparisons; never attribute
metric drift to code without a fixed dump.

## 3. Convert + validate (mac)

```bash
uv run lab2mj-convert --run <run_dir> --dump <ref.npz>   # --dump: measured terrain + PhysX plant
uv run lab2mj-validate --bundle <bundle> --dump <ref.npz> --out logs/validate/<name>
```

Acceptance bar: the obs gate (first-observation parity) must pass (~1e-6; an
obs failure is always a config/frame bug, never physics). The open_loop gate (action
replay) is 0.05 rad over 0.5 s; closed_loop runs the policy; behavior is
informative. `aggregate_sim2sim_matrix.py` rolls up
reports and prints per-row gate utilization (1.0 = at the gate; rows marked `*`
ride it and can flip on cross-machine float noise — compare utilizations, not
PASS/FAIL bits).

## 4. New-robot checklist (things that bit us)

- **Contact profile**: `--contact_profile auto` follows the actuator family and
  is validated on the shipped robots only — compliant-leg PD robots want
  `engagement`, stiff/actuator-net robots want `default`. Check both if the
  open_loop gate misses.
- **Delayed-PD actuators**: Isaac samples a per-reset delay lag; dumps record
  `init_actuator_lag_<group>` and the replay seeds it. An old dump without the
  keys replays at lag 0 and diverges.
- Triangle-inequality-violating authored inertias and USD-authored armature are
  handled automatically (exact-inertia stamping; USD armature is the converter
  default) — the freespace protocol verifies both.
- Closed kinematic chains (Digit-style) are unsupported (need equality
  constraints).
- **Robot USD must be Z-up** (stage `upAxis = Z`). `lab2mj-convert` rejects Y-up
  stages: the up-axis correction would put the MuJoCo root frame out of
  agreement with the Isaac root link frame that dumps, resets, and body-frame
  observations assume. Re-export the asset Z-up.
- If gates fail, run the `sim2sim-diagnose` skill's layered protocol before
  touching parameters.
