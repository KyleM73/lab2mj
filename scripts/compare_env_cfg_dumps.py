"""Compare mac-local env-config dumps against on-box ground-truth ``env.yaml`` files.

For every ground-truth dump available in the repo (``lab2mj/tests/fixtures`` and
``logs/<run>/params/env.yaml``), re-dump the same task with
``uv run lab2mj-dump-env-cfg`` (no Isaac Sim, shimmed Omniverse runtime) and
structurally deep-compare the two yamls. Every key must be identical except the
documented run-specific allowlist below.

Each dump runs in its own subprocess — one task per process (see
``lab2mj.dump_env_cfg``).

Usage::

    uv run python scripts/compare_env_cfg_dumps.py [--workdir DIR] [--only REGEX]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from lab2mj.env_yaml import load_env_yaml

REPO = Path(__file__).resolve().parents[1]

# Keys legitimately allowed to differ between a mac dump and an on-box training dump:
#   log_dir / io_descriptors_output_dir — run-specific filesystem paths set by the
#       train scripts at launch time.
#   seed — written from the agent config by the train scripts; matched via --seed 42
#       here, so a diff would indicate a run trained with a non-default seed.
#   sim.device — runtime device string; matched via --device cuda:0 here.
ALLOWLIST = {"log_dir", "io_descriptors_output_dir", "seed", "sim.device"}

# (name, ground-truth glob relative to repo root, task id, extra dumper args).
# --train_physx_buffers replicates contact_lab train.py's PhysX buffer scaling and is
# only used for runs trained through contact_lab's scripts/train.py.
CASES: list[tuple[str, str, str, list[str]]] = [
    ("anymal_b_flat", "logs/anymal_b_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Anymal-B-v0", []),
    ("anymal_b_rough", "logs/anymal_b_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Anymal-B-v0", []),
    ("anymal_c_flat", "logs/anymal_c_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Anymal-C-v0", []),
    ("anymal_c_rough", "logs/anymal_c_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Anymal-C-v0", []),
    ("anymal_c_nav", "logs/anymal_c_nav/*/params/env.yaml", "Isaac-Navigation-Flat-Anymal-C-v0", []),
    ("anymal_d_flat", "logs/anymal_d_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Anymal-D-v0", []),
    ("anymal_d_rough", "logs/anymal_d_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Anymal-D-v0", []),
    ("g1_flat", "logs/g1_flat/*/params/env.yaml", "Isaac-Velocity-Flat-G1-v0", []),
    ("g1_flat_fixture", "lab2mj/tests/fixtures/g1_flat_env.yaml", "Isaac-Velocity-Flat-G1-v0", []),
    ("g1_rough", "logs/g1_rough/*/params/env.yaml", "Isaac-Velocity-Rough-G1-v0", []),
    ("h1_flat", "logs/h1_flat/*/params/env.yaml", "Isaac-Velocity-Flat-H1-v0", []),
    ("h1_rough", "logs/h1_rough/*/params/env.yaml", "Isaac-Velocity-Rough-H1-v0", []),
    ("unitree_a1_flat", "logs/unitree_a1_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Unitree-A1-v0", []),
    ("unitree_a1_rough", "logs/unitree_a1_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Unitree-A1-v0", []),
    ("unitree_go1_flat", "logs/unitree_go1_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Unitree-Go1-v0", []),
    ("unitree_go1_rough", "logs/unitree_go1_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Unitree-Go1-v0", []),
    ("unitree_go2_flat", "logs/unitree_go2_flat/*/params/env.yaml", "Isaac-Velocity-Flat-Unitree-Go2-v0", []),
    ("unitree_go2_rough", "logs/unitree_go2_rough/*/params/env.yaml", "Isaac-Velocity-Rough-Unitree-Go2-v0", []),
    ("spot_flat_stock", "logs/spot_flat_stock/*/params/env.yaml", "Isaac-Velocity-Flat-Spot-v0", []),
    ("spot_velocity", "logs/spot_velocity/*/params/env.yaml", "spot-velocity-v0", ["--train_physx_buffers"]),
    (
        "spot_velocity_fixture",
        "lab2mj/tests/fixtures/spot_velocity_env.yaml",
        "spot-velocity-v0",
        ["--train_physx_buffers"],
    ),
]


def deep_compare(local: Any, ref: Any, path: str = "") -> list[tuple[str, str, str]]:
    """Return (key_path, local_repr, ref_repr) for every structural difference."""
    diffs: list[tuple[str, str, str]] = []
    # Numeric cross-type equality (1 vs 1.0) is allowed, but bool is not a number here:
    # True vs 1 is a real config difference.
    numeric = (
        isinstance(local, (int, float))
        and isinstance(ref, (int, float))
        and isinstance(local, bool) == isinstance(ref, bool)
    )
    if type(local) is not type(ref) and not numeric:
        diffs.append((path, f"{type(local).__name__}: {local!r}"[:120], f"{type(ref).__name__}: {ref!r}"[:120]))
    elif isinstance(local, dict):
        for key in sorted(set(local) | set(ref), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in local:
                diffs.append((sub, "<missing>", repr(ref[key])[:120]))
            elif key not in ref:
                diffs.append((sub, repr(local[key])[:120], "<missing>"))
            else:
                diffs.extend(deep_compare(local[key], ref[key], sub))
    elif isinstance(local, (list, tuple)):
        if len(local) != len(ref):
            diffs.append((path, f"len {len(local)}", f"len {len(ref)}"))
        else:
            for i, (lv, rv) in enumerate(zip(local, ref)):
                diffs.extend(deep_compare(lv, rv, f"{path}[{i}]"))
    elif local != ref and not (local != local and ref != ref):
        diffs.append((path, repr(local)[:120], repr(ref)[:120]))
    return diffs


def allowlisted(key_path: str) -> bool:
    return any(key_path == allowed or key_path.startswith(allowed + ".") for allowed in ALLOWLIST)


def run_case(name: str, ref_path: Path, task: str, extra: list[str], workdir: Path) -> dict[str, Any]:
    out_path = workdir / f"{name}.yaml"
    cmd = [
        sys.executable,
        "-m",
        "lab2mj.dump_env_cfg",
        "--task",
        task,
        "--out",
        str(out_path),
        "--seed",
        "42",
        "--device",
        "cuda:0",
        *extra,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if proc.returncode != 0:
        return {"name": name, "task": task, "status": "DUMP FAILED", "detail": proc.stderr.strip().splitlines()[-8:]}
    # Safe loader with the Isaac-specific tags; unsafe_load would execute arbitrary
    # code from a copied env.yaml.
    local = load_env_yaml(out_path)
    ref = load_env_yaml(ref_path)
    diffs = deep_compare(local, ref)
    real = [d for d in diffs if not allowlisted(d[0])]
    allowed = [d for d in diffs if allowlisted(d[0])]
    status = "IDENTICAL" if not diffs else ("ALLOWLISTED DIFFS" if not real else "REAL DIFFS")
    return {"name": name, "task": task, "status": status, "allowed": allowed, "real": real, "ref": str(ref_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default=None, help="Directory for local dumps (default: temp dir).")
    parser.add_argument("--only", default=None, help="Regex filter on case name.")
    args = parser.parse_args(argv)

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="env_cfg_dumps_"))
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, ref_glob, task, extra in CASES:
        if args.only and not re.search(args.only, name):
            continue
        matches = sorted(REPO.glob(ref_glob))
        if not matches:
            results.append({"name": name, "task": task, "status": "NO GROUND TRUTH", "ref": ref_glob})
            continue
        print(f"[INFO] {name}: dumping {task} ...", flush=True)
        results.append(run_case(name, matches[0], task, extra, workdir))

    print(f"\nLocal dumps written to {workdir}\n")
    print(f"{'case':<24} {'task':<40} status")
    print("-" * 100)
    failed = False
    for res in results:
        print(f"{res['name']:<24} {res['task']:<40} {res['status']}")
        for key_path, local, ref in res.get("allowed", []):
            print(f"    [allowlisted] {key_path}: local={local} ref={ref}")
        for key_path, local, ref in res.get("real", []):
            failed = True
            print(f"    [REAL DIFF]   {key_path}: local={local} ref={ref}")
        if res["status"] in ("DUMP FAILED", "NO GROUND TRUTH"):
            failed = failed or res["status"] == "DUMP FAILED"
            for line in res.get("detail", []):
                print(f"    {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
