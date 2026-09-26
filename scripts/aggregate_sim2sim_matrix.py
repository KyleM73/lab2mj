"""Aggregate sim2sim validation reports into one matrix table.

Scans directories containing ``report.json`` files produced by
``uv run lab2mj-validate`` and prints a markdown table (one row per
run) with the gate results, key metrics, and each row's gate
*utilization* (worst gated metric / its tolerance; 1.0 = exactly at the
gate), plus pass-count totals.

Comparing matrices: validation is bit-deterministic on one machine against a
fixed dump (measured: identical report.json across back-to-back runs), so
row-to-row changes come from (a) cross-machine float noise, visible as small
utilization shifts that flip PASS/FAIL only on gate-riding rows (marked *),
or (b) re-dumped references — PhysX GPU rollouts are not run-to-run
deterministic, so every re-dump is a different trajectory and
divergence-sensitive rows (rough terrain, aggressive commands) scatter.
Compare utilizations, and keep dumps pinned unless the dump protocol changed.

Usage::

    uv run python scripts/aggregate_sim2sim_matrix.py \\
        logs/validate_mj/matrix logs/validate_mj/measured [--out matrix.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# Utilization within this band of 1.0 is "gate-riding" (see module docstring).
GATE_RIDE_BAND = 0.02


def _gate_utilization(gate: dict[str, Any]) -> float | None:
    """Worst gated-metric fraction of its tolerance for an open/closed-loop parity result (1.0 = at the gate).

    PASS/FAIL is utilization <= 1 by construction; the margin to the gate is
    ``1 - utilization``, comparable across rows and machines.
    """
    metrics = gate.get("metrics") or {}
    dq, dz = metrics.get("max_joint_err_rad_gate_window"), metrics.get("max_root_dz_m_gate_window")
    dq_tol, dz_tol = gate.get("joint_tol_rad"), gate.get("root_dz_tol_m")
    if dq is None or dz is None or dq_tol is None or dz_tol is None:
        return None
    return max(float(dq) / float(dq_tol), float(dz) / float(dz_tol))


def _status(gate: dict[str, Any] | None) -> str:
    """PASS/FAIL for a gate the report ran; '-' for a gate absent from a partial run."""
    if not gate:
        return "-"
    return "PASS" if gate.get("passed") else "FAIL"


def _row(name: str, report: dict[str, Any]) -> dict[str, Any]:
    gates = report["gates"]
    # Old reports (pre-rename) use g0..g3 keys; read both so archived matrices still aggregate.
    obs = gates.get("obs") or gates.get("g0") or {}
    open_loop = gates.get("open_loop") or gates.get("g1") or {}
    closed_loop = gates.get("closed_loop") or gates.get("g2") or {}
    behavior = gates.get("behavior") if gates.get("behavior") is not None else gates.get("g3")
    open_loop_metrics = open_loop.get("metrics") or {}
    closed_loop_metrics = closed_loop.get("metrics") or {}
    behavior_metrics = (behavior or {}).get("metrics") or behavior
    return {
        "run": name,
        "obs": _status(obs if obs else None),
        "obs_err": max(
            (t.get("max_abs_err", 0.0) for t in (obs.get("terms") or []) if not t.get("informative", False)),
            default=None,
        ),
        "open_loop": _status(open_loop if open_loop else None),
        "open_loop_dq": open_loop_metrics.get("max_joint_err_rad_gate_window"),
        "open_loop_util": _gate_utilization(open_loop),
        "closed_loop": _status(closed_loop if closed_loop else None),
        "closed_loop_dq": closed_loop_metrics.get("max_joint_err_rad_gate_window"),
        "closed_loop_util": _gate_utilization(closed_loop),
        "stable": None if behavior is None else bool(behavior_metrics.get("stable", False)),
        "vel_ratio": (behavior_metrics or {}).get("vel_tracking_err_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", help="Directories scanned recursively for report.json files.")
    parser.add_argument("--out", default=None, help="Also write the markdown table to this path.")
    args = parser.parse_args()

    rows = []
    bundles: dict[str, list[dict[str, Any]]] = {}
    seen: dict[str, Path] = {}
    for root in args.dirs:
        for report_path in sorted(Path(root).rglob("report.json")):
            name = report_path.parent.name
            # Later directories on the command line supersede earlier ones for the
            # same run name (e.g. measured-terrain reruns replace procedural ones).
            seen[name] = report_path
    for name, report_path in sorted(seen.items()):
        report = json.loads(report_path.read_text())
        row = _row(name, report)
        rows.append(row)
        bundles.setdefault(Path(report.get("bundle", "?")).name, []).append(row)

    header = (
        "| run | obs | obs err | open loop | ol max dq | ol util "
        "| closed loop | cl max dq | cl util | stable | vel ratio |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]

    def util_cell(util: float | None) -> str:
        if util is None:
            return "-"
        ride = "*" if abs(util - 1.0) <= GATE_RIDE_BAND else ""
        return f"{util:.2f}{ride}"

    for r in rows:
        obs_err = "-" if r["obs_err"] is None else f"{r['obs_err']:.1e}"
        stable = "-" if r["stable"] is None else ("yes" if r["stable"] else "NO")
        lines.append(
            f"| {r['run']} | {r['obs']} | {obs_err} "
            f"| {r['open_loop']} | {_fmt(r['open_loop_dq'])} | {util_cell(r['open_loop_util'])} "
            f"| {r['closed_loop']} | {_fmt(r['closed_loop_dq'])} | {util_cell(r['closed_loop_util'])} "
            f"| {stable} | {_fmt(r['vel_ratio'], 2)} |"
        )

    # Per-gate totals count only the runs that actually ran the gate (partial --gates
    # runs record nothing for the others and must not read as failures).
    def total(key: str) -> str:
        ran = [r for r in rows if r[key] != "-"]
        return f"{sum(r[key] == 'PASS' for r in ran)}/{len(ran)}"

    riders = sum(
        1
        for r in rows
        for u in (r["open_loop_util"], r["closed_loop_util"])
        if u is not None and abs(u - 1.0) <= GATE_RIDE_BAND
    )
    ran_behavior = [r for r in rows if r["stable"] is not None]
    lines.append("")
    lines.append(
        f"Totals over {len(rows)} runs: obs {total('obs')}, open-loop {total('open_loop')}, "
        f"closed-loop {total('closed_loop')}, "
        f"stable {sum(bool(r['stable']) for r in ran_behavior)}/{len(ran_behavior)}. "
        f"Gate-riding cells (util within {GATE_RIDE_BAND:g} of 1.0, marked *): {riders} — "
        "their PASS/FAIL can flip on cross-machine float noise; compare utilizations instead."
    )
    # Per-bundle distribution rollup: with several reference trajectories per bundle,
    # the median is the transfer-quality signal and the max the worst case — a single
    # knife-edge trajectory should not decide a bundle's fate.
    lines.append("")
    lines.append("| bundle | dumps | open-loop pass | ol median dq | ol max dq | closed-loop pass |")
    lines.append("|---|---|---|---|---|---|")
    for bundle_name, brows in sorted(bundles.items()):
        ol_vals = [r["open_loop_dq"] for r in brows if r["open_loop_dq"] is not None]
        ol_pass = sum(1 for r in brows if r["open_loop"] == "PASS")
        cl_ran = [r for r in brows if r["closed_loop"] != "-"]
        cl_pass = sum(1 for r in cl_ran if r["closed_loop"] == "PASS")
        med = f"{float(np.median(ol_vals)):.4f}" if ol_vals else "-"
        worst = f"{max(ol_vals):.4f}" if ol_vals else "-"
        lines.append(
            f"| {bundle_name} | {len(brows)} | {ol_pass}/{len(brows)} | {med} | {worst} | {cl_pass}/{len(cl_ran)} |"
        )
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
