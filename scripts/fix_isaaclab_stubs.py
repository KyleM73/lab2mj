#!/usr/bin/env python3
"""Replace stub-only IsaacLab packages in site-packages with symlinks to the full source.

IsaacLab's setup.py uses ``packages=["isaaclab"]`` (top-level only), so pip/uv
installs just ``__init__.py`` — no submodules — and ``import isaaclab`` fails on the
missing ``config/extension.toml``. This script finds the full source in uv's git
cache and symlinks it into the venv so ``lab2mj-dump-env-cfg`` (isaac_shims), the
tests that use Isaac Lab's config layer, and type checkers can resolve it.

Usage:
    python scripts/fix_isaaclab_stubs.py        # auto-detect venv + cache
    python scripts/fix_isaaclab_stubs.py --dry  # preview without changes

Re-run after every ``uv sync --group isaac`` or ``uv cache clean`` (``make sync`` does).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PACKAGES = ["isaaclab", "isaaclab_assets", "isaaclab_tasks"]


def find_site_packages(venv: Path) -> Path:
    """Return the site-packages directory inside *venv*."""
    candidates = sorted(venv.glob("lib/python*/site-packages"))
    if not candidates:
        sys.exit(f"No site-packages found in {venv}")
    return candidates[-1]


def find_cache_source(site_packages: Path, pkg: str) -> Path | None:
    """Resolve the full source tree for *pkg* via its dist-info direct_url.json."""
    dist_infos = list(site_packages.glob(f"{pkg}-*.dist-info"))
    if not dist_infos:
        return None
    direct_url = dist_infos[0] / "direct_url.json"
    if not direct_url.exists():
        return None
    meta = json.loads(direct_url.read_text())
    vcs_info = meta.get("vcs_info", {})
    commit = vcs_info.get("commit_id", "")
    subdir = meta.get("subdirectory", "")
    if not commit:
        return None
    # uv stores git checkouts under ~/.cache/uv/git-v0/checkouts/<repo-hash>/<commit-prefix>/
    cache_root = Path.home() / ".cache" / "uv" / "git-v0" / "checkouts"
    if not cache_root.exists():
        return None
    for repo_dir in cache_root.iterdir():
        candidate = repo_dir / commit[:8] / subdir / pkg
        if candidate.is_dir() and any(candidate.iterdir()):
            # Verify it has actual submodules, not just __init__.py
            children = {p.name for p in candidate.iterdir()} - {"__init__.py", "__pycache__"}
            if children:
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry", action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    venv = Path(__file__).resolve().parent.parent / ".venv"
    if not venv.exists():
        sys.exit(f"Virtualenv not found at {venv}. Run 'uv sync --group isaac' first.")

    site_packages = find_site_packages(venv)
    print(f"site-packages: {site_packages}")

    for pkg in PACKAGES:
        installed = site_packages / pkg
        source = find_cache_source(site_packages, pkg)

        if source is None:
            print(f"  {pkg}: SKIP (not installed or cache miss)")
            continue

        if installed.is_symlink():
            target = installed.resolve()
            if target == source.resolve():
                print(f"  {pkg}: OK (already linked)")
                continue
            print(f"  {pkg}: RE-LINK {source}")
        else:
            print(f"  {pkg}: LINK {source}")

        if args.dry:
            continue

        # Remove stub and create symlink
        if installed.is_symlink() or installed.is_file():
            installed.unlink()
        elif installed.is_dir():
            shutil.rmtree(installed)
        installed.symlink_to(source)

    # IsaacLab's __init__.py expects config/extension.toml one level above the
    # package directory (i.e. in site-packages/).  The git source doesn't
    # include it in the pip distribution, so create it if missing.
    config_file = site_packages / "config" / "extension.toml"
    if not config_file.exists():
        # Try to find it next to the cached source
        isaaclab_source = find_cache_source(site_packages, "isaaclab")
        src_config = isaaclab_source.parent / "config" / "extension.toml" if isaaclab_source else None
        if src_config and src_config.exists():
            if not args.dry:
                config_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_config, config_file)
            print("  config/extension.toml: COPIED from cache")
        else:
            # Write a minimal version so `import isaaclab` works
            if not args.dry:
                config_file.parent.mkdir(parents=True, exist_ok=True)
                config_file.write_text(
                    '[package]\nversion = "0.47.2"\ntitle = "Isaac Lab"\n'
                    'description = "Isaac Lab framework"\n'
                    'repository = "https://github.com/isaac-sim/IsaacLab"\n'
                )
            print("  config/extension.toml: CREATED (minimal)")
    else:
        print("  config/extension.toml: OK")

    print("\nDone." if not args.dry else "\nDry run — no changes made.")


if __name__ == "__main__":
    main()
