"""Raw-MuJoCo conversion and runtime for IsaacLab-trained policies.

Submodules import lazily and never import ``isaaclab``; keep this module free of
heavy imports so it stays usable on CPU-only machines.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "MANIFEST_SCHEMA": "bundle",
    "read_manifest": "bundle",
    "write_manifest": "bundle",
    "convert_run": "convert",
    "MjEnv": "env",
    "run_validation": "validate",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from lab2mj.bundle import MANIFEST_SCHEMA, read_manifest, write_manifest
    from lab2mj.convert import convert_run
    from lab2mj.env import MjEnv
    from lab2mj.validate import run_validation


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
