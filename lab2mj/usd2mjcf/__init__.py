"""USD -> MJCF conversion: :func:`parse_usd` -> :class:`RobotIR` -> :func:`build_mjcf`.

Submodules are imported lazily so that importing this package pulls in neither
``pxr`` (parser) nor ``mujoco`` (builder/verify) until actually used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "GeomIR": "parser",
    "JointIR": "parser",
    "LinkIR": "parser",
    "MeshData": "parser",
    "RobotIR": "parser",
    "parse_usd": "parser",
    "BuildResult": "builder",
    "build_mjcf": "builder",
    "VerifyError": "verify",
    "verify_build": "verify",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from lab2mj.usd2mjcf.builder import BuildResult, build_mjcf
    from lab2mj.usd2mjcf.parser import GeomIR, JointIR, LinkIR, MeshData, RobotIR, parse_usd
    from lab2mj.usd2mjcf.verify import VerifyError, verify_build


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
