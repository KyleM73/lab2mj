"""Post-build checks: compiled MuJoCo model vs the USD-derived :class:`RobotIR`."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from lab2mj.quat import quat_to_rotmat
from lab2mj.usd2mjcf.builder import BuildResult
from lab2mj.usd2mjcf.parser import RobotIR

_COM_SCALE_FLOOR = 1e-3  # meters; keeps the relative test meaningful for near-origin CoMs
_INERTIA_SCALE_FLOOR = 1e-9


class VerifyError(RuntimeError):
    """Raised when the compiled model disagrees with the parsed USD values."""


def verify_build(robot: RobotIR, built: BuildResult | mujoco.MjSpec | mujoco.MjModel, *, rtol: float = 1e-6) -> None:
    """Assert per-link mass/CoM/inertia, joint ranges/axes, and total mass agree.

    Raises :class:`VerifyError` with a diff table of every failing quantity.
    """
    if isinstance(built, BuildResult):
        model = built.spec.compile()
    elif isinstance(built, mujoco.MjSpec):
        model = built.compile()
    else:
        model = built

    failures: list[tuple[str, str, str, float]] = []

    def check(label: str, expected: np.ndarray, actual: np.ndarray, scale_floor: float = 1.0) -> None:
        expected = np.atleast_1d(np.asarray(expected, dtype=np.float64))
        actual = np.atleast_1d(np.asarray(actual, dtype=np.float64))
        err = float(np.linalg.norm(actual - expected))
        rel = err / max(float(np.linalg.norm(expected)), scale_floor)
        if not (rel <= rtol):
            failures.append((label, _fmt(expected), _fmt(actual), rel))

    for link in robot.links:
        body = model.body(link.name)
        check(f"body:{link.name} mass", np.array([link.mass]), body.mass)
        check(f"body:{link.name} com_b", link.com_b, body.ipos, scale_floor=_COM_SCALE_FLOOR)
        inertia_ir_b = _full_inertia(link.inertia_diag, link.inertia_quat_b)
        inertia_mj_b = _full_inertia(body.inertia, body.iquat)
        check(f"body:{link.name} inertia_b", inertia_ir_b.ravel(), inertia_mj_b.ravel(), _INERTIA_SCALE_FLOOR)

    for joint in robot.joints:
        joint_id = model.joint(joint.name).id
        check(f"joint:{joint.name} axis_b", joint.axis_b, model.jnt_axis[joint_id])
        if math.isfinite(joint.limit_lower) and math.isfinite(joint.limit_upper):
            expected_range = np.array([joint.limit_lower, joint.limit_upper])
            check(f"joint:{joint.name} range", expected_range, model.jnt_range[joint_id])
        elif model.jnt_limited[joint_id]:
            failures.append((f"joint:{joint.name} range", "unlimited", _fmt(model.jnt_range[joint_id]), math.inf))

    check("total mass", np.array([robot.total_mass]), np.array([float(model.body_mass.sum())]))

    if failures:
        header = f"{'quantity':<32} {'expected':<36} {'actual':<36} {'rel_err':>10}"
        rows = [f"{label:<32} {exp:<36} {act:<36} {rel:>10.3e}" for label, exp, act, rel in failures]
        raise VerifyError("model/USD mismatch:\n" + "\n".join([header, *rows]))


def _full_inertia(diag: np.ndarray, quat: np.ndarray) -> np.ndarray:
    rot = quat_to_rotmat(np.asarray(quat, dtype=np.float64))
    return rot @ np.diag(np.asarray(diag, dtype=np.float64)) @ rot.T


def _fmt(value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return value
    flat = np.asarray(value).ravel()
    body = " ".join(f"{v:.6g}" for v in flat[:4])
    return body + (" ..." if flat.size > 4 else "")
