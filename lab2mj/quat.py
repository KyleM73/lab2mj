"""Quaternion / angle helpers shared across lab2mj (wxyz convention, float64).

Single home for the scalar quaternion math the converter (``usd2mjcf``), the
runtime (``commands`` / ``heightscan`` / ``events``), and the validation gates
all use: numpy ports of ``isaaclab.utils.math``, hand-rolled instead of scipy
so every consumer runs bit-identical formulas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wrap_to_pi(angles: Any) -> Any:
    """Wrap angles (rad) to [-pi, pi]; +pi maps to +pi and -pi maps to -pi."""
    angles = np.asarray(angles, dtype=np.float64)
    wrapped = (angles + np.pi) % (2.0 * np.pi)
    out = np.where((wrapped == 0.0) & (angles > 0.0), np.pi, wrapped - np.pi)
    return float(out) if out.ndim == 0 else out


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """Normalize and fix the sign of a wxyz quaternion (first nonzero component > 0)."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    q = q / n
    for c in q:
        if abs(c) > 1e-12:
            return q if c > 0 else -q
    return q


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions) of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of the wxyz quaternion ``quat``."""
    xyz = np.asarray(quat[1:], dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v - quat[0] * t + np.cross(xyz, t)


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix R_wb from a wxyz quaternion (columns are body axes in world)."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix -> canonical wxyz quaternion (Shepperd's method)."""
    m = np.asarray(mat, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] >= m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return quat_canonical(q)


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ-convention Euler angles (rad) to a wxyz quaternion."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dtype=np.float64,
    )


def quat_angle_rad(quat_a_wxyz: np.ndarray, quat_b_wxyz: np.ndarray) -> np.ndarray:
    """Rotation angle (rad) between unit wxyz quaternion arrays ``(..., 4)``, sign-invariant."""
    dot = np.abs(np.sum(np.asarray(quat_a_wxyz) * np.asarray(quat_b_wxyz), axis=-1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def yaw_quat(quat: np.ndarray) -> np.ndarray:
    """Extract the yaw-only component of a wxyz quaternion."""
    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)
    return out / np.linalg.norm(out)


def heading_w_from_quat(quat: np.ndarray) -> float:
    """Yaw heading (rad) of a frame whose forward direction is body +x.

    Closed form of ``atan2(f_y, f_x)`` for the rotated forward vector
    ``f = R(quat) @ [1, 0, 0]`` — also the yaw angle ``yaw_quat`` extracts.
    """
    w, x, y, z = quat
    return math.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))
