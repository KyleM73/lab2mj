"""Build a :class:`mujoco.MjSpec` / MJCF bundle from a parsed :class:`RobotIR`."""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np

from lab2mj.usd2mjcf.parser import GeomIR, RobotIR, compose_frame

_GEOM_TYPE = {
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
}
_JOINT_TYPE = {
    "revolute": mujoco.mjtJoint.mjJNT_HINGE,
    "prismatic": mujoco.mjtJoint.mjJNT_SLIDE,
}
_JOINT_OVERRIDE_KEYS = ("armature", "frictionloss", "damping")
_VISUAL_GROUP = 2
_COLLISION_GROUP = 3

# PhysX enforces joint limits as hard position-level constraints (TGS): a drive torque
# commanding past the limit pins the joint EXACTLY at the range endpoint. Reproducing that
# in MuJoCo needs four things at once (verified on a single-hinge repro of a low-inertia
# hand joint: armature 1e-3, joint damping 10, 15-35 Nm stall torque):
#
# 1. Hard impedance. The constraint force is regularized by R = (1-d)/d * A, so the
#    equilibrium violation under stall torque tau is tau*(1-d)/(d*m_eff*k) regardless of
#    how stiff the direct solref is. At MuJoCo's default dmax=0.95 that is ~0.01-0.02 rad
#    for a low-inertia hand joint; at dmax=0.9999 (mjMAXIMP) it is ~1e-4 rad.
# 2. Overdamping. Hard impedance shrinks the equilibrium violation below the bounce
#    amplitude, so a critically damped reference (restitution ~0.4) exits the constraint;
#    the drive's free acceleration (tau/m_eff ~ 3e4 rad/s^2) slams the joint back in and a
#    sustained ~15 Hz limit cycle of ~0.02-0.05 rad amplitude forms. Authoring the pair
#    heavily overdamped (zeta = b/(2*sqrt(k)) = 3) kills restitution: settling is
#    monotone and chatter-free.
# 3. Timestep-scaled gains. The constraint damping term is explicit in the current
#    velocity, so discrete stability bounds b*dt <= ~0.75: b = 0.75/dt, k = (b/(2*zeta))^2.
# 4. An activation margin. Without one, a joint resting exactly at the endpoint (the
#    PhysX steady state) has no active constraint row, so the drive gets one substep of
#    free flight (a0*dt ~ 25 rad/s) before the limit reacts, and the overdamped pair's
#    slow pole (k/b) takes ~50 ms to recover the kick. margin = width = 400*dt^2 keeps
#    the row active inside the band (impedance ramps _LIMIT_D0 -> dmax across it, so the
#    skin is gentle for joints merely passing near the limit) — the PhysX-limit analogue
#    of a contact distance. A stalled joint then rests ~1e-5 rad INSIDE the endpoint at
#    the substeps=4 timestep of 1.25 ms (vs ~1e-2 rad past it with MuJoCo's default
#    impedance), and the band also absorbs the per-substep kick at coarse timesteps, so
#    substeps=1 conversions stay chatter-free too.
_LIMIT_ZETA = 3.0  # overdamping ratio of the direct (-stiffness, -damping) solref pair
_LIMIT_DAMPING_DT = 0.75  # b * dt: discrete-stability bound for the explicit damping term
_LIMIT_MARGIN_DT2 = 400.0  # (margin = solimp width) / dt^2: activation band before the endpoint
_LIMIT_D0 = 0.001  # solimp impedance at the band surface (gentle first touch)
_LIMIT_DMAX = 0.9999  # solimp impedance at full band depth (mjMAXIMP: hardest allowed)

# Contact profiles. MuJoCo's default contact is hard at first touch (solimp d0 = 0.9:
# ~90% impedance at zero penetration, pyramidal friction cone), while PhysX engages a
# contact progressively — TGS ramps the response over the first fraction of a
# millimeter of penetration, and its patch friction resolves sliding as an isotropic
# Coulomb cone. On robots that resolve ground forces through compliant leg chains
# (soft PD gains, light legs — e.g. 13-15 kg quadrupeds at kp ~= 25 whose policies
# slide their feet in near-continuous contact), that difference dominates sim2sim
# error: at matched states against Isaac reference dumps the MuJoCo default produces
# 20-50% more normal force per foot, and the strict open-loop replay gate improves
# 4-6x under the "engagement" profile below. Heavier, stiffer robots (high-kp PD,
# series-elastic actuator-net LSTMs) track their references better with the MuJoCo
# default, so "engagement" is opt-in per robot (see
# ``lab2mj.actuators.ActuatorSet.contact_profile``).
#
# The profile authors three things:
# 1. Elliptic friction cone — matches PhysX slide friction (a foot sliding diagonally
#    sees mu * N, not the pyramid's inflated diagonal bound) and removes the pyramid's
#    normal-force coupling under tangential demand.
# 2. impratio 3 — friction rows 3x harder than normal rows, PhysX solves friction at
#    every TGS iteration so its stick constraint is comparatively stiffer.
# 3. A progressive-engagement solimp: impedance ramps from ~0 at the contact surface
#    to mjMAXIMP at 1 mm penetration (quadratic ramp, midpoint 0.5), so touchdown
#    force builds over the skin instead of arriving as a first-touch spike.
# PhysX contact_offset / rest_offset intentionally do NOT convert: the shipped assets
# author neither (PhysX auto-computes contact_offset; rest_offset defaults to 0, i.e.
# force only at penetration — MuJoCo's native behavior), and MuJoCo margin/gap cannot
# express speculative contacts anyway (measured: margin == gap is dynamically inert
# because inactive contacts never enter the solver, and margin alone applies force at
# a distance, which PhysX does not).
_ENGAGEMENT_SOLIMP = (0.001, 0.9999, 0.001, 0.5, 2.0)
_ENGAGEMENT_IMPRATIO = 3.0
CONTACT_PROFILES = ("default", "engagement")


def _limit_params(physics_dt: float) -> tuple[list[float], list[float], float]:
    """(solref, solimp, margin) authoring a PhysX-style hard joint limit at ``physics_dt``."""
    damping = _LIMIT_DAMPING_DT / physics_dt
    stiffness = (damping / (2.0 * _LIMIT_ZETA)) ** 2
    margin = _LIMIT_MARGIN_DT2 * physics_dt * physics_dt
    return [-stiffness, -damping], [_LIMIT_D0, _LIMIT_DMAX, margin, 0.5, 2.0], margin


@dataclasses.dataclass
class BuildResult:
    spec: mujoco.MjSpec
    xml: str
    assets: dict[str, bytes]  # bundle-relative paths ("assets/<name>.obj") -> file bytes


def build_mjcf(
    robot: RobotIR,
    *,
    physics_dt: float,
    model_name: str | None = None,
    joint_overrides: dict[str, dict[str, float]] | None = None,
    solref: tuple[float, float] | None = None,
    solimp: tuple[float, ...] | None = None,
    impratio: float | None = None,
    contact_profile: str = "default",
    default_qpos: dict[str, float] | None = None,
    root_pos_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    root_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    gravity_w: tuple[float, float, float] | None = None,
) -> BuildResult:
    """Build a compiled MjSpec + deterministic MJCF text + OBJ assets from ``robot``.

    ``physics_dt`` is the timestep the model integrates at (the caller's MuJoCo substep
    dt, not necessarily the Isaac ``sim.dt``); the model timestep, contact solref, and
    joint-limit constraint gains are all authored against it, so it has no default —
    every caller must pass the timestep it will actually step the model with.
    ``joint_overrides`` maps joint name -> {"armature" | "frictionloss" | "damping": value}
    (unspecified values stay 0, matching Isaac's actuator-cfg-driven joint params).
    ``solref`` defaults to ``(2 * physics_dt, 1.0)`` — the stiffest stable reference at
    the caller's physics timestep. PhysX contacts are near-rigid, so the default tracks
    the timestep instead of inheriting MuJoCo's softer ``(0.02, 1)``.
    ``contact_profile`` selects the contact model (see the profile comment block
    above). Explicit ``solimp`` / ``impratio`` arguments override the profile's
    values (calibration output); the friction cone stays profile-driven.
    ``gravity_w`` authors ``option/gravity`` (``None`` keeps MuJoCo's default
    ``(0, 0, -9.81)``); pass the source env's ``sim.gravity`` so the compiled model
    matches Isaac.
    """
    if contact_profile not in CONTACT_PROFILES:
        raise ValueError(f"unknown contact_profile {contact_profile!r}; expected one of {CONTACT_PROFILES}")
    joint_overrides = joint_overrides or {}
    default_qpos = default_qpos or {}
    for joint_name, overrides in joint_overrides.items():
        unknown = set(overrides) - set(_JOINT_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(f"joint_overrides[{joint_name!r}]: unknown keys {sorted(unknown)}")

    spec = mujoco.MjSpec()
    spec.modelname = model_name or robot.name
    spec.compiler.degree = False
    spec.compiler.meshdir = "assets"
    spec.option.timestep = physics_dt
    if gravity_w is not None:
        spec.option.gravity = [float(g) for g in gravity_w]
    if contact_profile == "engagement":
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = _ENGAGEMENT_IMPRATIO
    if impratio is not None:
        spec.option.impratio = float(impratio)

    spec.default.geom.solref = list(solref) if solref is not None else [2.0 * physics_dt, 1.0]
    if solimp is not None:
        spec.default.geom.solimp = list(solimp)
    elif contact_profile == "engagement":
        spec.default.geom.solimp = list(_ENGAGEMENT_SOLIMP)

    limit_solref, limit_solimp, limit_margin = _limit_params(physics_dt)
    joint_of_child = {joint.child: joint for joint in robot.joints}
    geoms_of_link: dict[str, list[GeomIR]] = {link.name: [] for link in robot.links}
    for geom in robot.geoms:
        geoms_of_link[geom.link].append(geom)

    assets: dict[str, bytes] = {}
    mjs_bodies: dict[str, mujoco.MjsBody] = {}
    mj_joint_names: list[str] = []
    conaffinity = 1 if robot.self_collisions_enabled else 0

    root_pos_home = np.asarray(root_pos_w, dtype=np.float64)
    root_quat_home = np.asarray(root_quat_w, dtype=np.float64)
    for link in robot.links:
        if link.parent is None:
            root_pos_home, root_quat_home = compose_frame(
                np.asarray(root_pos_w, dtype=np.float64),
                np.asarray(root_quat_w, dtype=np.float64),
                link.pos_p,
                link.quat_p,
            )
            body = spec.worldbody.add_body(name=link.name, pos=root_pos_home, quat=root_quat_home)
            free = body.add_freejoint()
            free.name = "freejoint"
        else:
            body = mjs_bodies[link.parent].add_body(name=link.name, pos=link.pos_p, quat=link.quat_p)
            joint = joint_of_child[link.name]
            mjs_joint = body.add_joint(
                name=joint.name,
                type=_JOINT_TYPE[joint.kind],
                pos=joint.pos_b,
                axis=joint.axis_b,
            )
            lower_finite, upper_finite = np.isfinite(joint.limit_lower), np.isfinite(joint.limit_upper)
            if lower_finite and upper_finite:
                mjs_joint.range = [joint.limit_lower, joint.limit_upper]
                mjs_joint.solref_limit = limit_solref
                mjs_joint.solimp_limit = limit_solimp
                mjs_joint.margin = limit_margin
            elif lower_finite or upper_finite:
                raise ValueError(
                    f"joint {joint.name}: one-sided limit [{joint.limit_lower}, {joint.limit_upper}] "
                    "is unsupported (MuJoCo ranges need both bounds; PhysX would enforce the single bound)"
                )
            overrides = joint_overrides.get(joint.name, {})
            mjs_joint.armature = float(overrides.get("armature", 0.0))
            mjs_joint.frictionloss = float(overrides.get("frictionloss", 0.0))
            mjs_joint.damping[0] = float(overrides.get("damping", 0.0))
            mj_joint_names.append(joint.name)
        mjs_bodies[link.name] = body

        body.explicitinertial = True
        body.mass = link.mass
        body.ipos = link.com_b
        body.iquat = link.inertia_quat_b
        body.inertia = link.inertia_diag

        for geom in geoms_of_link[link.name]:
            mjs_geom = body.add_geom(
                name=geom.name,
                type=_GEOM_TYPE[geom.kind],
                pos=geom.pos_b,
                quat=geom.quat_b,
            )
            if geom.kind == "mesh":
                assert geom.mesh is not None
                spec.add_mesh(name=geom.name, file=f"{geom.name}.obj")
                mjs_geom.meshname = geom.name
                assets[f"assets/{geom.name}.obj"] = _obj_bytes(geom.mesh.vertices_b, geom.mesh.faces)
            else:
                mjs_geom.size = geom.size
            if geom.is_collision:
                mjs_geom.contype = 1
                mjs_geom.conaffinity = conaffinity
                mjs_geom.group = _COLLISION_GROUP
            else:
                mjs_geom.contype = 0
                mjs_geom.conaffinity = 0
                mjs_geom.group = _VISUAL_GROUP

    for body1, body2 in robot.filtered_pairs:
        spec.add_exclude(name=f"{body1}_{body2}", bodyname1=body1, bodyname2=body2)

    for joint_name in mj_joint_names:
        actuator = spec.add_actuator(name=joint_name, target=joint_name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        actuator.gainprm[0] = 1.0  # gaintype=fixed, biastype=none: ctrl is joint torque

    qpos_home = [*root_pos_home, *root_quat_home] + [float(default_qpos.get(name, 0.0)) for name in mj_joint_names]
    spec.add_key(name="home", qpos=qpos_home)

    spec.assets = {path.rpartition("/")[2]: data for path, data in assets.items()}
    spec.compile()
    return BuildResult(spec=spec, xml=spec.to_xml(), assets=assets)


def _obj_bytes(vertices_b: np.ndarray, faces: np.ndarray) -> bytes:
    lines = [f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in vertices_b]
    lines += [f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}" for f in faces]
    return ("\n".join(lines) + "\n").encode()
