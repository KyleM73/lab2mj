"""Parse a robot articulation USD into a plain :class:`RobotIR`.

Uses only ``usd-core`` (imported lazily inside :func:`parse_usd`) plus numpy, so the
module can be imported on machines without ``pxr`` installed.

Frame suffixes follow the repo convention: ``_b`` = link (body-origin) frame,
``_w`` = world. ``_p`` additionally denotes the *parent link's* frame, used for the
static link placement at the zero joint configuration. Quaternions are wxyz
throughout (USD ``Gf.Quat`` and MuJoCo agree on this order).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path

import numpy as np

from lab2mj.quat import mat_to_quat, quat_canonical, quat_conj, quat_mul, quat_to_rotmat

_SQ2 = math.sqrt(0.5)
_GPRIM_TYPES = ("Mesh", "Sphere", "Cube", "Capsule", "Cylinder")
_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
# Quaternion rotating MuJoCo's canonical primitive axis (+Z) onto the USD axis token.
_AXIS_QUAT = {
    "X": np.array([_SQ2, 0.0, _SQ2, 0.0]),
    "Y": np.array([_SQ2, -_SQ2, 0.0, 0.0]),
    "Z": np.array([1.0, 0.0, 0.0, 0.0]),
}


# ---------------------------------------------------------------------------
# Frame helpers (wxyz, float64; quaternion math from lab2mj.quat).
# ---------------------------------------------------------------------------


def compose_frame(
    pos_a: np.ndarray, quat_a: np.ndarray, pos_b: np.ndarray, quat_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose rigid transforms: T_a ∘ T_b (apply T_b first, expressed in T_a's frame)."""
    return pos_a + quat_to_rotmat(quat_a) @ pos_b, quat_canonical(quat_mul(quat_a, quat_b))


def invert_frame(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qi = quat_conj(quat)
    return -(quat_to_rotmat(qi) @ pos), quat_canonical(qi)


def principal_inertia(
    inertia_full: np.ndarray, *, link_name: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full 3x3 inertia (about CoM) -> (compilable moments, principal-axes wxyz quat, exact moments).

    Principal moments of any rigid body satisfy the triangle inequality
    (each moment <= sum of the other two). PhysX accepts and simulates authored
    inertias that violate it; MuJoCo's COMPILER refuses them, so the first
    return value projects the offending moment onto the boundary (largest = sum
    of the other two) with a warning. The runtime integrates such tensors fine
    — the third return value carries the exact (unprojected) moments, which the
    converter records so :func:`lab2mj.env.stamp_exact_inertia` can
    restore them on the loaded model (measured 23x on a boundary-degenerate
    toe inertia, 0.52 rad of free-space plant error inside 0.5 s).
    """
    inertia_full = 0.5 * (inertia_full + inertia_full.T)
    eigval, eigvec = np.linalg.eigh(inertia_full)
    order = [2, 1, 0]  # eigh returns ascending; MuJoCo stores descending
    diag = np.maximum(eigval[order], 0.0)
    rot = eigvec[:, order]
    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]
    exact = diag.copy()
    if diag[0] > diag[1] + diag[2] and diag[0] > 0.0:
        corrected = diag[1] + diag[2]
        warnings.warn(
            f"link {link_name or '<unnamed>'}: authored inertia violates the rigid-body triangle inequality "
            f"(principal moments {diag.tolist()}); scene.xml carries the largest moment reduced "
            f"{diag[0]:.3e} -> {corrected:.3e} (MuJoCo cannot compile it); the exact PhysX-simulated "
            "value is recorded in the manifest and stamped back at load",
            stacklevel=2,
        )
        diag = np.array([corrected, diag[1], diag[2]])
    return diag, mat_to_quat(rot), exact


# ---------------------------------------------------------------------------
# IR dataclasses.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MeshData:
    """Triangle mesh baked into the owning link's frame (scale and xforms applied)."""

    vertices_b: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int64


@dataclasses.dataclass
class GeomIR:
    name: str
    link: str
    source_body: str  # pre-weld owning body (weld folding re-parents ``link``, never this)
    kind: str  # "mesh" | "sphere" | "box" | "capsule" | "cylinder"
    is_collision: bool
    pos_b: np.ndarray  # (3,) geom frame origin in link frame
    quat_b: np.ndarray  # (4,) wxyz geom frame orientation in link frame
    size: np.ndarray  # (3,) MuJoCo-style sizes; zeros for meshes
    mesh: MeshData | None = None


@dataclasses.dataclass
class JointIR:
    """Articulated joint. Frame quantities are expressed in the *child* link frame."""

    name: str
    parent: str
    child: str
    kind: str  # "revolute" | "prismatic"
    pos_b: np.ndarray  # (3,) joint anchor in child link frame
    axis_b: np.ndarray  # (3,) unit motion axis in child link frame
    limit_lower: float  # radians (revolute) / meters (prismatic); -inf when unlimited
    limit_upper: float  # radians (revolute) / meters (prismatic); +inf when unlimited
    # Authored physxJoint:armature (None when unauthored). PhysX simulates this value
    # whenever the actuator cfg does not override it (IsaacLab armature=None semantics);
    # the converter uses it as the default for such joints.
    armature_usd: float | None = None


@dataclasses.dataclass
class LinkIR:
    name: str
    parent: str | None  # None for the root link
    pos_p: np.ndarray  # (3,) link origin in parent link frame at zero configuration
    quat_p: np.ndarray  # (4,) wxyz link orientation in parent link frame
    mass: float
    com_b: np.ndarray  # (3,) center of mass in link frame
    inertia_diag: np.ndarray  # (3,) principal moments about the CoM, descending (MuJoCo-compilable)
    inertia_quat_b: np.ndarray  # (4,) wxyz principal-axes orientation in link frame
    # (3,) exact moments; None means identical to inertia_diag (only projected links differ).
    inertia_diag_exact: np.ndarray | None = None


@dataclasses.dataclass
class RobotIR:
    """Parsed articulation: fixed joints already welded into their parent links."""

    name: str
    links: list[LinkIR]  # depth-first, root first
    joints: list[JointIR]  # depth-first (MJCF) order, aligned with links[1:]
    geoms: list[GeomIR]  # grouped by link in depth-first link order
    isaac_joint_order: list[str]  # PhysX breadth-first joint enumeration
    self_collisions_enabled: bool
    filtered_pairs: list[tuple[str, str]]  # sorted link-name pairs from USD filteredPairs
    isaac_body_names: list[str] = dataclasses.field(default_factory=list)
    """PhysX breadth-first body enumeration, PRE-weld (includes bodies welded away)."""
    weld_parent: dict[str, str] = dataclasses.field(default_factory=dict)
    """Welded child body name -> direct parent body name (chains left unresolved)."""

    @property
    def root_link(self) -> LinkIR:
        return self.links[0]

    @property
    def total_mass(self) -> float:
        return float(sum(link.mass for link in self.links))


# ---------------------------------------------------------------------------
# Internal raw records.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _RawBody:
    path: str
    name: str
    mass: float
    com_b: np.ndarray
    inertia_b: np.ndarray  # 3x3 about CoM, link frame
    geoms: list[GeomIR] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _RawJoint:
    path: str
    name: str
    order: int  # stage traversal (declaration) index — PhysX BFS tie-break
    kind: str  # "revolute" | "prismatic" | "fixed"
    body0: str
    body1: str
    pos0_p: np.ndarray  # joint frame in the parent (body0) link frame
    quat0_p: np.ndarray
    pos1_b: np.ndarray  # joint frame in the child (body1) link frame
    quat1_b: np.ndarray
    axis: str = "Z"
    limit_lower: float = -math.inf
    limit_upper: float = math.inf
    armature_usd: float | None = None  # authored physxJoint:armature (PhysX simulates it)


def _gf_quat_to_np(gf_quat) -> np.ndarray:
    imag = gf_quat.GetImaginary()
    q = np.array([gf_quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)
    if np.linalg.norm(q) < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_canonical(q)


def _vec_or_zero(value, scale: float = 1.0) -> np.ndarray:
    if value is None:
        return np.zeros(3)
    vec = np.array(value, dtype=np.float64)
    if not np.all(np.isfinite(vec)):
        return np.zeros(3)
    return vec * scale


def _mass_vec_or_warn(attr, scale: float, path: str, what: str) -> np.ndarray:
    """Authored finite MassAPI vector attribute, or zeros with a warning naming the prim.

    PhysX derives unauthored (or non-finite sentinel) CoM/inertia from collision geometry;
    the parser does not, so falling back to zeros can silently diverge from PhysX.
    """
    if attr.HasAuthoredValue():
        vec = np.array(attr.Get(), dtype=np.float64)
        if np.all(np.isfinite(vec)):
            return vec * scale
    warnings.warn(
        f"rigid body {path}: physics:{what} has no finite authored value; assuming zeros "
        "(geometry-derived mass properties are unsupported)",
        stacklevel=3,
    )
    return np.zeros(3)


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Fan-triangulate polygon faces given faceVertexCounts / faceVertexIndices."""
    faces: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        for k in range(1, count - 1):
            faces.append((indices[offset], indices[offset + k], indices[offset + k + 1]))
        offset += count
    return np.array(faces, dtype=np.int64)


def _shift_inertia(inertia_com: np.ndarray, offset_b: np.ndarray, mass: float) -> np.ndarray:
    """Parallel-axis shift of a CoM inertia to a point displaced by ``offset_b``."""
    d = np.asarray(offset_b, dtype=np.float64)
    return inertia_com + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


# ---------------------------------------------------------------------------
# Parsing.
# ---------------------------------------------------------------------------


def parse_usd(
    usd_path: str | Path,
    *,
    robot_name: str | None = None,
    physx_plant: dict[str, tuple[float, np.ndarray, np.ndarray]] | None = None,
) -> RobotIR:
    """Parse a robot USD into :class:`RobotIR`.

    Requires authored ``physics:mass`` on every rigid body (accumulating mass from
    geometry is unsupported). Fixed joints are welded: the child link is merged into
    its parent with combined mass/CoM/inertia, matching PhysX articulation behavior.

    ``physx_plant`` maps body name -> ``(mass, com_b (3,), inertia_b (3, 3))``
    (link frame, inertia about the CoM) as resolved by PhysX — recorded in
    reference dumps. When given, it replaces the authored mass properties BEFORE
    welding: PhysX validates authored inertias and silently recomputes invalid
    ones (measured 23x on a boundary-degenerate authored toe inertia), so the
    authored values are not always the plant the reference ran with. Bodies
    absent from the mapping keep their authored values.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise FileNotFoundError(f"could not open USD stage: {usd_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if up_axis not in ("Y", "Z"):
        raise ValueError(f"unsupported stage upAxis {up_axis!r}")
    mpu = meters_per_unit
    # Stage mass unit (PhysX honors it per the UsdPhysics schema): masses scale by kpu,
    # inertias by kpu * mpu^2. Kilogram stages (kpu == 1) are unaffected.
    kpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage))

    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bodies: dict[str, _RawBody] = {}
    raw_joints: list[_RawJoint] = []
    art_root_prims = []
    filtered_raw: list[tuple[str, str]] = []

    # Instance proxies are essential: Isaac robot USDs hide meshes behind instanceable
    # prims, invisible to a default stage traversal.
    for order, prim in enumerate(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())):
        path = str(prim.GetPath())
        type_name = prim.GetTypeName()

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root_prims.append(prim)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            rel = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel()
            filtered_raw.extend((path, str(target)) for target in rel.GetTargets())

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = _parse_body(prim, UsdPhysics, mpu, kpu)
            if physx_plant is not None and body.name in physx_plant:
                mass, com_b, inertia_b = physx_plant[body.name]
                inertia_b = np.asarray(inertia_b, dtype=np.float64)
                rel = float(np.linalg.norm(inertia_b - body.inertia_b)) / max(
                    float(np.linalg.norm(body.inertia_b)), 1e-12
                )
                if rel > 0.01:
                    warnings.warn(
                        f"rigid body {body.name}: PhysX-resolved inertia differs from the authored value "
                        f"by {rel:.0%} (PhysX recomputes invalid authored inertias); authoring the PhysX plant",
                        stacklevel=2,
                    )
                body.mass = float(mass)
                body.com_b = np.asarray(com_b, dtype=np.float64)
                body.inertia_b = inertia_b
            bodies[path] = body
        elif type_name.startswith("Physics") and type_name.endswith("Joint"):
            joint = _parse_joint(prim, order, UsdPhysics, mpu)
            if joint is not None:
                raw_joints.append(joint)
        elif type_name in _GPRIM_TYPES:
            geom = _parse_geom(prim, bodies, xf_cache, Gf, UsdGeom, UsdPhysics, mpu)
            if geom is not None:
                body_path, geom_ir = geom
                bodies[body_path].geoms.append(geom_ir)

    if not bodies:
        raise ValueError(f"no UsdPhysics rigid bodies found in {usd_path}")

    for joint in raw_joints:
        for endpoint in (joint.body0, joint.body1):
            if endpoint not in bodies:
                raise ValueError(f"joint {joint.path} targets {endpoint}, which is not a parsed rigid body")

    isaac_body_names, weld_parent = _isaac_body_enumeration(bodies, raw_joints)
    fold_map = _weld_fixed_joints(bodies, raw_joints)
    articulated = [j for j in raw_joints if j.kind != "fixed"]
    alive = {path: body for path, body in bodies.items() if path not in fold_map}

    names = [body.name for body in alive.values()]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate link names after welding: {sorted(names)}")

    out_joints: dict[str, list[_RawJoint]] = {path: [] for path in alive}
    root_paths = set(alive)
    for joint in articulated:
        parent_path = _resolve(joint.body0, fold_map)
        out_joints[parent_path].append(joint)
        root_paths.discard(joint.body1)
    for joints_of_body in out_joints.values():
        joints_of_body.sort(key=lambda j: j.order)
    if len(root_paths) != 1:
        raise ValueError(
            f"expected exactly one articulation root link, found {sorted(root_paths)} — a body with no "
            "enabled joint to the tree (e.g. attached via a DISABLED joint, which PhysX simulates as a "
            "free rigid body) is not convertible as part of the articulation"
        )
    root_path = next(iter(root_paths))

    links, joints, geoms = _finalize_tree(alive, out_joints, root_path, up_axis)
    isaac_joint_order = _bfs_joint_order(out_joints, root_path)

    name_of = {path: body.name for path, body in alive.items()}
    pairs: set[tuple[str, str]] = set()
    for source_path, target_path in filtered_raw:
        for source in _resolve_filtered(source_path, bodies, fold_map):
            for target in _resolve_filtered(target_path, bodies, fold_map):
                if source == target:
                    continue
                name_a, name_b = name_of[source], name_of[target]
                pairs.add((name_a, name_b) if name_a <= name_b else (name_b, name_a))

    self_collisions = False
    for prim in art_root_prims:
        attr = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if attr and attr.Get():
            self_collisions = True

    if robot_name is None:
        default_prim = stage.GetDefaultPrim()
        robot_name = default_prim.GetName() if default_prim else Path(str(usd_path)).stem

    if not any(g.is_collision for g in geoms):
        unresolved = [str(dep) for dep in stage.GetRootLayer().GetExternalReferences() if dep]
        raise ValueError(
            f"robot USD {usd_path} produced no collision geoms — a robot that cannot collide is never valid. "
            + (
                f"The root layer references external assets {unresolved}; if any failed to resolve "
                "(watch for pxr warnings above), collision/visual meshes live in those layers."
                if unresolved
                else "Check that collision prims carry UsdPhysics.CollisionAPI."
            )
        )

    return RobotIR(
        name=robot_name,
        links=links,
        joints=joints,
        geoms=geoms,
        isaac_joint_order=isaac_joint_order,
        self_collisions_enabled=self_collisions,
        filtered_pairs=sorted(pairs),
        isaac_body_names=isaac_body_names,
        weld_parent=weld_parent,
    )


def _parse_body(prim, UsdPhysics, mpu: float, kpu: float) -> _RawBody:
    path = str(prim.GetPath())
    mass_api = UsdPhysics.MassAPI(prim)
    if not prim.HasAPI(UsdPhysics.MassAPI) or not mass_api.GetMassAttr().HasAuthoredValue():
        raise ValueError(
            f"rigid body {path} has no authored physics:mass; accumulating mass from geometry is unsupported"
        )
    mass = float(mass_api.GetMassAttr().Get()) * kpu
    if mass <= 0:
        raise ValueError(f"rigid body {path} has non-positive mass {mass}")
    com_b = _mass_vec_or_warn(mass_api.GetCenterOfMassAttr(), mpu, path, "centerOfMass")
    inertia_scale = kpu * mpu * mpu
    diag = np.maximum(_mass_vec_or_warn(mass_api.GetDiagonalInertiaAttr(), inertia_scale, path, "diagonalInertia"), 0.0)
    paxes = mass_api.GetPrincipalAxesAttr().Get()
    quat = _gf_quat_to_np(paxes) if paxes is not None else np.array([1.0, 0.0, 0.0, 0.0])
    rot = quat_to_rotmat(quat)
    inertia_b = rot @ np.diag(diag) @ rot.T
    return _RawBody(path=path, name=prim.GetName(), mass=mass, com_b=com_b, inertia_b=inertia_b)


_JOINT_KIND = {
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsFixedJoint": "fixed",
}


def _parse_joint(prim, order: int, UsdPhysics, mpu: float) -> _RawJoint | None:
    path = str(prim.GetPath())
    type_name = prim.GetTypeName()
    if type_name not in _JOINT_KIND:
        raise NotImplementedError(f"unsupported joint type {type_name} at {path}")

    joint_api = UsdPhysics.Joint(prim)
    if joint_api.GetJointEnabledAttr().Get() is False:
        return None
    targets0 = joint_api.GetBody0Rel().GetTargets()
    targets1 = joint_api.GetBody1Rel().GetTargets()
    if not targets0 or not targets1:
        raise NotImplementedError(f"joint {path} lacks a body0/body1 target (world-anchored joints unsupported)")

    lr0 = joint_api.GetLocalRot0Attr().Get()
    lr1 = joint_api.GetLocalRot1Attr().Get()
    raw = _RawJoint(
        path=path,
        name=prim.GetName(),
        order=order,
        kind=_JOINT_KIND[type_name],
        body0=str(targets0[0]),
        body1=str(targets1[0]),
        pos0_p=_vec_or_zero(joint_api.GetLocalPos0Attr().Get(), mpu),
        quat0_p=_gf_quat_to_np(lr0) if lr0 is not None else np.array([1.0, 0.0, 0.0, 0.0]),
        pos1_b=_vec_or_zero(joint_api.GetLocalPos1Attr().Get(), mpu),
        quat1_b=_gf_quat_to_np(lr1) if lr1 is not None else np.array([1.0, 0.0, 0.0, 0.0]),
    )
    if raw.kind != "fixed":
        if raw.kind == "revolute":
            api = UsdPhysics.RevoluteJoint(prim)
            to_si = math.radians  # USD revolute limits are degrees
        else:
            api = UsdPhysics.PrismaticJoint(prim)
            to_si = lambda v: v * mpu  # noqa: E731  # USD prismatic limits are linear stage units
        raw.axis = str(api.GetAxisAttr().Get() or "X")
        lower = api.GetLowerLimitAttr().Get()
        upper = api.GetUpperLimitAttr().Get()
        raw.limit_lower = to_si(lower) if lower is not None and math.isfinite(lower) else -math.inf
        raw.limit_upper = to_si(upper) if upper is not None and math.isfinite(upper) else math.inf
        armature_attr = prim.GetAttribute("physxJoint:armature")
        if armature_attr and armature_attr.HasAuthoredValue():
            raw.armature_usd = float(armature_attr.Get())
    return raw


def _parse_geom(prim, bodies, xf_cache, Gf, UsdGeom, UsdPhysics, mpu: float):
    body_prim = prim.GetParent()
    while body_prim and not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        body_prim = body_prim.GetParent()
    if not body_prim:
        return None
    body_path = str(body_prim.GetPath())
    if body_path not in bodies:
        return None

    type_name = prim.GetTypeName()
    is_collision = False
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        is_collision = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    if not is_collision:
        if type_name != "Mesh":
            return None
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            return None

    rel_mat, _ = xf_cache.ComputeRelativeTransform(prim, body_prim)
    body = bodies[body_path]
    index = sum(1 for g in body.geoms if g.is_collision == is_collision)
    name = f"{body.name}_{'col' if is_collision else 'vis'}{index}"

    if type_name == "Mesh":
        if is_collision and prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation == "convexDecomposition":
                warnings.warn(
                    f"{prim.GetPath()}: convexDecomposition approximation without authored subshapes; "
                    "falling back to a single convex hull",
                    stacklevel=2,
                )
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        mat44 = np.array(rel_mat, dtype=np.float64)  # row-vector convention: p' = p @ M
        vertices_b = (points @ mat44[:3, :3] + mat44[3, :3]) * mpu
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        faces = _triangulate(counts, indices)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, ::-1].copy()
        geom = GeomIR(
            name=name,
            link=body.name,
            source_body=body.name,
            kind="mesh",
            is_collision=is_collision,
            pos_b=np.zeros(3),
            quat_b=np.array([1.0, 0.0, 0.0, 0.0]),
            size=np.zeros(3),
            mesh=MeshData(vertices_b=vertices_b, faces=faces),
        )
        return body_path, geom

    transform = Gf.Transform(rel_mat)
    pos_b = np.array(transform.GetTranslation(), dtype=np.float64) * mpu
    quat_b = _gf_quat_to_np(transform.GetRotation().GetQuat())
    scale = np.abs(np.array(transform.GetScale(), dtype=np.float64))

    if type_name == "Sphere":
        _require_uniform(scale, prim, (0, 1, 2))
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        kind, size = "sphere", np.array([radius * scale[0] * mpu, 0.0, 0.0])
    elif type_name == "Cube":
        edge = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        kind, size = "box", 0.5 * edge * scale * mpu
    else:  # Capsule | Cylinder
        api = UsdGeom.Capsule(prim) if type_name == "Capsule" else UsdGeom.Cylinder(prim)
        axis = str(api.GetAxisAttr().Get() or "Z")
        axis_idx = _AXIS_INDEX[axis]
        perp = [i for i in range(3) if i != axis_idx]
        _require_uniform(scale, prim, perp)
        radius = float(api.GetRadiusAttr().Get()) * scale[perp[0]] * mpu
        half_height = 0.5 * float(api.GetHeightAttr().Get()) * scale[axis_idx] * mpu
        kind = "capsule" if type_name == "Capsule" else "cylinder"
        size = np.array([radius, half_height, 0.0])
        quat_b = quat_canonical(quat_mul(quat_b, _AXIS_QUAT[axis]))

    geom = GeomIR(
        name=name,
        link=body.name,
        source_body=body.name,
        kind=kind,
        is_collision=is_collision,
        pos_b=pos_b,
        quat_b=quat_b,
        size=size,
    )
    return body_path, geom


def _require_uniform(scale: np.ndarray, prim, axes) -> None:
    values = scale[list(axes)]
    if np.ptp(values) > 1e-5 * max(1.0, float(np.max(values))):
        raise ValueError(f"{prim.GetPath()}: non-uniform scale {scale} is unsupported for this primitive")


def _resolve(path: str, fold_map: dict[str, str]) -> str:
    while path in fold_map:
        path = fold_map[path]
    return path


def _resolve_prefix(path: str, bodies: dict[str, _RawBody], fold_map: dict[str, str]) -> str | None:
    """Map a prim path to the surviving link whose body prim is the path or an ancestor of it."""
    candidate = path
    while candidate:
        if candidate in bodies:
            return _resolve(candidate, fold_map)
        candidate = candidate.rpartition("/")[0]
    return None


def _resolve_filtered(path: str, bodies: dict[str, _RawBody], fold_map: dict[str, str]) -> list[str]:
    """Surviving link(s) a FilteredPairs prim path refers to.

    A path at or under a body resolves to that link; a path on an ANCESTOR of bodies
    applies to every descendant body (UsdPhysics FilteredPairsAPI is inherited), which
    a walk-up-only resolution would silently drop.
    """
    up = _resolve_prefix(path, bodies, fold_map)
    if up is not None:
        return [up]
    prefix = path.rstrip("/") + "/"
    return sorted({_resolve(body_path, fold_map) for body_path in bodies if body_path.startswith(prefix)})


def _isaac_body_enumeration(
    bodies: dict[str, _RawBody], raw_joints: list[_RawJoint]
) -> tuple[list[str], dict[str, str]]:
    """Isaac (PhysX breadth-first) body names plus the weld map ``child -> parent``.

    PhysX keeps fixed-jointed bodies as articulation links; the MJCF builder welds
    them into their parents. The enumeration walks the PRE-weld joint tree breadth
    first with children in USD declaration order, recovering the full Isaac body
    list and which parent each welded name folds into (direct parents; callers
    resolve chains). Must run before :func:`_weld_fixed_joints` mutates ``bodies``.
    """
    children: dict[str, list[tuple[str, str]]] = {path: [] for path in bodies}
    child_paths: set[str] = set()
    for joint in sorted(raw_joints, key=lambda j: j.order):
        if joint.body0 in children:
            children[joint.body0].append((joint.body1, joint.kind))
            child_paths.add(joint.body1)
    roots = [path for path in bodies if path not in child_paths]
    if len(roots) != 1:
        raise ValueError(
            f"expected exactly one articulation root body, found {[bodies[r].name for r in roots]} — a body "
            "with no enabled joint to the tree (e.g. attached via a DISABLED joint) is not part of the "
            "articulation"
        )
    isaac_body_names: list[str] = []
    weld_parent: dict[str, str] = {}
    queue = list(roots)
    while queue:
        path = queue.pop(0)
        isaac_body_names.append(bodies[path].name)
        for child, kind in children[path]:
            if kind == "fixed":
                weld_parent[bodies[child].name] = bodies[path].name
            queue.append(child)
    return isaac_body_names, weld_parent


def _weld_fixed_joints(bodies: dict[str, _RawBody], raw_joints: list[_RawJoint]) -> dict[str, str]:
    """Merge every fixed-jointed child into its parent, mutating ``bodies`` and joint frames."""
    fold_map: dict[str, str] = {}
    for weld in [j for j in raw_joints if j.kind == "fixed"]:
        parent_path = _resolve(weld.body0, fold_map)
        child_path = weld.body1
        if child_path in fold_map or parent_path == child_path:
            raise ValueError(f"fixed joint {weld.path}: child {child_path} already welded")
        parent, child = bodies[parent_path], bodies[child_path]

        pos_pc, quat_pc = compose_frame(weld.pos0_p, weld.quat0_p, *invert_frame(weld.pos1_b, weld.quat1_b))
        rot_pc = quat_to_rotmat(quat_pc)

        child_com_p = pos_pc + rot_pc @ child.com_b
        total_mass = parent.mass + child.mass
        com_new = (parent.mass * parent.com_b + child.mass * child_com_p) / total_mass
        child_inertia_p = rot_pc @ child.inertia_b @ rot_pc.T
        parent.inertia_b = _shift_inertia(parent.inertia_b, parent.com_b - com_new, parent.mass) + _shift_inertia(
            child_inertia_p, child_com_p - com_new, child.mass
        )
        parent.mass = total_mass
        parent.com_b = com_new

        for geom in child.geoms:
            if geom.mesh is not None:
                # Mesh vertices live in the link frame under an identity geom pose;
                # re-anchoring therefore transforms the vertices and leaves the pose alone.
                geom.mesh.vertices_b = geom.mesh.vertices_b @ rot_pc.T + pos_pc
            else:
                geom.pos_b, geom.quat_b = compose_frame(pos_pc, quat_pc, geom.pos_b, geom.quat_b)
            parent.geoms.append(geom)
        child.geoms = []

        # Re-anchor joints that currently hang off the welded child so chains of welds
        # and joints below welded links keep consistent parent-frame data.
        for joint in raw_joints:
            if joint is not weld and _resolve(joint.body0, fold_map) == child_path:
                joint.pos0_p, joint.quat0_p = compose_frame(pos_pc, quat_pc, joint.pos0_p, joint.quat0_p)
        fold_map[child_path] = parent_path
    return fold_map


def _finalize_tree(
    alive: dict[str, _RawBody],
    out_joints: dict[str, list[_RawJoint]],
    root_path: str,
    up_axis: str,
) -> tuple[list[LinkIR], list[JointIR], list[GeomIR]]:
    name_of = {path: body.name for path, body in alive.items()}
    links: list[LinkIR] = []
    joints: list[JointIR] = []
    geoms: list[GeomIR] = []

    root_quat_p = np.array([1.0, 0.0, 0.0, 0.0])
    if up_axis == "Y":
        root_quat_p = np.array([_SQ2, _SQ2, 0.0, 0.0])  # rotate stage +Y up to MuJoCo +Z up

    def add_link(path: str, parent_name: str | None, pos_p: np.ndarray, quat_p: np.ndarray) -> None:
        body = alive[path]
        diag, quat_i, diag_exact = principal_inertia(body.inertia_b, link_name=body.name)
        links.append(
            LinkIR(
                name=body.name,
                parent=parent_name,
                pos_p=pos_p,
                quat_p=quat_p,
                mass=body.mass,
                com_b=body.com_b,
                inertia_diag=diag,
                inertia_diag_exact=diag_exact,
                inertia_quat_b=quat_i,
            )
        )
        for geom in body.geoms:
            geom.link = body.name
            geoms.append(geom)
        for joint in out_joints[path]:
            child_path = joint.body1
            pos_pc, quat_pc = compose_frame(joint.pos0_p, joint.quat0_p, *invert_frame(joint.pos1_b, joint.quat1_b))
            axis_b = quat_to_rotmat(joint.quat1_b) @ np.eye(3)[_AXIS_INDEX[joint.axis]]
            joints.append(
                JointIR(
                    name=joint.name,
                    parent=body.name,
                    child=name_of[child_path],
                    kind=joint.kind,
                    pos_b=joint.pos1_b,
                    axis_b=axis_b,
                    limit_lower=joint.limit_lower,
                    limit_upper=joint.limit_upper,
                    armature_usd=joint.armature_usd,
                )
            )
            add_link(child_path, body.name, pos_pc, quat_pc)

    add_link(root_path, None, np.zeros(3), root_quat_p)
    return links, joints, geoms


def _bfs_joint_order(out_joints: dict[str, list[_RawJoint]], root_path: str) -> list[str]:
    """PhysX articulation joint order: breadth-first, children in USD declaration order."""
    order: list[str] = []
    queue = [root_path]
    while queue:
        path = queue.pop(0)
        for joint in out_joints[path]:
            order.append(joint.name)
            queue.append(joint.body1)
    return order
