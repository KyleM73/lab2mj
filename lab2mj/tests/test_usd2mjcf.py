"""Tests for the USD -> MJCF converter (offline; skips when cached USD assets are absent)."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from .shared import FIXTURES, G1_USD, HAS_PXR, SPOT_USD, needs_g1  # noqa: F401

# Link masses + calibrated joint ranges extracted from the hand-baked Spot MJCF.
SPOT_BAKED_VALUES = FIXTURES / "spot_baked_values.json"

needs_spot = pytest.mark.skipif(
    not (HAS_PXR and SPOT_USD.exists()),
    reason="requires pxr (usd-core) and data/usd_cache/spot.usd",
)

ISAAC_SPOT_JOINT_ORDER = [
    "fl_hx",
    "fr_hx",
    "hl_hx",
    "hr_hx",
    "fl_hy",
    "fr_hy",
    "hl_hy",
    "hr_hy",
    "fl_kn",
    "fr_kn",
    "hl_kn",
    "hr_kn",
]

# The hand-baked spot.xml carries per-leg calibrated limits on a few upper bounds
# (measured robot data); the USD authors one nominal range per joint type, so those
# bounds legitimately differ by more than the default 1e-4 tolerance.
CALIBRATED_RANGE_TOL = {
    ("fr_hy", 1): 6e-2,
    ("fl_kn", 1): 1e-2,
    ("fr_kn", 1): 1e-2,
    ("hr_kn", 1): 2e-3,
}


@pytest.fixture(scope="module")
def spot():
    from lab2mj.usd2mjcf import build_mjcf, parse_usd

    robot = parse_usd(SPOT_USD)
    return robot, build_mjcf(robot, physics_dt=0.005)


@pytest.fixture(scope="module")
def g1():
    from lab2mj.usd2mjcf import build_mjcf, parse_usd

    robot = parse_usd(G1_USD)
    return robot, build_mjcf(robot, physics_dt=0.005)


def _baked_spot_values() -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    baked = json.loads(SPOT_BAKED_VALUES.read_text())
    masses = {name: float(mass) for name, mass in baked["masses"].items()}
    ranges = {name: (float(lo), float(hi)) for name, (lo, hi) in baked["ranges"].items()}
    return masses, ranges


@needs_spot
class TestSpot:
    def test_isaac_joint_order(self, spot):
        robot, _ = spot
        assert robot.isaac_joint_order == ISAAC_SPOT_JOINT_ORDER

    def test_masses_match_baked(self, spot):
        robot, built = spot
        model = built.spec.compile()
        baked_masses, _ = _baked_spot_values()
        assert set(baked_masses) == {link.name for link in robot.links}
        for name, baked in baked_masses.items():
            converted = float(model.body(name).mass[0])
            assert converted == pytest.approx(baked, rel=1e-5), name

    def test_joint_ranges_match_baked(self, spot):
        _, built = spot
        model = built.spec.compile()
        _, baked_ranges = _baked_spot_values()
        assert len(baked_ranges) == 12
        for name, baked in baked_ranges.items():
            joint_id = model.joint(name).id
            for bound in (0, 1):
                tol = CALIBRATED_RANGE_TOL.get((name, bound), 1e-4)
                assert abs(model.jnt_range[joint_id][bound] - baked[bound]) <= tol, (name, bound)

    def test_massless_feet_folded(self, spot):
        robot, _ = spot
        names = {link.name for link in robot.links}
        assert len(robot.links) == 13
        assert not any("foot" in name for name in names)
        # The foot collision sphere survives the weld, re-parented into the lower leg.
        foot_geoms = [g for g in robot.geoms if g.link == "fl_lleg" and g.kind == "sphere"]
        assert len(foot_geoms) == 1
        np.testing.assert_allclose(foot_geoms[0].pos_b, [0.0, 0.0, -0.3365], atol=1e-6)

    def test_verify_passes(self, spot):
        from lab2mj.usd2mjcf import verify_build

        robot, built = spot
        verify_build(robot, built)

    def test_deterministic_output(self):
        from lab2mj.usd2mjcf import build_mjcf, parse_usd

        first = build_mjcf(parse_usd(SPOT_USD), physics_dt=0.005)
        second = build_mjcf(parse_usd(SPOT_USD), physics_dt=0.005)
        assert first.xml == second.xml
        assert first.assets == second.assets


def _single_hinge_robot(limit: tuple[float, float] = (-0.5, 0.5)):
    """Minimal one-hinge RobotIR mirroring a low-inertia hand joint (no USD needed)."""
    from lab2mj.usd2mjcf.parser import GeomIR, JointIR, LinkIR, RobotIR

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    zero = np.zeros(3)
    links = [
        LinkIR(
            name="base",
            parent=None,
            pos_p=zero,
            quat_p=identity,
            mass=100.0,
            com_b=zero,
            inertia_diag=np.array([10.0, 10.0, 10.0]),
            inertia_quat_b=identity,
        ),
        LinkIR(
            name="finger",
            parent="base",
            pos_p=np.array([0.1, 0.0, 0.0]),
            quat_p=identity,
            mass=0.05,
            com_b=np.array([0.05, 0.0, 0.0]),
            inertia_diag=np.array([2e-4, 2e-4, 1e-5]),
            inertia_quat_b=identity,
        ),
    ]
    joints = [
        JointIR(
            name="finger_joint",
            parent="base",
            child="finger",
            kind="revolute",
            pos_b=zero,
            axis_b=np.array([0.0, 0.0, 1.0]),
            limit_lower=limit[0],
            limit_upper=limit[1],
        )
    ]
    geoms = [
        GeomIR(
            name="finger_geom",
            link="finger",
            source_body="finger",
            kind="box",
            is_collision=False,
            pos_b=np.array([0.05, 0.0, 0.0]),
            quat_b=identity,
            size=np.array([0.05, 0.01, 0.01]),
        )
    ]
    return RobotIR(
        name="hinge_rig",
        links=links,
        joints=joints,
        geoms=geoms,
        isaac_joint_order=["finger_joint"],
        self_collisions_enabled=False,
        filtered_pairs=[],
    )


class TestJointLimitConstraint:
    """PhysX pins limited joints at the range endpoint; the authored limit must match."""

    def test_limit_solref_solimp_scale_with_timestep(self):
        from lab2mj.usd2mjcf import build_mjcf

        for dt in (0.00125, 0.005):
            model = build_mjcf(_single_hinge_robot(), physics_dt=dt).spec.compile()
            joint_id = model.joint("finger_joint").id
            damping = 0.75 / dt
            stiffness = (damping / 6.0) ** 2
            margin = 400.0 * dt * dt
            np.testing.assert_allclose(model.jnt_solref[joint_id], [-stiffness, -damping])
            np.testing.assert_allclose(model.jnt_solimp[joint_id], [0.001, 0.9999, margin, 0.5, 2.0])
            np.testing.assert_allclose(model.jnt_margin[joint_id], margin)
            assert model.jnt_limited[joint_id] == 1

    def test_stall_torque_pins_at_endpoint_without_chatter(self):
        """Drive torque commanding past the limit must settle within ~1e-4 rad of the
        endpoint (PhysX pins exactly there), monotone, without the bounce limit cycle."""
        from lab2mj.usd2mjcf import build_mjcf

        dt = 0.00125
        built = build_mjcf(
            _single_hinge_robot(),
            physics_dt=dt,
            joint_overrides={"finger_joint": {"armature": 1e-3, "damping": 10.0}},
        )
        model = mujoco.MjModel.from_xml_string(built.xml, {name: data for name, data in built.assets.items()})
        # No floor in the bundle: drop gravity so the free-floating rig holds still and
        # the heavy base (100 kg) absorbs the hinge reaction torque.
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        for tau in (15.0, 35.0):
            mujoco.mj_resetData(model, data)
            data.ctrl[model.actuator("finger_joint").id] = tau
            for _ in range(int(2.0 / dt)):
                mujoco.mj_step(model, data)
            hinge_adr = model.jnt_qposadr[model.joint("finger_joint").id]
            tail = []
            for _ in range(int(0.5 / dt)):
                mujoco.mj_step(model, data)
                tail.append(float(data.qpos[hinge_adr]))
            tail_arr = np.asarray(tail) - 0.5
            assert np.all(np.isfinite(tail_arr))
            # Rests inside the activation margin a hair short of the endpoint, far
            # below the ~0.01-0.02 rad penetration MuJoCo's default impedance allows.
            assert abs(tail_arr.mean()) < 1e-4, tau
            assert tail_arr.std() < 1e-5, tau

    def test_endpoint_rest_state_gets_no_free_flight_kick(self):
        """A joint teleported exactly onto the endpoint under stall torque (the PhysX
        steady state, and how Isaac reference states replay) must stay put: the margin
        keeps the constraint row active, so the drive never gets a free substep."""
        from lab2mj.usd2mjcf import build_mjcf

        dt = 0.00125
        built = build_mjcf(
            _single_hinge_robot(),
            physics_dt=dt,
            joint_overrides={"finger_joint": {"armature": 1e-3, "damping": 10.0}},
        )
        model = mujoco.MjModel.from_xml_string(built.xml, {name: data for name, data in built.assets.items()})
        model.opt.gravity[:] = 0.0
        data = mujoco.MjData(model)
        hinge_adr = model.jnt_qposadr[model.joint("finger_joint").id]
        data.qpos[hinge_adr] = 0.5
        data.ctrl[model.actuator("finger_joint").id] = 25.0
        worst = 0.0
        for _ in range(16):  # one policy step at substeps=4, decimation=4
            mujoco.mj_step(model, data)
            worst = max(worst, abs(float(data.qpos[hinge_adr]) - 0.5))
        assert worst < 1e-4

    def test_unlimited_joint_stays_unlimited(self):
        from lab2mj.usd2mjcf import build_mjcf

        robot = _single_hinge_robot(limit=(-np.inf, np.inf))
        model = build_mjcf(robot, physics_dt=0.00125).spec.compile()
        assert model.jnt_limited[model.joint("finger_joint").id] == 0


@needs_g1
class TestG1:
    def test_structure(self, g1):
        robot, built = g1
        model = built.spec.compile()
        assert len(robot.isaac_joint_order) == 37
        assert int((model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE)).sum()) == 37
        assert int((model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)).sum()) == 1
        assert bool((model.body_mass[1:] > 0).all())

    def test_verify_passes(self, g1):
        from lab2mj.usd2mjcf import verify_build

        robot, built = g1
        verify_build(robot, built)


class TestWeldGeomReanchoring:
    def test_mesh_and_primitive_stay_coincident_through_weld(self):
        from lab2mj.usd2mjcf.parser import GeomIR, MeshData, _RawBody, _RawJoint, _weld_fixed_joints

        eye = np.eye(3)
        # Unit tetra centered at (0.1, 0, -0.3) in the CHILD link frame, plus a sphere at
        # the same point: after welding, both must land at the same parent-frame location.
        center_child = np.array([0.1, 0.0, -0.3])
        verts = center_child + 0.01 * np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
        mesh_geom = GeomIR(
            name="foot_vis0",
            link="foot",
            source_body="foot",
            kind="mesh",
            is_collision=False,
            pos_b=np.zeros(3),
            quat_b=np.array([1.0, 0, 0, 0]),
            size=np.zeros(3),
            mesh=MeshData(vertices_b=verts.copy(), faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])),
        )
        sphere_geom = GeomIR(
            name="foot_col0",
            link="foot",
            source_body="foot",
            kind="sphere",
            is_collision=True,
            pos_b=center_child.copy(),
            quat_b=np.array([1.0, 0, 0, 0]),
            size=np.array([0.03, 0, 0]),
        )
        parent = _RawBody(path="/shank", name="shank", mass=1.0, com_b=np.zeros(3), inertia_b=eye * 1e-3)
        child = _RawBody(
            path="/foot",
            name="foot",
            mass=0.1,
            com_b=np.zeros(3),
            inertia_b=eye * 1e-5,
            geoms=[mesh_geom, sphere_geom],
        )
        # Fixed joint: child frame sits 0.35 below the shank origin, rotated 90 deg about x.
        half = np.sqrt(0.5)
        weld = _RawJoint(
            path="/shank/foot_fixed",
            name="foot_fixed",
            order=0,
            kind="fixed",
            body0="/shank",
            body1="/foot",
            pos0_p=np.array([0.0, 0.0, -0.35]),
            quat0_p=np.array([half, half, 0.0, 0.0]),
            pos1_b=np.zeros(3),
            quat1_b=np.array([1.0, 0, 0, 0]),
        )
        bodies = {"/shank": parent, "/foot": child}
        _weld_fixed_joints(bodies, [weld])

        merged = {g.name: g for g in parent.geoms}
        merged_mesh = merged["foot_vis0"].mesh
        assert merged_mesh is not None
        mesh_center_p = merged_mesh.vertices_b.mean(axis=0)
        sphere_p = merged["foot_col0"].pos_b
        np.testing.assert_allclose(mesh_center_p, sphere_p, atol=1e-12)
        # And both match the hand-computed child->parent transform of the point.
        rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        expected = np.array([0.0, 0.0, -0.35]) + rot @ center_child
        np.testing.assert_allclose(sphere_p, expected, atol=1e-12)


class TestTriangleInequalityProjection:
    def test_violating_inertia_is_projected_with_warning(self):
        from lab2mj.usd2mjcf.parser import principal_inertia

        bad = np.diag([0.0104, 0.00034, 0.00011])
        with pytest.warns(UserWarning, match="triangle inequality"):
            diag, _, exact = principal_inertia(bad, link_name="left_toe")
        assert diag[0] == pytest.approx(0.00045)
        assert diag[0] <= diag[1] + diag[2] + 1e-15
        # The exact (PhysX-simulated) moment survives for the load-time stamp.
        assert exact[0] == pytest.approx(0.0104)

    def test_valid_inertia_untouched(self):
        import warnings as _w

        from lab2mj.usd2mjcf.parser import principal_inertia

        good = np.diag([0.02, 0.015, 0.01])
        with _w.catch_warnings():
            _w.simplefilter("error")
            diag, _, exact = principal_inertia(good, link_name="ok")
        np.testing.assert_allclose(diag, [0.02, 0.015, 0.01])
        np.testing.assert_allclose(exact, diag)


class TestContactProfile:
    """The "engagement" profile authors the PhysX-style progressive contact model."""

    def test_default_profile_keeps_mujoco_contact(self):
        from lab2mj.usd2mjcf import build_mjcf

        model = build_mjcf(_single_hinge_robot(), physics_dt=0.005).spec.compile()
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        assert model.opt.impratio == pytest.approx(1.0)
        np.testing.assert_allclose(model.geom_solimp[0], [0.9, 0.95, 0.001, 0.5, 2.0])

    def test_engagement_profile_authors_contact_model(self):
        from lab2mj.usd2mjcf import build_mjcf
        from lab2mj.usd2mjcf.builder import _ENGAGEMENT_IMPRATIO, _ENGAGEMENT_SOLIMP

        built = build_mjcf(_single_hinge_robot(), physics_dt=0.005, contact_profile="engagement")
        model = built.spec.compile()
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        assert model.opt.impratio == pytest.approx(_ENGAGEMENT_IMPRATIO)
        np.testing.assert_allclose(model.geom_solimp[0], _ENGAGEMENT_SOLIMP)
        # solref default is untouched by the profile (still 2 * physics_dt, dampratio 1)
        np.testing.assert_allclose(model.geom_solref[0], [0.01, 1.0])
        # joint-limit constraint authoring is independent of the contact profile
        joint_id = model.joint("finger_joint").id
        np.testing.assert_allclose(model.jnt_solimp[joint_id], [0.001, 0.9999, 400.0 * 0.005**2, 0.5, 2.0])
        # the profile survives XML round-tripping (scene.xml is spec.to_xml())
        assert 'cone="elliptic"' in built.xml
        assert 'impratio="3"' in built.xml

    def test_explicit_solimp_overrides_profile_solimp(self):
        from lab2mj.usd2mjcf import build_mjcf

        explicit = (0.8, 0.9, 0.002, 0.5, 2.0)
        model = build_mjcf(
            _single_hinge_robot(), physics_dt=0.005, contact_profile="engagement", solimp=explicit
        ).spec.compile()
        np.testing.assert_allclose(model.geom_solimp[0], explicit)
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC  # cone/impratio still author

    def test_unknown_profile_rejected(self):
        from lab2mj.usd2mjcf import build_mjcf

        with pytest.raises(ValueError, match="contact_profile"):
            build_mjcf(_single_hinge_robot(), physics_dt=0.005, contact_profile="physx")
