"""Tests for converter CLI helpers (`lab2mj.convert`) — no Isaac, no network."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from lab2mj.convert import _resolve_actuator_nets, resolve_usd
from lab2mj.env_yaml import load_env_yaml, parse_env_dict
from lab2mj.ir import ActuatorGroupIR

from .shared import FIXTURES, G1_USD, SPOT_USD, needs_g1, needs_spot

URL = "https://example.com/assets/robot.usd"


@pytest.fixture()
def ir():
    ir = parse_env_dict(load_env_yaml(FIXTURES / "g1_flat_env.yaml"))
    ir.usd_path = URL
    return ir


class TestResolveUsd:
    def test_plain_basename_wins_over_keyed_dir(self, ir, tmp_path, capsys):
        url_sha = hashlib.sha256(URL.encode()).hexdigest()[:8]
        keyed_dir = tmp_path / f"{url_sha}_robot"
        keyed_dir.mkdir()
        (keyed_dir / "robot.usd").write_bytes(b"keyed")
        (tmp_path / "robot.usd").write_bytes(b"plain")
        # Pre-seeded flat entries short-circuit dependency mirroring (with a notice).
        assert resolve_usd(None, ir, cache_dir=tmp_path) == tmp_path / "robot.usd"
        assert "NOTICE" in capsys.readouterr().out


class TestDownload:
    def test_download_is_atomic_and_uses_timeout(self, ir, tmp_path, monkeypatch):
        import io
        import urllib.request

        captured = {}

        def fake_urlopen(url, timeout=None):
            captured["timeout"] = timeout
            return io.BytesIO(b"usd-bytes")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        path = resolve_usd(None, ir, cache_dir=tmp_path)
        url_sha = hashlib.sha256(URL.encode()).hexdigest()[:8]
        assert path.parent == tmp_path / f"{url_sha}_robot"
        assert path.read_bytes() == b"usd-bytes"
        assert captured["timeout"] is not None
        assert not list(tmp_path.rglob("*.part"))

    def test_failed_download_leaves_no_cache_entry(self, ir, tmp_path, monkeypatch):
        import urllib.request

        class BrokenResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                raise OSError("connection reset")

        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: BrokenResponse())
        with pytest.raises(OSError):
            resolve_usd(None, ir, cache_dir=tmp_path)
        # No truncated file and no leftover temp file: the next resolve retries
        # (an empty keyed directory may remain; it is not treated as a cache hit).
        assert not any(p.is_file() for p in tmp_path.rglob("*"))


def _net_group(name: str, network_file: str) -> ActuatorGroupIR:
    return ActuatorGroupIR(
        name=name,
        joint_names_expr=[".*"],
        model="actuator_net_lstm",
        effort_limit=80.0,
        velocity_limit=7.5,
        saturation_effort=120.0,
        network_file=network_file,
    )


class TestResolveActuatorNets:
    def test_basename_collision_disambiguated_by_group_name(self, ir, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        net_a = tmp_path / "a" / "net.pt"
        net_b = tmp_path / "b" / "net.pt"
        net_a.write_bytes(b"a")
        net_b.write_bytes(b"b")
        g1, g2 = _net_group("legs", str(net_a)), _net_group("arms", str(net_b))
        ir.actuators = [g1, g2]
        files = _resolve_actuator_nets(ir)
        assert g1.network_bundle_path == "assets/actuator_nets/net.pt"
        assert g2.network_bundle_path == "assets/actuator_nets/arms_net.pt"
        assert files[g1.network_bundle_path] == net_a
        assert files[g2.network_bundle_path] == net_b

    def test_shared_source_shares_bundle_entry(self, ir, tmp_path):
        net = tmp_path / "net.pt"
        net.write_bytes(b"jit")
        g1, g2 = _net_group("legs", str(net)), _net_group("arms", str(net))
        ir.actuators = [g1, g2]
        files = _resolve_actuator_nets(ir)
        assert g1.network_bundle_path == g2.network_bundle_path == "assets/actuator_nets/net.pt"
        assert files == {"assets/actuator_nets/net.pt": net}

    def test_url_source_downloads_into_cache(self, ir, tmp_path, monkeypatch):
        import io
        import urllib.request

        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: io.BytesIO(b"jit-bytes"))
        url = "https://example.com/nets/anydrive_3_lstm_jit.pt"
        group = _net_group("legs", url)
        ir.actuators = [group]
        files = _resolve_actuator_nets(ir, cache_dir=tmp_path)
        assert group.network_bundle_path == "assets/actuator_nets/anydrive_3_lstm_jit.pt"
        (rel,) = files
        assert files[rel].parent == tmp_path
        assert files[rel].read_bytes() == b"jit-bytes"

    def test_pd_groups_are_ignored(self, ir):
        # The fixture's own PD groups have no network_file and must pass through.
        assert _resolve_actuator_nets(ir) == {}
        assert all(g.network_bundle_path is None for g in ir.actuators)


def _write_500hz_env_yaml(out_dir: Path) -> Path:
    """Spot fixture re-timed to the stock Spot cadence: sim.dt=0.002, decimation=10."""
    text = (FIXTURES / "spot_velocity_env.yaml").read_text()
    assert text.count("\n  dt: 0.005\n") == 1
    assert text.count("\ndecimation: 4\n") == 1
    text = text.replace("\n  dt: 0.005\n", "\n  dt: 0.002\n")
    text = text.replace("\ndecimation: 4\n", "\ndecimation: 10\n")
    path = out_dir / "spot_500hz_env.yaml"
    path.write_text(text)
    return path


@needs_spot
class TestNonDefaultTiming:
    """Timing generality: the stock Spot task runs sim.dt=0.002 / decimation=10 (500 Hz
    physics, 50 Hz policy). Every derived timestep — model timestep, contact solref,
    joint-limit gains, runtime cadence — must follow env.yaml."""

    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory):
        from lab2mj.convert import convert_run

        yaml_path = _write_500hz_env_yaml(tmp_path_factory.mktemp("cfg"))
        out = convert_run(yaml_path, usd=SPOT_USD, out=tmp_path_factory.mktemp("spot_500hz_bundle"))
        return yaml_path, out

    def test_manifest_timing_and_substeps(self, bundle):
        from lab2mj.bundle import read_manifest

        _, out = bundle
        timing = read_manifest(out)["timing"]
        assert timing["physics_dt"] == pytest.approx(0.002)
        assert timing["decimation"] == 10
        assert timing["physics_substeps"] == 4  # solver_position_iteration_count

    def test_model_timestep_and_constraint_gains_track_substep_dt(self, bundle):
        _, out = bundle
        model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
        substep_dt = 0.002 / 4
        assert model.opt.timestep == pytest.approx(substep_dt, abs=1e-15)
        # Contact solref (2 * substep dt, 1) on every contact-enabled geom.
        contact = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        assert contact.any()
        np.testing.assert_allclose(model.geom_solref[contact, 0], 2.0 * substep_dt, atol=1e-15)
        np.testing.assert_allclose(model.geom_solref[contact, 1], 1.0, atol=1e-15)
        # Joint-limit constraint gains authored against the substep dt.
        damping = 0.75 / substep_dt
        stiffness = (damping / 6.0) ** 2
        margin = 400.0 * substep_dt * substep_dt
        jid = model.joint("fl_hx").id
        np.testing.assert_allclose(model.jnt_solref[jid], [-stiffness, -damping])
        np.testing.assert_allclose(model.jnt_margin[jid], margin)

    def test_runtime_steps_at_patched_cadence(self, bundle):
        from lab2mj.env import MjEnv

        _, out = bundle
        env = MjEnv(out, strict=True, seed=0)
        assert env.physics_dt == pytest.approx(0.002)
        assert env.decimation == 10
        assert env.policy_dt == pytest.approx(0.02)
        assert env.substep_dt == pytest.approx(0.0005)
        obs = env.reset()
        assert np.all(np.isfinite(obs))
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(env.action_dim))
            assert np.all(np.isfinite(obs))


class TestUsdDependencyDownload:
    def test_relative_dependencies_are_mirrored(self, tmp_path):
        from lab2mj.convert import _download_usd_with_dependencies

        remote = tmp_path / "remote"
        (remote / "Props").mkdir(parents=True)
        (remote / "robot.usda").write_text(
            '#usda 1.0\n(\n    subLayers = [@./Props/instanceable_meshes.usda@]\n)\n\ndef "root" {}\n'
        )
        (remote / "Props" / "instanceable_meshes.usda").write_text('#usda 1.0\n\ndef "meshes" {}\n')
        cache = tmp_path / "cache"
        root = _download_usd_with_dependencies(f"file://{remote}/robot.usda", cache)
        assert root.is_file()
        assert (root.parent / "Props" / "instanceable_meshes.usda").is_file()

    def test_escaping_dependency_raises(self, tmp_path):
        from lab2mj.convert import _download_usd_with_dependencies

        remote = tmp_path / "remote" / "nested"
        remote.mkdir(parents=True)
        (remote / "robot.usda").write_text('#usda 1.0\n(\n    subLayers = [@../../evil.usda@]\n)\n\ndef "root" {}\n')
        (tmp_path / "evil.usda").write_text("#usda 1.0\n")
        cache = tmp_path / "cache"
        with pytest.raises(ValueError, match="escapes the cache directory"):
            _download_usd_with_dependencies(f"file://{remote}/robot.usda", cache)


class TestZeroCollisionGuard:
    def test_robot_without_collision_geoms_raises(self, tmp_path):
        pxr = pytest.importorskip("pxr")  # noqa: F841
        from lab2mj.usd2mjcf.parser import parse_usd

        usd = tmp_path / "nocol.usda"
        usd.write_text(
            """#usda 1.0
(
    defaultPrim = "robot"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        point3f physics:centerOfMass = (0, 0, 0)
        float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
    }
}
"""
        )
        with pytest.raises(ValueError, match="no collision geoms"):
            parse_usd(usd)


class TestMeasuredCollisionPlan:
    """Resolution/window resolution for measured-terrain collision resampling."""

    @staticmethod
    def _measured(nx: int = 1201, ny: int = 2001, res: float = 0.1):
        from lab2mj.terrain import MeasuredTerrainData

        return MeasuredTerrainData(
            height_grid=np.zeros((nx, ny), dtype=np.float32),
            grid_origin_xy=(-0.5 * (nx - 1) * res, -0.5 * (ny - 1) * res),
            grid_resolution=res,
            mesh_vertices_w=np.zeros((3, 3), dtype=np.float32),
            mesh_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            env_origin_w=np.zeros(3),
        )

    @staticmethod
    def _dump(x0: float = 0.0, x1: float = 5.0, y0: float = 0.0, y1: float = 1.0):
        traj = np.zeros((10, 3))
        traj[:, 0] = np.linspace(x0, x1, 10)
        traj[:, 1] = np.linspace(y0, y1, 10)
        return {"root_pos_w": traj}

    def test_auto_picks_quarter_resolution_within_window_budget(self):
        from lab2mj.convert import _measured_collision_plan

        res, window = _measured_collision_plan(self._measured(), self._dump(), "auto", 2.0, False)
        assert res == pytest.approx(0.025)
        assert window == pytest.approx((-2.0, 7.0, -2.0, 3.0))

    def test_auto_full_extent_falls_back_to_coarser_candidate(self):
        from lab2mj.convert import _measured_collision_plan

        # Full extent: 120 x 200 m; 0.025 needs ~38M nodes (over budget), 0.05 ~9.6M
        # (still over), so auto keeps the record grid.
        res, window = _measured_collision_plan(self._measured(), self._dump(), "auto", 2.0, True)
        assert res is None and window is None

    def test_record_and_coarser_than_record_are_passthrough(self):
        from lab2mj.convert import _measured_collision_plan

        assert _measured_collision_plan(self._measured(), self._dump(), "record", 2.0, False) == (None, None)
        assert _measured_collision_plan(self._measured(), self._dump(), 0.1, 2.0, False) == (None, None)
        assert _measured_collision_plan(self._measured(), self._dump(), 0.2, 2.0, False) == (None, None)

    def test_explicit_resolution_over_budget_raises(self):
        from lab2mj.convert import _measured_collision_plan

        with pytest.raises(ValueError, match="budget"):
            _measured_collision_plan(self._measured(), self._dump(), 0.025, 2.0, True)

    def test_explicit_windowed_resolution(self):
        from lab2mj.convert import _measured_collision_plan

        res, window = _measured_collision_plan(self._measured(), self._dump(), 0.05, 1.0, False)
        assert res == pytest.approx(0.05)
        assert window == pytest.approx((-1.0, 6.0, -1.0, 2.0))


@needs_g1
class TestContactOverrides:
    """Calibration output (--contact_solimp / --contact_impratio) must reach the
    compiled model and the manifest."""

    def test_overrides_author_model_and_manifest(self, tmp_path_factory):
        import mujoco

        from lab2mj.bundle import read_manifest
        from lab2mj.convert import convert_run

        solimp = (0.5, 0.99, 0.001, 0.5, 2.0)
        out = convert_run(
            FIXTURES / "g1_flat_env.yaml",
            usd=G1_USD,
            out=tmp_path_factory.mktemp("g1_contact_override"),
            contact_solimp=solimp,
            contact_impratio=5.0,
        )
        model = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
        assert model.opt.impratio == pytest.approx(5.0)
        robot_geoms = np.nonzero(model.geom_priority == 1)[0]
        np.testing.assert_allclose(model.geom_solimp[robot_geoms], np.tile(solimp, (len(robot_geoms), 1)), rtol=1e-5)
        manifest = read_manifest(out)
        assert manifest["sim"]["contact_impratio"] == pytest.approx(5.0)
        assert manifest["sim"]["contact_solimp"] == pytest.approx(list(solimp))


class TestLegacyFriction:
    JOINTS = ["fl_hx", "fl_hy", "fl_kn"]

    def _ir(self):
        from lab2mj.ir import EventIR

        raw = load_env_yaml(FIXTURES / "spot_velocity_env.yaml")
        ir = parse_env_dict(raw)
        for group in ir.actuators:
            group.friction = group.dynamic_friction = group.viscous_friction = None
        coefficients = {".*_h[xy]": 0.008, ".*_kn": 0.18}
        ir.events.append(
            EventIR(
                name="legacy", func="m:set_legacy_joint_friction", mode="startup", params={"coefficients": coefficients}
            )
        )
        return ir

    def test_writes_fixed_friction_into_the_groups(self):
        from lab2mj.actuators import ActuatorSet
        from lab2mj.convert import apply_legacy_friction, parse_legacy_friction

        ir = self._ir()
        equivalents = parse_legacy_friction(".*_h[xy]=0.1,0.1,0.0; .*_kn=1.5,1.0,0.5")
        record = apply_legacy_friction(ir, equivalents, self.JOINTS)
        assert record is not None and record["coefficients"] == {".*_h[xy]": 0.008, ".*_kn": 0.18}
        aset = ActuatorSet.from_ir(ir.actuators, self.JOINTS)
        np.testing.assert_allclose(aset.static_friction, [0.1, 0.1, 1.5])
        np.testing.assert_allclose(aset.dynamic_friction, [0.1, 0.1, 1.0])
        np.testing.assert_allclose(aset.viscous_friction, [0.0, 0.0, 0.5])

    def test_requires_matching_equivalents(self):
        from lab2mj.convert import apply_legacy_friction

        with pytest.raises(ValueError, match="--legacy_friction"):
            apply_legacy_friction(self._ir(), None, self.JOINTS)
        with pytest.raises(ValueError, match="must equal"):
            apply_legacy_friction(self._ir(), {".*_kn": (1.5, 1.0, 0.5)}, self.JOINTS)
