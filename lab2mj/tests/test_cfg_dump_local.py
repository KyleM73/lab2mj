"""Tests for the Isaac-free env-config dump path (isaac_shims + lab2mj.dump_env_cfg).

Each test runs in a subprocess: installing the shims registers a meta-path finder and
patches ``pxr``, which must not leak into the rest of the test session.
"""

import importlib.util
import subprocess
import sys

import pytest
import yaml

from .shared import REPO_ROOT as REPO
from .shared import SPOT_ENV_YAML

# Keys a training run resolves at launch time; everything else must match exactly.
RUNTIME_KEYS = ("log_dir", "io_descriptors_output_dir")

needs_isaaclab = pytest.mark.skipif(importlib.util.find_spec("isaaclab") is None, reason="isaaclab not installed")
needs_contact_lab = pytest.mark.skipif(
    importlib.util.find_spec("contact_lab") is None, reason="contact_lab not installed (registers spot-velocity-v0)"
)


@needs_isaaclab
def test_shim_semantics():
    """One interpreter for all shim checks (each subprocess costs ~1-2 s): install()
    is idempotent, the isaaclab config layer imports under shims, and reading a
    settings key outside KNOWN_SETTINGS raises instead of returning a stub."""
    code = (
        "from lab2mj.isaac_shims import ShimSettingsError, install, is_installed\n"
        "install()\n"
        "install()\n"
        "assert is_installed()\n"
        "import carb\n"
        "settings = carb.settings.get_settings()\n"
        "assert settings.get('/persistent/isaac/asset_root/cloud').startswith('https://')\n"
        "try:\n"
        "    settings.get('/some/unknown/key')\n"
        "except ShimSettingsError:\n"
        "    print('RAISED')\n"
        "import isaaclab.utils.assets as assets\n"
        "print(assets.ISAACLAB_NUCLEUS_DIR)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "RAISED" in proc.stdout
    assert (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab" in proc.stdout
    )


@needs_isaaclab
@needs_contact_lab
@pytest.mark.skipif(not SPOT_ENV_YAML.exists(), reason=f"missing fixture {SPOT_ENV_YAML}")
def test_local_dump_matches_spot_fixture(tmp_path):
    """A mac dump of spot-velocity-v0 must equal the on-box fixture up to runtime keys."""
    out_path = tmp_path / "env.yaml"
    cmd = [
        sys.executable,
        "-m",
        "lab2mj.dump_env_cfg",
        "--task",
        "spot-velocity-v0",
        "--out",
        str(out_path),
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--train_physx_buffers",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr

    local = yaml.unsafe_load(out_path.read_text())
    ref = yaml.unsafe_load(SPOT_ENV_YAML.read_text())
    for data in (local, ref):
        for key in RUNTIME_KEYS:
            data.pop(key, None)
    assert local == ref
