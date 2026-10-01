"""Import shims for the Omniverse runtime namespaces (``carb``, ``omni``, ``isaacsim``).

IsaacLab's config layer (``isaaclab``, ``isaaclab_tasks``, ``isaaclab_assets``,
``contact_lab.tasks``) imports Omniverse runtime modules at module scope, so on a
machine without Isaac Sim the packages cannot even be imported — even though a
fully-resolved ``params/env.yaml`` is nothing but ``class_to_dict(env_cfg)`` of
plain ``@configclass`` dataclasses. :func:`install` registers stub modules for the
missing runtime namespaces so the config layer imports cleanly and env configs can
be instantiated and dumped without Isaac Sim (see
``lab2mj.dump_env_cfg``).

Loud-failure design — the shims must be a no-op for every value that can end up in
a dumped yaml:

* ``carb.settings`` is the only shimmed API whose return values leak into configs
  (the Nucleus asset-root URLs baked into ``usd_path`` fields). It is backed by
  :data:`KNOWN_SETTINGS`, a curated map of setting keys to the *real* values the
  Isaac Sim 5.1 kit apps define. Reading any key not in the map raises
  :class:`ShimSettingsError` instead of returning a placeholder, so an unnoticed
  new settings read can never silently leak a stub value into a dump.
* Every other shimmed attribute resolves to an inert placeholder class. Inert
  placeholders support the operations the config layer performs on them at import
  time (subclassing, instantiation, attribute chains, ``__name__``/``__module__``
  introspection) and nothing else; if one is ever serialized it stringifies to its
  ``omni.``/``carb.``/``isaacsim.`` qualified name, which a structural comparison
  against a real dump immediately exposes.
* Isaac Lab add-on packages that the config layer imports but a config-only install
  lacks (:data:`_OPTIONAL_PACKAGES`; Isaac Lab 2.3.2's ``isaaclab.scene`` imports
  ``isaaclab_contrib``) are stubbed the same way, only when they are not installed.
* :func:`install` refuses to run when the real runtime is present or when
  ``isaaclab`` modules that bake settings values into module constants were
  already imported without the shims.

The stubs cover import and config construction only; anything that needs the
simulator still requires Isaac Sim.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types

_SHIMMED_NAMESPACES = ("carb", "omni", "isaacsim")
# Isaac Lab packages imported by the config layer that config-only installs may lack.
_OPTIONAL_PACKAGES = ("isaaclab_contrib",)

# isaaclab modules that read carb.settings at module scope and bake the result into
# module-level constants. They must not be imported before the shims are installed.
_SETTINGS_SENSITIVE_MODULES = (
    "isaaclab.utils.assets",
    "isaaclab.utils.pretrained_checkpoint",
)

# Real values of every carb setting the isaaclab config layer reads. The URLs are the
# ``persistent.isaac.asset_root.*`` values from Isaac Sim 5.1's kit apps and match the
# ``usd_path`` prefixes in env.yaml dumps produced on a real Isaac installation. A
# ``None`` value is itself real: carb returns None for keys no kit app defines.
_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
KNOWN_SETTINGS: dict[str, str | None] = {
    "/persistent/isaac/asset_root/cloud": _ASSET_ROOT,
    "/persistent/isaac/asset_root/default": _ASSET_ROOT,
    "/persistent/isaac/asset_root/nvidia": _ASSET_ROOT,
    "/persistent/isaaclab/asset_root/pretrained_checkpoints": None,
}

# NVIDIA-only USD schema modules that usd-core does not ship.
_MISSING_PXR_SCHEMAS = ("PhysxSchema", "PhysicsSchemaTools", "Semantics")


class ShimSettingsError(RuntimeError):
    """A carb setting outside :data:`KNOWN_SETTINGS` was read through the shim."""


class _StrictSettings:
    """``carb.settings`` stand-in that only answers for curated keys."""

    def get(self, key: str) -> str | None:
        if key in KNOWN_SETTINGS:
            return KNOWN_SETTINGS[key]
        raise ShimSettingsError(
            f"carb settings key {key!r} is not in lab2mj.isaac_shims.KNOWN_SETTINGS; "
            "add its real value there rather than letting a stub value leak into a config dump"
        )


class _InertMeta(type):
    """Metaclass giving inert placeholder classes chained attribute access."""

    def __getattr__(cls, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _make_inert(f"{cls.__shim_qualname__}.{name}")
        setattr(cls, name, sub)
        return sub

    def __repr__(cls):
        return f"<isaac-shim inert {cls.__shim_qualname__}>"


def _inert_init(self, *args, **kwargs) -> None:
    pass


def _inert_instance_getattr(self, name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return _make_inert(f"{type(self).__shim_qualname__}().{name}")()


def _inert_instance_call(self, *args, **kwargs):
    return self


def _make_inert(qualname: str) -> type:
    """Create an inert placeholder class for the shimmed attribute ``qualname``.

    A class (not an instance) so that runtime types like
    ``isaacsim.core.api.simulation_context.SimulationContext`` can be subclassed.
    ``__name__``/``__module__`` mirror the real attribute so ``configclass`` and
    ``callable_to_string`` introspection behave as they would on the real module.
    """
    name = qualname.rsplit(".", 1)[-1]
    module = qualname.rsplit(".", 1)[0] if "." in qualname else qualname
    return _InertMeta(
        name,
        (),
        {
            "__module__": module,
            "__shim_qualname__": qualname,
            "__init__": _inert_init,
            "__getattr__": _inert_instance_getattr,
            "__call__": _inert_instance_call,
        },
    )


class _ShimModule(types.ModuleType):
    """Module whose unknown attributes resolve to inert placeholder classes."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        val = _make_inert(f"{self.__name__}.{name}")
        setattr(self, name, val)
        return val


class _ShimLoader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        mod = _ShimModule(spec.name)
        mod.__path__ = []
        return mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _ShimFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespaces: tuple[str, ...]) -> None:
        self.namespaces = namespaces

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname.split(".")[0] in self.namespaces:
            return importlib.machinery.ModuleSpec(fullname, _ShimLoader(), is_package=True)
        return None


def is_installed() -> bool:
    """Whether the shim finder is on ``sys.meta_path``."""
    return any(isinstance(finder, _ShimFinder) for finder in sys.meta_path)


def install() -> None:
    """Install the shims. Idempotent. Must run before any isaaclab config import.

    Raises:
        RuntimeError: If a real Omniverse runtime module is already imported (the
            shims are for machines without Isaac Sim), or if a settings-sensitive
            isaaclab module was already imported without the shims (its module
            constants would hold values the shim did not control).
    """
    if is_installed():
        return
    for ns in _SHIMMED_NAMESPACES:
        if ns in sys.modules and not isinstance(sys.modules[ns], _ShimModule):
            raise RuntimeError(
                f"real {ns!r} module already imported; isaac_shims is only for machines without Isaac Sim"
            )
    for mod_name in _SETTINGS_SENSITIVE_MODULES:
        if mod_name in sys.modules:
            raise RuntimeError(f"{mod_name!r} was imported before isaac_shims.install(); its constants are untrusted")

    missing = tuple(pkg for pkg in _OPTIONAL_PACKAGES if importlib.util.find_spec(pkg) is None)
    sys.meta_path.insert(0, _ShimFinder(_SHIMMED_NAMESPACES + missing))

    # carb.settings.get_settings() must hand out the strict settings object even when
    # callers only ``import carb`` — create both modules eagerly and link them.
    carb = importlib.import_module("carb")
    carb_settings = importlib.import_module("carb.settings")
    strict = _StrictSettings()
    carb_settings.get_settings = lambda: strict  # ty: ignore[unresolved-attribute]
    carb.settings = carb_settings  # ty: ignore[unresolved-attribute]

    # usd-core provides real pxr modules; only the NVIDIA-only schemas are stubbed.
    import pxr

    for schema in _MISSING_PXR_SCHEMAS:
        if not hasattr(pxr, schema):
            mod = _ShimModule(f"pxr.{schema}")
            sys.modules[f"pxr.{schema}"] = mod
            setattr(pxr, schema, mod)
