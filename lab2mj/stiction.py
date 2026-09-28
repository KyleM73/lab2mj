"""PhysX >= 5 static joint friction (stiction) emulation for converted bundles.

PhysX's joint-friction triple (PxJointAxis) splits Coulomb friction into a
STATIC effort — acting only on stationary joints as a breakaway threshold —
and a DYNAMIC effort resisting motion. MuJoCo's ``dof_frictionloss`` is
single-valued: its stiction limit equals its moving Coulomb effort. The
converter therefore authors ``dof_frictionloss = dynamic_friction``, and this
module supplies the missing static behavior at runtime.

Measured PhysX capture rule (Spot free-fall dump, static 0.18 / dynamic 0
knees): a joint whose stopping impulse fits within one physics step —
``|qd| * M_jj <= static_effort * physics_dt`` — has its velocity zeroed
exactly and is then held while the net joint torque stays below the static
effort (the last pre-capture ``|qd|`` matched ``static * dt / M`` to ~2 %,
and the held joint reads ``qd == 0`` to float32 noise afterwards).

:class:`JointStiction` mirrors that as a Karnopp switch with hysteresis on
the model's friction bound, applied before every MuJoCo step:

* a SLIDING joint carries the dynamic bound; it becomes STUCK when it enters
  the capture window (MuJoCo's frictionloss constraint then zeroes and holds
  it, exactly like PhysX's capture);
* a STUCK joint that has not yet been zeroed is *capturing*: the friction
  row removes at most ``static * substep_dt / M`` of velocity per MuJoCo
  substep, so completing PhysX's one-physics-step capture takes up to
  ``physics_substeps`` substeps. It stays stuck while its speed keeps
  falling; capture completes when ``|qd| <= SLIP_EPS`` (the joint is HELD).
  A capturing joint whose speed stops falling is saturated (the net torque
  exceeds the static bound) and is released — without waiting for it to
  leave the window;
* a HELD joint is released the moment it actually slips
  (``|qd| > SLIP_EPS``) — i.e. the instant the holding torque saturates the
  static bound — because PhysX applies the DYNAMIC effort to any moving
  joint. Without the hysteresis the released joint would keep grinding
  against the static bound until it exits the capture window (measured: a
  65 ms-late breakaway on fl_kn under a slowly-rising load);
* a released joint stays SLIDING: it cannot re-stick while it keeps slipping
  inside the capture window (a joint that just broke away is still slow
  enough to re-enter it every substep, which would alternate the
  static/dynamic bounds and re-create the late-breakaway grind). It becomes
  capturable again once it leaves the window or comes to rest.

Friction rows of EVERY robot joint (not only the stiction joints) carry the near-hard
impedance below: MuJoCo's default friction-row regularization makes ``frictionloss`` a
velocity-ramped soft Coulomb (measured on a G1 walking rollout with 1.2-2.4 N*m dynamic
friction: 20-40 % of the bound below 0.05 rad/s, 78 % at 0.2-0.5 rad/s, 92 % at 1-2 rad/s),
while PhysX bounds a moving joint's friction constraint at the full dynamic effort. Hardened,
the applied friction is >= 95 % of the bound above 0.02 rad/s.

``M_jj`` is the diagonal of the joint-space inertia (armature included) —
exact for leaf joints, an approximation of the articulated stopping impulse
elsewhere, where it only shifts the capture instant by O(one step). The
window uses the ISAAC physics dt (PhysX performs the capture within one of
its own steps) regardless of the finer MuJoCo substep the model integrates
at.
"""

from __future__ import annotations

import mujoco
import numpy as np

from lab2mj.bundle import XML_RTOL

__all__ = ["JointStiction"]

# A held joint's residual velocity under the hardened friction row measured
# ~2e-6 rad/s at maximum static load; a truly slipping joint crosses 1e-5
# within a fraction of a millisecond for any saturation excess above ~1 % of
# the static effort.
SLIP_EPS = 1.0e-5


def _harden_friction_rows(model: mujoco.MjModel, dof_adr: np.ndarray) -> None:
    """Near-hard impedance on the friction rows of ``dof_adr`` (see module docstring).

    PhysX holds a captured joint at qd == 0 exactly (float32 noise) and applies the full
    dynamic effort to a moving joint; MuJoCo's DEFAULT friction-row regularization lets a
    loaded stuck joint creep at ~mrad/s, which mis-times the breakaway (measured: a 5 mrad
    creep shifted fl_kn's release by 50 ms and cost 0.5 rad downstream), and ramps the
    moving friction with speed. The stiffest stable reference removes both. Rows exist only
    for joints with nonzero ``frictionloss``, so frictionless joints are unaffected.
    """
    model.dof_solimp[dof_adr, 0] = 0.9999
    model.dof_solimp[dof_adr, 1] = 0.9999
    model.dof_solref[dof_adr, 0] = 2.0 * model.opt.timestep
    model.dof_solref[dof_adr, 1] = 1.0


class JointStiction:
    """Per-step static-friction switch for the joints whose static effort exceeds the dynamic one.

    Args:
        model: Compiled model whose ``dof_frictionloss`` carries the DYNAMIC
            friction efforts (the converter's authoring).
        dof_adr: Per-joint dof addresses, aligned with ``static``/``dynamic``.
        static: Per-joint static friction efforts (PhysX ``friction`` cfg field).
        dynamic: Per-joint dynamic friction efforts.
        physics_dt: The ISAAC physics step the capture window is computed
            against (NOT the MuJoCo substep timestep).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        dof_adr: np.ndarray,
        static: np.ndarray,
        dynamic: np.ndarray,
        physics_dt: float,
    ) -> None:
        static = np.asarray(static, dtype=np.float64)
        dynamic = np.asarray(dynamic, dtype=np.float64)
        dof_adr = np.asarray(dof_adr, dtype=np.intp)
        if not (static.shape == dynamic.shape == dof_adr.shape):
            raise ValueError(f"shape mismatch: static {static.shape}, dynamic {dynamic.shape}, dofs {dof_adr.shape}")
        _harden_friction_rows(model, dof_adr)
        mask = static > dynamic
        self.dof_adr = dof_adr[mask]
        self.static = static[mask]
        self.dynamic = dynamic[mask]
        self.physics_dt = float(physics_dt)
        self.active = bool(self.dof_adr.size)
        if self.active:
            authored = model.dof_frictionloss[self.dof_adr]
            # The exact float64 dynamic values are stamped back below.
            if not np.allclose(authored, self.dynamic, rtol=XML_RTOL, atol=1e-12):
                raise ValueError(
                    f"model dof_frictionloss {authored} != dynamic friction {self.dynamic} on the stiction joints; "
                    "the bundle was not produced by lab2mj.convert or is stale"
                )
            model.dof_frictionloss[self.dof_adr] = self.dynamic
            # Addresses of the diagonal M_jj in the sparse inertia: legacy qM (< 3.3) or CSR M.
            if hasattr(model, "M_rowadr"):
                self._m_diag_adr = model.M_rowadr[self.dof_adr] + model.M_rownnz[self.dof_adr] - 1
            else:
                self._m_diag_adr = model.dof_Madr[self.dof_adr]
        self._static_dt = self.static * self.physics_dt
        self._stuck = np.zeros(self.dof_adr.shape, dtype=bool)
        self._held = np.zeros(self.dof_adr.shape, dtype=bool)
        self._sliding = np.zeros(self.dof_adr.shape, dtype=bool)
        self._prev_qd = np.full(self.dof_adr.shape, np.inf)

    def reset(self) -> None:
        """Forget the stick state (episode reset); capture re-detects on the next apply."""
        self._stuck[:] = False
        self._held[:] = False
        self._sliding[:] = False
        self._prev_qd[:] = np.inf

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Set each stiction joint's friction bound for the substep about to integrate.

        Call once before every ``mj_step``. Sliding joints stick when they
        enter the capture window ``|qd| <= static * physics_dt / M_jj``.
        A stuck joint is capturing until the friction row has zeroed it
        (``|qd| <= SLIP_EPS`` -> held): it stays stuck while its speed keeps
        falling — the capture spans up to ``physics_substeps`` substeps — and
        is released if the speed stops falling (saturated mid-capture). A held
        joint is released the moment it actually slips (``|qd| > SLIP_EPS``);
        it then stays sliding until it leaves the window or comes to rest.
        """
        if not self.active:
            return
        m_diag = (data.M if hasattr(data, "M") else data.qM)[self._m_diag_adr]
        qd = np.abs(data.qvel[self.dof_adr])
        slipping = qd > SLIP_EPS
        # |qd| <= static * physics_dt / M_jj, in multiply form (all quantities >= 0).
        in_window = qd * np.maximum(m_diag, 1e-12) <= self._static_dt
        held_prev, stuck_prev = self._held, self._stuck
        capturing = stuck_prev & ~held_prev
        release = slipping & (held_prev | (capturing & (qd >= self._prev_qd)))
        # A released joint is SLIDING and must not re-stick while it still slips
        # inside the capture window (a just-broken-away joint re-enters it every
        # substep); the latch clears once it leaves the window or comes to rest.
        sliding = release | (self._sliding & slipping & in_window)
        self._stuck = (stuck_prev | in_window) & ~sliding
        self._held = stuck_prev & (held_prev | ~slipping) & ~release
        self._sliding = sliding
        self._prev_qd = qd
        model.dof_frictionloss[self.dof_adr] = np.where(self._stuck, self.static, self.dynamic)
