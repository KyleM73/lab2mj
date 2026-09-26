---
name: sim2sim-diagnose
description: Layered diagnosis for lab2mj validation-gate failures — which protocol isolates which error source, plus measured dead ends not worth re-trying.
---

# Diagnose sim2sim gate failures

Work the layers in order; each isolates one error source. All replays run on
mac CPU against pinned dumps.

## Layer order

1. **obs gate (first-observation parity) fails** → config/frame bug, never physics. Check:
   dump/bundle joint order (`check_dump_joint_order` raises on mismatch),
   command pinning (pose tasks pin the recorded world goal
   `command_<term>_goal_w`, not the CLI vector), measured-terrain bundles
   require the dump's exact terrain instance (env-origin check raises).
2. **Free-space plant** (`replay_freespace_mujoco.py` vs the robot's freespace
   dump) → isolates mass/inertia/armature/joint-friction/gravity. This is the
   diagnostic of choice when a robot's open_loop error ignores contact sweeps.
   Sensitivity flags (`--zero-armature` etc.) prove the test detects the
   parameter it targets.
3. **Contact drops** (`replay_contact_mujoco.py` vs the contact dump) →
   isolates the contact model at canonical impacts. The STIFF drop is the
   meaningful gate (healthy: pre-impact dq ≤0.026 rad, impact+5 ≈0.01–0.02);
   passive drops diverge post-impact by design (no damping).
4. **One-step parity** (`one_step_parity.py`) → per-step model error at matched
   states, free of chaotic compounding; spike joints/steps localize the
   mechanism. Caveat: DelayedPD/LSTM robots carry a reconstruction floor
   (mid-episode buffer contents can't be rebuilt), so absolute medians are only
   meaningful for implicit-PD robots.
5. **Contact calibration** (`calibrate_contact.py`, grid over solimp × impratio
   × solref × cone × noslip, scored by the open_loop gate) → apply winners via
   `lab2mj-convert --contact_solimp/--contact_impratio/--contact_solref`.
   A candidate MUST clear two bars before adoption: no regression across the
   bundle's whole dump ensemble (score every dump, not the failing one), and an
   on-box closed_loop gate check.

## Measured dead ends (don't re-run these)

- **Open-loop calibration gains do not transfer closed-loop**: softer contact
  tracks recorded trajectories better but changes the feedback dynamics the
  policy experiences — a candidate that improved the open_loop gate on 4/6 spot rows
  regressed the closed_loop gate on all six. Never adopt from open-loop
  evidence alone.
- The calibration grid (incl. cone/noslip) has been exhausted twice on the
  shipped fleet; residual spot fails (`spot_flat_stock_ens_fwd2` open_loop,
  `spot_velocity_ens_turn` closed_loop) are dump-specific contact divergence, not
  parameter-fixable without ensemble regressions.
- **Nav-task open_loop error is dominated by the frozen low-level policy's own closed-loop
  compounding** (informative `max_ll_action_err_*` metrics separate it) — no
  converter-level fix.
- Run-to-run matrix instability comes from re-dumped references (PhysX GPU
  nondeterminism), not the validator: same-machine validation is
  bit-deterministic on a fixed dump. Pin dumps; compare gate utilizations.
- contact_offset/margin conversion was a measured no-op; the batched
  contact-force rotate/scatter measured slower than the per-contact loop.

Regression oracle for any runtime change: `spot_velocity` +
`spot_vel_fwd_settled_phys.npz`, `--gates obs,open_loop` → obs 2.98e-08, open_loop 0.0302 rad
exactly (plus bit-identical fixture re-conversion for converter changes).
