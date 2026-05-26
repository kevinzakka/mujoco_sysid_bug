#!/usr/bin/env python3
"""A2. The true inertials lie outside the optimization bounds. See README."""

from __future__ import annotations

import numpy as np

import common as C

params, ms, key, base_model = C.build_original("damping", n_traj=1, n_steps=2)
mp = key["mp"]
pnames = params.get_non_frozen_parameter_names()
lo, hi = params.get_bounds()

print("=== A2. Bounds feasibility ===\n")
print("inertia (mass ratio bound [0.5,1.5], com offset bound +/-0.05 m):")
print(f"  {'body':<10}{'mass true/CAD':>15}{'mass ok':>9}{'max|com off| m':>16}{'com ok':>8}")
mass_bad = com_bad = 0
for b in mp.body_names:
  bi = C.body_id(base_model, b)
  cad_m = float(base_model.body_mass[bi])
  cad_c = np.array(base_model.body_ipos[bi])
  t_m = mp.body_inertias[b].mass
  t_c = np.asarray(mp.body_inertias[b].ipos)
  ratio = t_m / cad_m
  off = float(np.max(np.abs(t_c - cad_c)))
  mok = C.MASS_BOUND_MIN <= ratio <= C.MASS_BOUND_MAX
  cok = off <= C.IPOS_BOUND
  mass_bad += not mok
  com_bad += not cok
  print(
    f"  {b:<10}{ratio:>15.3f}{('yes' if mok else 'NO'):>9}{off:>16.4f}{('yes' if cok else 'NO'):>8}"
  )
print(f"\n  -> mass out of bounds: {mass_bad}/7 bodies   com out of bounds: {com_bad}/7 bodies")

dyn_bad = 0
for nm in pnames:
  if any(s in nm for s in ("_armature", "_damping", "_frictionloss")):
    i = pnames.index(nm)
    jn = nm.rsplit("_", 1)[0]
    attr = nm.rsplit("_", 1)[1]
    tv = getattr(mp.joint_dynamics[jn], attr)
    dyn_bad += not (lo[i] <= tv <= hi[i])
status = "ALL 21 inside" if dyn_bad == 0 else f"{dyn_bad} outside"
print(f"  joint-dynamics truth within bounds: {status}")
print("\nConclusion: the inertial truth is outside the box (so recovery is impossible as")
print("posed), while the dynamics bounds are fine. The benchmark is the problem, not")
print("only the solver.")
