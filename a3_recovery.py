#!/usr/bin/env python3
"""A3. Feasible bounds: residual ~0 but inertia is garbage (non-identifiability). See README.

Usage: a3_recovery.py [N_TRAJ N_STEPS ITERS]
"""

from __future__ import annotations

import sys

import numpy as np
from mujoco import sysid

import common as C

N_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 150

params, ms, key, base_model = C.build_original("damping", N_TRAJ, N_STEPS, bounds="wide")
C.interior_start(params, fields=("damping", "frictionloss", "armature"))
mp = key["mp"]
target = key["target_model"]
rfn = sysid.build_residual_fn(models_sequences=[ms])
opt_params, _ = sysid.optimize(
  initial_params=params,
  residual_fn=rfn,
  optimizer="scipy_parallel_fd",
  verbose=False,
  max_iters=ITERS,
)
r = np.concatenate(rfn(opt_params.as_vector(), opt_params)[0])
rn = np.linalg.norm(r)
print("=== A3. Recovery on the reporter's benchmark (wide bounds, scaled solver) ===")
print(f"final ||r|| = {rn:.4g}  (the data is fit well; contrast with the inertia errors below)\n")


def ev(name):
  return float(np.atleast_1d(opt_params[name].value)[0])


print("joint dynamics (true -> estimated, rel err %):")
print(f"  {'joint':<11}{'damping':>22}{'armature':>22}{'friction':>22}")
for jn in mp.joint_names:
  d = mp.joint_dynamics[jn]

  def cell(tv, a):
    e = ev(f"{jn}_{a}")
    return f"{tv:.2f}->{e:.2f} ({100 * abs(e - tv) / max(abs(tv), 1e-9):.0f}%)"

  dmp = cell(d.damping, "damping")
  arm = cell(d.armature, "armature")
  fr = cell(d.frictionloss, "frictionloss")
  print(f"  {jn:<11}{dmp:>22}{arm:>22}{fr:>22}")

rec = C.recovered_model(opt_params)
print("\ninertia (truth -> estimated; note the large errors despite the tiny residual):")
print(f"  {'body':<10}{'mass t->e (err%)':>24}{'inertia trace t->e (err%)':>30}")
for b in mp.body_names:
  ri, ti = C.body_id(rec, b), C.body_id(target, b)
  em, tm = float(rec.body_mass[ri]), float(target.body_mass[ti])
  et, tt = float(np.sum(rec.body_inertia[ri])), float(np.sum(target.body_inertia[ti]))
  c1 = f"{tm:.2f}->{em:.2f} ({100 * abs(em - tm) / tm:.0f}%)"
  c2 = f"{tt:.4f}->{et:.4f} ({100 * abs(et - tt) / tt:.0f}%)"
  print(f"  {b:<10}{c1:>24}{c2:>30}")
