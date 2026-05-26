#!/usr/bin/env python3
"""A4. A light prior toward CAD makes the inertials physical (realistic benchmark). See README.

Prior: inertia -> CAD, armature -> theory, damping/frictionloss free.
Usage: a4_regularization.py [W ITERS SEED N_TRAJ N_STEPS]
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from mujoco import sysid

import common as C
import realistic as B

W = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_TRAJ = int(sys.argv[4]) if len(sys.argv) > 4 else 10
N_STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 1200

params, ms, key, base_model = B.build(seed=SEED, n_traj=N_TRAJ, n_steps=N_STEPS)
pnames = params.get_non_frozen_parameter_names()
center = params.as_vector().copy()  # prior center: theta_CAD / arm_theory
C.interior_start(params)  # damping/frictionloss start interior, not at the 1e-8 bound
sigma = np.full(len(pnames), np.inf)
for j, nm in enumerate(pnames):
  if "_inertia[" in nm:
    b = nm.split("_inertia[")[0]
    c = int(nm.split("[")[1].rstrip("]"))
    sigma[j] = key["sigma"][b][c]
  elif nm.endswith("_armature"):
    sigma[j] = 0.3 * max(center[j], 1e-3)
reg = np.where(np.isfinite(sigma))[0]
inv = 1.0 / sigma[reg]
cen = center[reg]

base_rfn = sysid.build_residual_fn(models_sequences=[ms])


def reg_rfn(x, p, **kw):
  res, preds, recs = base_rfn(x, p, **kw)
  rows = W * (x[reg] - cen) * inv if x.ndim == 1 else W * (x[reg, :] - cen[:, None]) * inv[:, None]
  return list(res) + [rows], preds, recs


opt_params, _ = sysid.optimize(
  initial_params=params,
  residual_fn=reg_rfn,
  optimizer="scipy_parallel_fd",
  verbose=False,
  max_iters=ITERS,
)
rdata = np.concatenate(base_rfn(opt_params.as_vector(), opt_params)[0])
print(f"=== A4. Regularized recovery, realistic benchmark (W={W}) ===")
print(f"data ||r|| = {np.linalg.norm(rdata):.4g}\n")

jn = key["joints"]


def ev(name):
  return float(np.atleast_1d(opt_params[name].value)[0])


print("dynamics (true -> estimated, rel err %):")
print(f"  {'joint':<11}{'damping':>22}{'armature':>22}{'friction':>22}")
for i, j in enumerate(jn):
  dt_, am_, fr_ = key["damp_true"][i], key["arm_true"][i], key["fric_true"][i]

  def cell(tv, a):
    e = ev(f"{j}_{a}")
    return f"{tv:.2f}->{e:.2f} ({100 * abs(e - tv) / abs(tv):.0f}%)"

  print(
    f"  {j:<11}{cell(dt_, 'damping'):>22}{cell(am_, 'armature'):>22}{cell(fr_, 'frictionloss'):>22}"
  )

rec = C.recovered_model(opt_params)
true_model, _, _, _ = B.make_truth(SEED)


def bid(m, nm):
  return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY.value, nm)


print("\ninertia (truth -> estimated):")
print(f"  {'body':<10}{'mass t->e (err%)':>24}{'|com err| m':>14}{'trace t->e (err%)':>22}")
for b in key["bodies"]:
  ri, ti = bid(rec, b), bid(true_model, b)
  em, tm = float(rec.body_mass[ri]), float(true_model.body_mass[ti])
  ec, tc = np.array(rec.body_ipos[ri]), np.array(true_model.body_ipos[ti])
  et, tt = float(np.sum(rec.body_inertia[ri])), float(np.sum(true_model.body_inertia[ti]))
  c1 = f"{tm:.2f}->{em:.2f} ({100 * abs(em - tm) / tm:.0f}%)"
  c3 = f"{tt:.4f}->{et:.4f} ({100 * abs(et - tt) / tt:.0f}%)"
  print(f"  {b:<10}{c1:>24}{np.linalg.norm(ec - tc):>14.4f}{c3:>22}")
