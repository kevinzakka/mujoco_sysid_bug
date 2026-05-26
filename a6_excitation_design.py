#!/usr/bin/env python3
"""A6. A slow+fast designed excitation recovers every parameter. See README.

Compares worst-determined sensitivity across designs, then runs the full
identification with the mix + light prior, scored against truth.
Usage: a6_excitation_design.py [N_SLOW N_FAST N_STEPS ITERS]
"""

from __future__ import annotations

import sys

import numpy as np
from mujoco import minimize as mj_minimize
from mujoco import sysid

import common as C
import realistic as B

N_SLOW = int(sys.argv[1]) if len(sys.argv) > 1 else 2
N_FAST = int(sys.argv[2]) if len(sys.argv) > 2 else 2
N_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 120
SLOW_FREQ, FAST_SPEED = 0.4, 2.5
EPS = np.finfo(np.float64).eps ** 0.5

tm, base_model, base_spec, key = B.make_truth(0)
dt = float(tm.opt.timestep)
jn = key["joints"]


def truth_x(params):
  pn = params.get_non_frozen_parameter_names()
  x = params.as_vector().copy()
  for j, nm in enumerate(pn):
    if "_inertia[" in nm:
      b = nm.split("_inertia[")[0]
      c = int(nm.split("[")[1].rstrip("]"))
      x[j] = key["theta_true"][b][c]
    elif nm.endswith("_armature"):
      x[j] = key["arm_true"][jn.index(nm.rsplit("_", 1)[0])]
    elif nm.endswith("_damping"):
      x[j] = key["damp_true"][jn.index(nm.rsplit("_", 1)[0])]
    elif nm.endswith("_frictionloss"):
      x[j] = key["fric_true"][jn.index(nm.rsplit("_", 1)[0])]
  return x


def worst_sens(commands):
  """Minimum friction and damping parameter sensitivity over the joints.

  D-optimality (total information volume) is dominated by the easy inertia/damping
  directions and under-rewards friction, so it misleadingly prefers all-fast. The
  binding constraint for a complete identification is the WORST-determined
  parameter (an E-optimal view): a good experiment maximizes the minimum
  sensitivity, keeping both friction and damping observable.
  """
  params, ms = B.build(seed=0, n_traj=len(commands), n_steps=N_STEPS, commands=commands)[:2]
  lo, hi = params.get_bounds()
  x = np.clip(truth_x(params), lo, hi)
  rfn = sysid.build_residual_fn(models_sequences=[ms])

  def res(z):
    r, _, _ = rfn(z, params)
    return np.concatenate(r)

  J = np.asarray(
    mj_minimize.jacobian_fd(
      residual=res,
      x=x.reshape(-1, 1),
      r=res(x.reshape(-1, 1)),
      eps=EPS,
      n_res=0,
      bounds=[lo.reshape(-1, 1), hi.reshape(-1, 1)],
    )[0],
    np.float64,
  )
  pn = params.get_non_frozen_parameter_names()
  ref = B.ref_scale(params, key)
  s = np.linalg.norm(J, axis=0) * ref  # fixed-scaled sensitivity
  fric = min(s[pn.index(n)] for n in pn if n.endswith("_frictionloss"))
  damp = min(s[pn.index(n)] for n in pn if n.endswith("_damping"))
  arm = min(s[pn.index(n)] for n in pn if n.endswith("_armature"))
  return fric, damp, arm


total = N_SLOW + N_FAST
slow = [B.slow_signal(tm, N_STEPS, dt, seed=300 + i, freq=SLOW_FREQ) for i in range(total)]
fast = [B.multisine(tm, N_STEPS, dt, seed=400 + i, speed=FAST_SPEED) for i in range(total)]
mixed = [B.slow_signal(tm, N_STEPS, dt, seed=500 + i, freq=SLOW_FREQ) for i in range(N_SLOW)] + [
  B.multisine(tm, N_STEPS, dt, seed=600 + i, speed=FAST_SPEED) for i in range(N_FAST)
]

print(f"=== A6. Excitation design ({total} trajectories each) ===")
print("1) worst-determined sensitivity per design (higher = better). friction wants")
print("   slow motion; damping and armature want fast motion. Each single-speed")
print("   design starves the term that needs the opposite motion; only the mix keeps")
print("   all three observable:")
print(f"   {'design':<22}{'min friction':>14}{'min damping':>13}{'min armature':>14}")
for name, cmds in (
  ("all-fast", fast),
  ("all-slow", slow),
  (f"mix ({N_SLOW} slow+{N_FAST} fast)", mixed),
):
  fr, dm, ar = worst_sens(cmds)
  print(f"   {name:<22}{fr:>14.3f}{dm:>13.3f}{ar:>14.3f}")

print("\n2) Full identification with the mix + light prior (W=0.3):")
params, ms, _, _ = B.build(seed=0, n_traj=len(mixed), n_steps=N_STEPS, commands=mixed)
pn = params.get_non_frozen_parameter_names()
center = params.as_vector().copy()
C.interior_start(params)  # damping/frictionloss start interior, not at the 1e-8 bound
sigma = np.full(len(pn), np.inf)
for j, nm in enumerate(pn):
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
W = 0.3


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
print(f"   data ||r|| = {np.linalg.norm(rdata):.4g}\n")


def ev(name):
  return float(np.atleast_1d(opt_params[name].value)[0])


print(f"   {'joint':<11}{'damping':>20}{'armature':>20}{'frictionloss':>22}")
for i, j in enumerate(jn):
  dtv, amv, frv = key["damp_true"][i], key["arm_true"][i], key["fric_true"][i]

  def cell(tv, a):
    e = ev(f"{j}_{a}")
    return f"{tv:.2f}->{e:.2f} ({100 * abs(e - tv) / abs(tv):.0f}%)"

  dmp = cell(dtv, "damping")
  arm = cell(amv, "armature")
  frc = cell(frv, "frictionloss")
  print(f"   {j:<11}{dmp:>20}{arm:>20}{frc:>22}")

rec = C.recovered_model(opt_params)
print("\n   inertia (truth -> estimated):")
print(f"   {'body':<10}{'mass t->e (err%)':>22}{'|com err| m':>14}{'trace t->e (err%)':>22}")
for b in key["bodies"]:
  ri, ti = C.body_id(rec, b), C.body_id(tm, b)
  em, t_m = float(rec.body_mass[ri]), float(tm.body_mass[ti])
  ec, tc = np.array(rec.body_ipos[ri]), np.array(tm.body_ipos[ti])
  et, tt = float(np.sum(rec.body_inertia[ri])), float(np.sum(tm.body_inertia[ti]))
  c1 = f"{t_m:.2f}->{em:.2f} ({100 * abs(em - t_m) / t_m:.0f}%)"
  c3 = f"{tt:.4f}->{et:.4f} ({100 * abs(et - tt) / tt:.0f}%)"
  print(f"   {b:<10}{c1:>22}{np.linalg.norm(ec - tc):>14.4f}{c3:>22}")
