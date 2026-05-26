#!/usr/bin/env python3
"""A7. Initialization decides whether the solve succeeds. See README.

Starting damping/frictionloss at their 1e-8 lower bound makes the optimizer fail on
some trajectory realizations; an interior start fixes it. The failure is not a
backend property: both the mujoco and scipy backends fail at the bound and converge
identically from an interior start.
Usage: a7_initialization.py [N_SEEDS N_STEPS ITERS]
"""

from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from mujoco import sysid

import common as C
import realistic as B

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 200

tm, _, _, key = B.make_truth(0)
dt = float(tm.opt.timestep)
jn = key["joints"]


def design(seed):
  return [B.slow_signal(tm, N_STEPS, dt, seed=seed * 31 + i, freq=0.4) for i in range(2)] + \
         [B.multisine(tm, N_STEPS, dt, seed=seed * 31 + 50 + i, speed=2.5) for i in range(2)]


def recover(cmds, optimizer="mujoco", interior=True):
  params, ms, _, _ = B.build(seed=0, n_traj=len(cmds), n_steps=cmds[0].shape[0], commands=cmds)
  center = params.as_vector().copy()
  if interior:
    C.interior_start(params)
  pn = params.get_non_frozen_parameter_names()
  sigma = np.full(len(pn), np.inf)
  for j, nm in enumerate(pn):
    if "_inertia[" in nm:
      b = nm.split("_inertia[")[0]
      sigma[j] = key["sigma"][b][int(nm.split("[")[1].rstrip("]"))]
    elif nm.endswith("_armature"):
      sigma[j] = 0.3 * max(center[j], 1e-3)
  reg = np.where(np.isfinite(sigma))[0]
  inv, cen = 1.0 / sigma[reg], center[reg]
  base_rfn = sysid.build_residual_fn(models_sequences=[ms])
  w_reg = 0.3

  def reg_rfn(x, p, **kw):
    r, pr, rc = base_rfn(x, p, **kw)
    rows = w_reg * (x[reg] - cen) * inv if x.ndim == 1 else \
        w_reg * (x[reg, :] - cen[:, None]) * inv[:, None]
    return list(r) + [rows], pr, rc

  with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
    opt, _ = sysid.optimize(initial_params=params, residual_fn=reg_rfn,
                            optimizer=optimizer, verbose=False, max_iters=ITERS)
  res = float(np.linalg.norm(np.concatenate(base_rfn(opt.as_vector(), opt)[0])))

  def ev(n):
    return float(np.atleast_1d(opt[n].value)[0])

  d = max(100 * abs(ev(f"{j}_damping") - key["damp_true"][i]) / key["damp_true"][i]
          for i, j in enumerate(jn))
  f = max(100 * abs(ev(f"{j}_frictionloss") - key["fric_true"][i]) / key["fric_true"][i]
          for i, j in enumerate(jn))
  return res, d, f


print(f"=== A7. Initialization (realistic benchmark, {N_SEEDS} seeds, mujoco backend) ===")
print("per seed, worst-joint damping/frictionloss error (residual in parentheses):")
print(f"  {'seed':>4}{'start at 1e-8 bound':>26}{'interior start':>22}")
for s in range(N_SEEDS):
  cmds = design(s)
  rb, db, fb = recover(cmds, interior=False)
  ri, di, fi = recover(cmds, interior=True)
  print(f"  {s:>4}     {db:>3.0f}/{fb:<3.0f}% (r={rb:>4.2f})        {di:>3.0f}/{fi:<3.0f}% (r={ri:>4.2f})")

print("\nnot a backend property -- both backends, seed 0 (worst damp% , residual):")
print(f"  {'backend':<20}{'1e-8 bound':>18}{'interior':>18}")
cmds = design(0)
for bk in ("mujoco", "scipy_parallel_fd"):
  rb, db, _ = recover(cmds, optimizer=bk, interior=False)
  ri, di, _ = recover(cmds, optimizer=bk, interior=True)
  print(f"  {bk:<20}{f'{db:.0f}% / {rb:.2f}':>18}{f'{di:.0f}% / {ri:.2f}':>18}")
