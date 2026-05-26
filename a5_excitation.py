#!/usr/bin/env python3
"""A5. Distal joints are under-excited; the fast/slow sensitivity tradeoff. See README.

Reports per-joint velocity and sensitivity (baseline vs designed) and a speed sweep.
Usage: a5_excitation.py [N_TRAJ N_STEPS]
"""

from __future__ import annotations

import sys

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import minimize as mj_minimize
from mujoco import sysid

import common as C
import realistic as B

N_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800
N_HARM = 6
EPS = np.finfo(np.float64).eps ** 0.5


multisine = B.multisine  # shared implementation (Hann-windowed finite Fourier)


def truth_x(params, key):
  pn = params.get_non_frozen_parameter_names()
  jn = key["joints"]
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


def colnorms(params, ms, key):
  pn = params.get_non_frozen_parameter_names()
  lo, hi = params.get_bounds()
  x = np.clip(truth_x(params, key), lo, hi)
  rfn = sysid.build_residual_fn(models_sequences=[ms])

  def res(z):
    r, _, _ = rfn(z, params)
    return np.concatenate(r)

  r0 = res(x.reshape(-1, 1))
  J = np.asarray(
    mj_minimize.jacobian_fd(
      residual=res,
      x=x.reshape(-1, 1),
      r=r0,
      eps=EPS,
      n_res=0,
      bounds=[lo.reshape(-1, 1), hi.reshape(-1, 1)],
    )[0],
    np.float64,
  )
  return np.linalg.norm(J, axis=0), pn, J


def qvel_rms(tm, cmds):
  nv = tm.nv
  out = []
  for q in cmds:
    s0 = sysid.create_initial_state(tm, qpos=q[0], qvel=np.zeros(nv), act=np.zeros(tm.na))
    st, _ = rollout.rollout(tm, mujoco.MjData(tm), s0, q[:-1])
    out.append(np.squeeze(st, 0)[:, 1 + tm.nq : 1 + tm.nq + nv])
  return np.sqrt(np.mean(np.concatenate(out) ** 2, axis=0))


tm, base_model, base_spec, key = B.make_truth(0)
dt = float(tm.opt.timestep)
designed = [multisine(tm, N_STEPS, dt, seed=100 + i) for i in range(N_TRAJ)]
baseline_cmds = [C.load_trajectory(i, N_STEPS)[0] for i in range(1, N_TRAJ + 1)]

print("=== A5. Excitation design ===\n1) per-joint RMS velocity (baseline -> designed):")
vb, vd = qvel_rms(tm, baseline_cmds), qvel_rms(tm, designed)
for j in range(tm.nv):
  print(f"   joint{j + 1}: {vb[j]:.4f} -> {vd[j]:.4f}  ({vd[j] / max(vb[j], 1e-9):.0f}x)")

cnb, pn, _ = colnorms(*B.build(seed=0, n_traj=N_TRAJ, n_steps=N_STEPS)[:2], key)
cnd, _, _ = colnorms(*B.build(seed=0, n_traj=N_TRAJ, n_steps=N_STEPS, commands=designed)[:2], key)
print("\n2) absolute sensitivity |J col| for joint-dynamics params (baseline -> designed):")
print(f"   {'param':<26}{'baseline':>12}{'designed':>12}{'gain':>8}")
for nm in pn:
  if any(s in nm for s in ("_armature", "_damping", "_frictionloss")):
    j = pn.index(nm)
    print(f"   {nm:<26}{cnb[j]:>12.2e}{cnd[j]:>12.2e}{cnd[j] / max(cnb[j], 1e-30):>7.0f}x")

print("\n3) sensitivity vs excitation speed (friction wants slow, damping wants fast):")
print(f"   {'speed':>6}{'RMS vel':>10}{'fric sens':>11}{'damp sens':>11}")
for speed in (0.25, 0.5, 1.0, 2.0, 4.0):
  cmds = [multisine(tm, N_STEPS, dt, seed=200 + i, speed=speed) for i in range(N_TRAJ)]
  rms = float(np.sqrt(np.mean(qvel_rms(tm, cmds) ** 2)))
  params, ms = B.build(seed=0, n_traj=N_TRAJ, n_steps=N_STEPS, commands=cmds)[:2]
  cn, pn2, _ = colnorms(params, ms, key)
  ref = B.ref_scale(params, key)
  fric = np.mean([(cn * ref)[pn2.index(n)] for n in pn2 if n.endswith("_frictionloss")])
  damp = np.mean([(cn * ref)[pn2.index(n)] for n in pn2 if n.endswith("_damping")])
  print(f"   {speed:>6.2f}{rms:>10.3f}{fric:>11.2f}{damp:>11.2f}")
