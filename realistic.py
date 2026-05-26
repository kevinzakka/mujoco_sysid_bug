#!/usr/bin/env python3
"""Realistic sysid benchmark built on the reporter's 7-dof arm.

The reporter's benchmark made the synthetic truth adversarially far from CAD,
which is unrealistic and incompatible with any honest prior. Here we instead take
the CAD model (base.xml) and perturb it within physical tolerances to produce the
truth, so that CAD is a good prior, the way real system identification works.

Construction:
  * CAD = base.xml inertials (the prior center).
  * truth inertials = CAD perturbed in log-Cholesky theta space:
      theta_true = theta_CAD + N(0, SIGMA)
    SIGMA is per-component physical: log-mass ~20%, log-stretch ~20%, shear,
    com ~2 cm. Perturbing in theta space keeps the pseudo-inertia positive
    definite by construction.
  * armature: theoretical value per joint (rotor inertia x gear^2); true armature
    = theory x (1 +/- ~10%). The theory value is a prior.
  * damping, frictionloss: realistic true values, identified from data (no prior).
  * commands: reuse the existing trajN.npz setpoints; only the model changes.

Everything is seeded and reproducible.
"""

from __future__ import annotations

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

import common as C

# Per-component prior std in log-Cholesky theta space: [alpha, d1,d2,d3, s12,s23,s13, t1,t2,t3].
# alpha=log-mass (~20%), d=log-stretch (~20%), com in meters (2 cm). The shear (s)
# entries are filled PER BODY, scaled to that body's diagonal stretch, because a
# fixed shear is huge relative to a thin link's tiny Cholesky diagonal.
SIGMA_BASE = np.array([0.10, 0.10, 0.10, 0.10, 0.0, 0.0, 0.0, 0.02, 0.02, 0.02])
SHEAR_FRAC = 0.20  # shear std as a fraction of the body's mean diagonal stretch


def sigma_for(theta_cad):
  """Per-component prior std for one body, with shear scaled to its diagonal."""
  s = SIGMA_BASE.copy()
  shear_scale = float(np.exp(theta_cad[1:4]).mean())  # mean principal stretch
  s[4:7] = SHEAR_FRAC * shear_scale
  return s


# Theoretical armature (rotor inertia x gear^2), larger for proximal joints.
ARM_THEORY = np.array([0.30, 0.30, 0.20, 0.15, 0.08, 0.08, 0.05])
# True viscous damping and Coulomb frictionloss (what we sysID; no prior).
DAMP_TRUE = np.array([2.0, 2.0, 1.5, 1.0, 0.5, 0.5, 0.3])
FRIC_TRUE = np.array([0.8, 0.8, 0.5, 0.4, 0.2, 0.2, 0.15])

WIDE = dict(
  mass_bound_mult=np.array([0.1, 10.0]),
  ipos_bound_off=np.array([-0.3, 0.3]),
  stretch_bound_mult=np.array([0.2, 5.0]),
  shear_bound_off=np.array([-2.0, 2.0]),
)


def make_truth(seed=0):
  """Build the true model and the answer key (deterministic given seed)."""
  rng = np.random.default_rng(seed)
  base_spec = mujoco.MjSpec.from_file(str(C.BASE_XML))
  base_model = base_spec.compile()
  jn = [j.name for j in base_spec.joints]
  bn = [b.name for b in base_spec.bodies if b.name != "world"]

  true_spec = mujoco.MjSpec.from_file(str(C.BASE_XML))
  key = {"theta_cad": {}, "theta_true": {}, "sigma": {}, "bodies": bn, "joints": jn}
  for b in bn:
    p = sysid.body_inertia_param(
      true_spec,
      base_model,
      b,
      inertia_type=sysid.InertiaType.Pseudo,
      param_name=f"{b}_inertia",
      **WIDE,
    )
    theta_cad = p.nominal.copy()
    sig = sigma_for(theta_cad)
    theta_true = theta_cad + rng.normal(0.0, sig)
    p.value = theta_true
    p.modifier(true_spec, p)  # write truth into the spec
    key["theta_cad"][b] = theta_cad
    key["theta_true"][b] = theta_true
    key["sigma"][b] = sig

  arm_true = ARM_THEORY * (1.0 + rng.normal(0.0, 0.10, size=len(jn)))
  for i, j in enumerate(jn):
    joint = true_spec.joint(j)
    joint.armature = float(arm_true[i])
    joint.damping[0] = float(DAMP_TRUE[i])
    joint.frictionloss = float(FRIC_TRUE[i])
  key["arm_theory"] = ARM_THEORY.copy()
  key["arm_true"] = arm_true
  key["damp_true"] = DAMP_TRUE.copy()
  key["fric_true"] = FRIC_TRUE.copy()

  true_model = true_spec.compile()
  return true_model, base_model, base_spec, key


def build(seed=0, n_traj=5, n_steps=1200, commands=None):
  """Return everything needed to identify: params, residual_fn, prior, key, ms.

  commands: optional list of joint-position command arrays (each n_steps x njoint).
    If None, the reused trajN.npz setpoints are used. Pass designed excitations
    here to evaluate optimal experiment design.
  """
  true_model, base_model, base_spec, key = make_truth(seed)
  jn = key["joints"]
  bn = key["bodies"]
  dt = float(true_model.opt.timestep)
  an = [f"act_{j}" for j in jn]
  sn = [f"{j}_pos" for j in jn] + [f"{j}_vel" for j in jn]

  if commands is None:
    cmd_list = []
    for i in range(1, n_traj + 1):
      q, _ = C.load_trajectory(i, n_steps)
      cmd_list.append(q)
  else:
    cmd_list = [np.asarray(c) for c in commands]

  ist, cts, sts, names = [], [], [], []
  for idx, q in enumerate(cmd_list):
    t = np.arange(q.shape[0]) * dt
    s0 = sysid.create_initial_state(
      true_model, qpos=q[0], qvel=np.zeros(true_model.nv), act=np.zeros(true_model.na)
    )
    st, sd = rollout.rollout(true_model, mujoco.MjData(true_model), s0, q[:-1])
    st = np.squeeze(st, 0)
    sd = np.squeeze(sd, 0)
    ist.append(s0)
    names.append(f"traj{idx + 1}")
    cts.append(sysid.TimeSeries.from_control_names(t, q, true_model, names=an))
    sts.append(sysid.TimeSeries.from_names(st[:, 0], sd, true_model, names=sn))
  ms = sysid.ModelSequences(
    "arm", base_spec.copy(), names, ist, cts, sts, allow_missing_sensors=True
  )

  params = sysid.ParameterDict()
  ps = base_spec.copy()
  pm = ps.compile()
  for b in bn:
    params.add(
      sysid.body_inertia_param(
        ps, pm, b, inertia_type=sysid.InertiaType.Pseudo, param_name=f"{b}_inertia", **WIDE
      )
    )
  for i, j in enumerate(jn):
    params.add(
      sysid.Parameter(
        f"{j}_armature",
        nominal=float(key["arm_theory"][i]),
        min_value=0.0,
        max_value=float(key["arm_theory"][i] * 4 + 0.1),
        modifier=C.make_param_modifier(j, "armature"),
      )
    )
    params.add(
      sysid.Parameter(
        f"{j}_damping",
        nominal=1e-8,
        min_value=0.0,
        max_value=20.0,
        modifier=C.make_param_modifier(j, "damping"),
      )
    )
    params.add(
      sysid.Parameter(
        f"{j}_frictionloss",
        nominal=1e-8,
        min_value=0.0,
        max_value=10.0,
        modifier=C.make_param_modifier(j, "frictionloss"),
      )
    )

  return params, ms, key, base_model


def multisine(model, n_steps, dt, seed, speed=1.0, n_harm=6, amp_frac=0.8):
  """Hann-windowed finite Fourier (Swevers-style) excitation, one per joint.

  q_j(t) = mid_j + w(t) * sum_k A_jk sin(2 pi * speed * k * t / T + phi_jk),
  with w a Hann window (start/end at rest at the joint center) and
  sum_k |A_jk| = amp_frac * halfrange_j so the command stays inside joint limits.
  `speed` multiplies every harmonic frequency, scaling velocity.
  """
  rng = np.random.default_rng(seed)
  lo = model.jnt_range[:, 0]
  hi = model.jnt_range[:, 1]
  mid = 0.5 * (lo + hi)
  half = 0.5 * (hi - lo)
  t = np.arange(n_steps) * dt
  T = n_steps * dt
  w = 0.5 * (1.0 - np.cos(2 * np.pi * t / T))
  q = np.tile(mid, (n_steps, 1))
  for j in range(model.njnt):
    a = rng.uniform(0.3, 1.0, n_harm)
    a *= amp_frac * half[j] / a.sum()
    ph = rng.uniform(0, 2 * np.pi, n_harm)
    q[:, j] = mid[j] + w * sum(
      a[k] * np.sin(2 * np.pi * speed * (k + 1) * t / T + ph[k]) for k in range(n_harm)
    )
  return np.clip(q, lo + 1e-6, hi - 1e-6)


def slow_signal(model, n_steps, dt, seed, freq=0.35, amp_frac=0.30):
  """Genuinely-slow, single-low-frequency excitation for identifying frictionloss.

  Each joint follows one low-frequency Hann-windowed sine, so velocity stays low
  and reverses sign, the regime where Coulomb friction dominates the joint torque
  balance (the position-control analog of ramping torque until the joint breaks
  free). Unlike a multi-sine at low `speed`, the single low frequency keeps it
  actually slow.
  """
  rng = np.random.default_rng(seed)
  lo = model.jnt_range[:, 0]
  hi = model.jnt_range[:, 1]
  mid = 0.5 * (lo + hi)
  half = 0.5 * (hi - lo)
  t = np.arange(n_steps) * dt
  T = n_steps * dt
  w = 0.5 * (1.0 - np.cos(2 * np.pi * t / T))
  q = np.tile(mid, (n_steps, 1))
  for j in range(model.njnt):
    ph = rng.uniform(0, 2 * np.pi)
    q[:, j] = mid[j] + w * amp_frac * half[j] * np.sin(2 * np.pi * freq * t + ph)
  return np.clip(q, lo + 1e-6, hi - 1e-6)


def ref_scale(params, key):
  """FIXED per-parameter reference scale (prior sigma for inertia/armature,
  characteristic magnitudes for damping/frictionloss). Makes a D-optimality
  criterion dimensionless and comparable across excitation designs."""
  pn = params.get_non_frozen_parameter_names()
  jn = key["joints"]
  ref = np.ones(len(pn))
  for j, nm in enumerate(pn):
    if "_inertia[" in nm:
      b = nm.split("_inertia[")[0]
      c = int(nm.split("[")[1].rstrip("]"))
      ref[j] = key["sigma"][b][c]
    elif nm.endswith("_armature"):
      ref[j] = 0.3 * key["arm_theory"][jn.index(nm.rsplit("_", 1)[0])]
    elif nm.endswith("_damping"):
      ref[j] = 1.0
    elif nm.endswith("_frictionloss"):
      ref[j] = 0.5
  return ref


if __name__ == "__main__":
  tm, bm, bs, key = make_truth(0)
  print("realistic benchmark truth vs CAD (sanity check):")
  print(f"  {'joint':<11}{'armature theory->true':>24}{'damping true':>14}{'friction true':>15}")
  for i, j in enumerate(key["joints"]):
    arm = f"{key['arm_theory'][i]:.3f}->{key['arm_true'][i]:.3f}"
    print(f"  {j:<11}{arm:>24}{key['damp_true'][i]:>14.2f}{key['fric_true'][i]:>15.2f}")
  print("\n  body: mass and inertia-trace, CAD -> TRUE (should be a modest, physical change):")
  for b in key["bodies"]:
    ci = mujoco.mj_name2id(bm, mujoco.mjtObj.mjOBJ_BODY.value, b)
    ti = mujoco.mj_name2id(tm, mujoco.mjtObj.mjOBJ_BODY.value, b)
    cm, tmass = float(bm.body_mass[ci]), float(tm.body_mass[ti])
    ct, tt = float(np.sum(bm.body_inertia[ci])), float(np.sum(tm.body_inertia[ti]))
    print(
      f"    {b:<10} mass {cm:.2f}->{tmass:.2f} ({100 * (tmass - cm) / cm:+.0f}%)   "
      f"trace {ct:.4f}->{tt:.4f} ({100 * (tt - ct) / ct:+.0f}%)"
    )
