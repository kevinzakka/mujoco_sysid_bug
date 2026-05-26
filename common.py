"""Shared setup for the issue-3284 analysis (self-contained).

Builds the reporter's benchmark (identify base.xml to match the target model)
directly from the data files in the repo root, and exposes:

  * build_original(field, n_traj, n_steps, bounds) -> (params, ms, key, base_model)
  * jacobian_at(params, ms, x) -> finite-difference Jacobian of the residual
  * residual_vector(params, ms, x) -> concatenated residual
  * recovered_model(opt_params) -> compiled MjModel from estimated parameters

"key" carries the ground truth (the JSON parameters the synthetic data was
generated from) so every script scores RECOVERY against truth, not just residual.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import minimize as mj_minimize
from mujoco import sysid

DATA_DIR = pathlib.Path(__file__).resolve().parent
BASE_XML = DATA_DIR / "base.xml"
TARGET_XML = DATA_DIR / "target.xml"
PARAMS_JSON = DATA_DIR / "mujoco_model_parameters.json"

# Bounds / start values used by the reporter's reproducer.
BOUND_MULT = 3.0
MASS_BOUND_MIN, MASS_BOUND_MAX = 0.5, 1.5
IPOS_BOUND = 0.05
STRETCH_BOUND_MIN, STRETCH_BOUND_MAX = 0.5, 1.5
SHEAR_BOUND = 0.25
VALUE_POS, VALUE_ZERO = 1e-8, 0.0
DYNAMICS_ALIASES = {"friction": "frictionloss", "zero": "none"}

EPS = np.finfo(np.float64).eps ** 0.5


@dataclasses.dataclass(frozen=True)
class BodyInertia:
  mass: float
  ipos: np.ndarray
  fullinertia: np.ndarray


@dataclasses.dataclass(frozen=True)
class JointDynamics:
  armature: float
  damping: float
  frictionloss: float


@dataclasses.dataclass(frozen=True)
class ModelParameters:
  body_names: list
  joint_names: list
  body_inertias: dict
  joint_dynamics: dict


def canonical_dynamics_field(field):
  return DYNAMICS_ALIASES.get(field, field)


def _as_vector(payload, key, size):
  v = np.asarray(payload[key], dtype=np.float64)
  if v.shape != (size,):
    raise ValueError(f"expected {key} shape ({size},), got {v.shape}")
  return v


def load_model_parameters(path=PARAMS_JSON):
  payload = json.loads(pathlib.Path(path).read_text())
  bn = list(payload["body_names"])
  jn = list(payload["joint_names"])
  bp = payload["body_inertias"]
  bodies = {
    n: BodyInertia(
      float(bp[n]["mass"]), _as_vector(bp[n], "ipos", 3), _as_vector(bp[n], "fullinertia", 6)
    )
    for n in bn
  }
  jp = payload["joint_dynamics"]
  joints = {
    n: JointDynamics(
      float(jp[n]["armature"]), float(jp[n]["damping"]), float(jp[n]["frictionloss"])
    )
    for n in jn
  }
  return ModelParameters(bn, jn, bodies, joints)


def apply_model_parameters(spec, mp):
  for b in mp.body_names:
    body = spec.body(b)
    inertia = mp.body_inertias[b]
    body.explicitinertial = True
    body.mass = inertia.mass
    body.ipos = inertia.ipos
    body.inertia[:] = 0.0
    body.iquat[:] = np.nan
    body.fullinertia[:] = inertia.fullinertia
  for j in mp.joint_names:
    joint = spec.joint(j)
    d = mp.joint_dynamics[j]
    joint.armature = d.armature
    joint.damping[0] = d.damping
    joint.frictionloss = d.frictionloss


def make_param_modifier(joint_name, attr):
  if attr == "armature":

    def mod(s, p):
      s.joint(joint_name).armature = float(p.value[0])
  elif attr == "damping":

    def mod(s, p):
      s.joint(joint_name).damping[0] = float(p.value[0])
  elif attr == "frictionloss":

    def mod(s, p):
      s.joint(joint_name).frictionloss = float(p.value[0])
  else:
    raise ValueError(attr)
  return mod


def load_trajectory(idx, max_steps):
  data = np.load(DATA_DIR / f"traj{idx}.npz")
  q = np.asarray(data["q"], dtype=np.float64)
  dq = np.asarray(data["dq"], dtype=np.float64) if "dq" in data else np.zeros_like(q)
  n = min(q.shape[0], max_steps)
  return q[:n], dq[:n]


def build_original(field="none", n_traj=10, n_steps=4000, bounds="reporter"):
  """Build the reporter's benchmark: identify base.xml to match the target model.

  field: which joint-dynamics field starts at 1e-8 ("armature"/"damping"/
    "frictionloss"); others start at 0. "none" => all start at 0.
  bounds: "reporter" (tight inertia bounds) or "wide" (so the truth is reachable,
    isolating non-identifiability from infeasibility).
  """
  field = canonical_dynamics_field(field)
  mp = load_model_parameters()
  base_spec = mujoco.MjSpec.from_file(str(BASE_XML))
  base_model = base_spec.compile()
  target_spec = mujoco.MjSpec.from_file(str(TARGET_XML))
  apply_model_parameters(target_spec, mp)
  target_model = target_spec.compile()
  dt = float(target_model.opt.timestep)

  an = [f"act_{j}" for j in mp.joint_names]
  sn = [f"{j}_pos" for j in mp.joint_names] + [f"{j}_vel" for j in mp.joint_names]
  ist, cts, sts, names = [], [], [], []
  for i in range(1, n_traj + 1):
    q, dq = load_trajectory(i, n_steps)
    t = np.arange(q.shape[0]) * dt
    s0 = sysid.create_initial_state(
      target_model, qpos=q[0], qvel=dq[0], act=np.zeros(target_model.na)
    )
    state, sd = rollout.rollout(target_model, mujoco.MjData(target_model), s0, q[:-1])
    state = np.squeeze(state, 0)
    sd = np.squeeze(sd, 0)
    ist.append(s0)
    names.append(f"traj{i}")
    cts.append(sysid.TimeSeries.from_control_names(t, q, target_model, names=an))
    sts.append(sysid.TimeSeries.from_names(state[:, 0], sd, target_model, names=sn))
  ms = sysid.ModelSequences(
    "arm", base_spec.copy(), names, ist, cts, sts, allow_missing_sensors=True
  )

  if bounds == "wide":
    mass_b = np.array([0.1, 10.0])
    ipos_b = np.array([-0.3, 0.3])
    stretch_b = np.array([0.2, 5.0])
    shear_b = np.array([-2.0, 2.0])
  else:
    mass_b = np.array([MASS_BOUND_MIN, MASS_BOUND_MAX])
    ipos_b = np.array([-IPOS_BOUND, IPOS_BOUND])
    stretch_b = np.array([STRETCH_BOUND_MIN, STRETCH_BOUND_MAX])
    shear_b = np.array([-SHEAR_BOUND, SHEAR_BOUND])

  params = sysid.ParameterDict()
  ps = base_spec.copy()
  pm = ps.compile()
  for b in mp.body_names:
    params.add(
      sysid.body_inertia_param(
        ps,
        pm,
        b,
        inertia_type=sysid.InertiaType.Pseudo,
        mass_bound_mult=mass_b,
        ipos_bound_off=ipos_b,
        stretch_bound_mult=stretch_b,
        shear_bound_off=shear_b,
        param_name=f"{b}_inertia",
      )
    )
  for n in mp.joint_names:
    d = mp.joint_dynamics[n]
    av = VALUE_POS if field == "armature" else VALUE_ZERO
    dv = VALUE_POS if field == "damping" else VALUE_ZERO
    fv = VALUE_POS if field == "frictionloss" else VALUE_ZERO
    params.add(
      sysid.Parameter(
        f"{n}_armature",
        nominal=av,
        min_value=av,
        max_value=max(d.armature * BOUND_MULT, 1e-3),
        modifier=make_param_modifier(n, "armature"),
      )
    )
    params.add(
      sysid.Parameter(
        f"{n}_damping",
        nominal=dv,
        min_value=dv,
        max_value=max(d.damping * BOUND_MULT, 1.0),
        modifier=make_param_modifier(n, "damping"),
      )
    )
    params.add(
      sysid.Parameter(
        f"{n}_frictionloss",
        nominal=fv,
        min_value=fv,
        max_value=max(d.frictionloss * BOUND_MULT, 1.0),
        modifier=make_param_modifier(n, "frictionloss"),
      )
    )

  key = {"mp": mp, "target_model": target_model, "base_model": base_model}
  return params, ms, key, base_model


def residual_vector(params, ms, x):
  rfn = sysid.build_residual_fn(models_sequences=[ms])
  res, _, _ = rfn(x, params)
  return np.concatenate(res)


def jacobian_at(params, ms, x):
  lo, hi = params.get_bounds()
  rfn = sysid.build_residual_fn(models_sequences=[ms])

  def res(z):
    r, _, _ = rfn(z, params)
    return np.concatenate(r)

  r0 = res(x.reshape(-1, 1))
  J, _ = mj_minimize.jacobian_fd(
    residual=res,
    x=x.reshape(-1, 1),
    r=r0,
    eps=EPS,
    n_res=0,
    bounds=[lo.reshape(-1, 1), hi.reshape(-1, 1)],
  )
  return np.asarray(J, np.float64)


def interior_start(params, fields=("damping", "frictionloss"), frac=0.05):
  """Start the given dynamics parameters in the interior, not at the lower bound.

  The optimizer fails (on both the mujoco and scipy backends) when a parameter
  starts at its lower bound (~1e-8). Starting a fraction of the way up from the
  bound fixes it; the result is insensitive to the exact fraction.
  """
  lo, hi = params.get_bounds()
  pn = params.get_non_frozen_parameter_names()
  for i, nm in enumerate(pn):
    if any(nm.endswith("_" + f) for f in fields):
      params[nm].value = np.array([lo[i] + frac * (hi[i] - lo[i])])


def recovered_model(opt_params):
  spec = mujoco.MjSpec.from_file(str(BASE_XML))
  for _, p in opt_params.items():
    p.modifier(spec, p)
  return spec.compile()


def body_id(model, name):
  return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name)
