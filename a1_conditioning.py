#!/usr/bin/env python3
"""A1. J^T J is numerically singular; the roundoff -> step mechanism. See README.

Usage: a1_conditioning.py [N_TRAJ N_STEPS]
"""

from __future__ import annotations

import sys

import numpy as np

import common as C

N_TRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 800

print(f"=== A1. Conditioning at the start point ({N_TRAJ} traj x {N_STEPS} steps) ===\n")
hdr = f"{'field':<14}{'cond(J)':>12}{'cond(J^T J)':>14}{'cond(J^T J) scaled':>20}"
print(hdr)
print("-" * len(hdr))

mech = None
for field in ("none", "armature", "damping", "frictionloss"):
  params, ms, key, base_model = C.build_original(field, N_TRAJ, N_STEPS)
  x0 = params.as_vector()
  J = C.jacobian_at(params, ms, x0)
  s = np.linalg.svd(J, compute_uv=False)
  condJ = s[0] / s[-1]
  cn = np.linalg.norm(J, axis=0)
  D = np.where(cn > 0, 1.0 / cn, 0.0)
  ss = np.linalg.svd(J * D[None, :], compute_uv=False)
  print(f"{field:<14}{condJ:>12.2e}{condJ**2:>14.2e}{(ss[0] / ss[-1]) ** 2:>20.2e}")
  if field == "damping":
    mech = (J, C.residual_vector(params, ms, x0))

print("\nfloat64 can represent condition numbers up to ~1e16; cond(J^T J) ~ 1e19 means")
print("J^T J is numerically singular, so its factorization is decided by roundoff.\n")

print("=== Mechanism: roundoff in J^T J -> optimizer step (damping case) ===\n")
J, r0 = mech
n = J.shape[1]
grad = J.T @ r0
blocks = np.array_split(np.arange(J.shape[0]), 7)
hess_a = J.T @ J  # one summation order
hess_b = sum(J[b].T @ J[b] for b in blocks[::-1])  # different order (reversed blocks)
rel = np.linalg.norm(hess_a - hess_b) / np.linalg.norm(hess_a)
cn = np.linalg.norm(J, axis=0)
Dm = np.where(cn > 0, 1.0 / cn, 0.0)
Js = J * Dm[None, :]
hsa = Js.T @ Js
hsb = sum(Js[b].T @ Js[b] for b in blocks[::-1])
gs = Js.T @ r0
print(f"relative difference between the two summation orders of J^T J: {rel:.2e}  (roundoff)")
print(f"\n{'mu':>8}{'||step||':>12}{'rel step diff RAW':>20}{'rel step diff SCALED':>22}")
for mu in (1e-6, 1e-3, 1.0):
  da = np.linalg.solve(hess_a + mu * np.eye(n), -grad)
  db = np.linalg.solve(hess_b + mu * np.eye(n), -grad)
  sa = np.linalg.solve(hsa + mu * np.eye(n), -gs)
  sb = np.linalg.solve(hsb + mu * np.eye(n), -gs)
  rr = np.linalg.norm(da - db) / max(np.linalg.norm(da), 1e-300)
  rs = np.linalg.norm(sa - sb) / max(np.linalg.norm(sa), 1e-300)
  print(f"{mu:>8g}{np.linalg.norm(da):>12.2e}{rr:>20.2e}{rs:>22.2e}")
