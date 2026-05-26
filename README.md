# Analysis of mujoco/issues/3284

This repository analyzes the system-identification issue reported in
[google-deepmind/mujoco#3284](https://github.com/google-deepmind/mujoco/issues/3284):
the `mujoco.sysid` optimizer's final residual depends substantially on the
OpenBLAS thread count.

We find three largely independent things:

1. The thread dependence is a consequence of numerical ill-conditioning, not a
   defect in OpenBLAS. The Gauss-Newton Hessian is numerically singular at every
   operating point we tested, because the inertial parameters are not identifiable
   from joint data (point 2), so its factorization is governed by floating-point
   rounding, which the thread count perturbs.
2. The benchmark is ill-posed independently of the optimizer: the true parameters
   lie outside the imposed bounds, and the inertial parameters are not
   identifiable from joint measurements.
3. Starting damping and frictionloss at their `1e-8` lower bound makes the
   optimizer fail on some trajectory realizations on both backends; an interior
   start fixes it and the two backends become indistinguishable.

The data are synthetic, generated from a known parameter set, so we report
parameter recovery against that ground truth. The residual is not a reliable
proxy for recovery: section 2 exhibits a near-zero residual together with
order-of-magnitude parameter errors. Identification below uses
`scipy_parallel_fd` for the parameter scaling that addresses (1); with the
interior initialization in (3), the default mujoco backend converges identically
(section 4).

## Usage

```bash
uv sync
uv run a1_conditioning.py  # optional size args, e.g. a3_recovery.py 10 1200 150
```

| script | result |
| --- | --- |
| `a1_conditioning.py` | the Gauss-Newton Hessian is numerically singular; the rounding-to-step mechanism |
| `a2_bounds.py` | the true inertials lie outside the optimization bounds |
| `a3_recovery.py` | with feasible bounds the residual reaches ~0 while the inertia is wrong by orders of magnitude |
| `a4_regularization.py` | a light prior toward CAD makes the inertials physical (realistic benchmark) |
| `a5_excitation.py` | per-joint sensitivity gain (baseline vs designed); friction/inertia speed tradeoff |
| `a6_excitation_design.py` | per-design worst-determined-parameter table; recovery with a synthetic slow+fast mix |
| `a7_initialization.py` | starting damping/frictionloss at the 1e-8 lower bound makes the solve fail on both backends; an interior start fixes it |

`common.py` builds the reporter's benchmark from the model files; `realistic.py`
builds a benchmark whose truth is CAD perturbed within physical tolerances, used
by a4-a6. The model files (`base.xml`, `target.xml`,
`mujoco_model_parameters.json`) and command trajectories (`traj*.npz`) are the
original reporter data.

## Findings

### 1. The thread sensitivity is numerical ill-conditioning

At the start point the Gauss-Newton Hessian `J^T J`, which the optimizer forms and
factorizes each iteration, is numerically singular:

```
field           cond(J)     cond(J^T J)   cond(J^T J) after per-column scaling
none            2.03e9      4.12e18       6.48e14
armature        1.98e9      3.92e18       4.56e14
damping         2.14e9      4.58e18       6.77e14
frictionloss    2.19e9      4.80e18       6.32e14
```

Double precision resolves condition numbers up to about `1e16`, so
`cond(J^T J) ~ 4e18` means `J^T J` is numerically singular and its
factorization is determined by rounding. Forming `J^T J` under two different
summation orders, which is what changing the BLAS thread count does, changes it
by `9.2e-15`, yet the regularized step changes by 7 percent at the small
regularizer used early in the solve:

```
mu        ||step||   relative step change (raw)   relative step change (scaled)
1e-6      3.0e4      7.2e-2                        5.0e-8
1e-3      8.2e2      5.8e-3                        2.9e-11
1.0       1.5e1      4.7e-6                        3.1e-14
```

Per-column scaling reduces `cond(J^T J)` by about four orders of magnitude and the
step's sensitivity to rounding by about seven, which removes the thread
dependence. The raw ill-conditioning has two sources: the parameters are
optimized in physical units, so the columns of `J` span about nine orders of
magnitude in norm, and the experiment leaves some directions weakly excited. The
singularity itself is inherent: `cond(J^T J)` is huge or numerically infinite at
every operating point we tested (boundary start, interior, truth dynamics),
because the inertial parameters are not identifiable from joint data (section 2).

### 2. The benchmark is ill-posed

**The truth lies outside the bounds.** The reproducer constrains inertia to mass
in `[0.5, 1.5] x CAD` and the center of mass within `0.05 m` of CAD. The true
values violate this: link7's true mass is `4.24x` its CAD value, and 6 of 7 true
centers of mass are outside the bound (offsets up to `0.106 m`). All 21
joint-dynamics truths lie inside their bounds. The inertials are therefore
unreachable as posed.

**The inertials are not identifiable from joint data.** With feasible (wide)
bounds the optimizer fits the data to a residual of `~1.6e-3`, while the
inertials are wrong by orders of magnitude:

```
damping:       all 7 joints recovered to 0%
frictionloss:  all 7 joints recovered to 0%
armature:      joints 3-7 to 0%; joints 1 and 2 wrong by 72% and 16%
inertia trace: wrong by up to ~2600%
```

This is observational equivalence: many inertial parameter sets produce identical
joint trajectories (the serial-chain base-parameter result). The dissipative
terms (damping, friction) are independent of the inertial terms and are recovered
exactly; armature multiplies acceleration as inertia does, so on the proximal
joints it is confounded with the link inertia.

### 3. Identification on a realistic benchmark

The reporter's synthetic truth is far from CAD. `realistic.py` instead perturbs
CAD within physical tolerances, so CAD is a usable prior, as on a real robot.

**Recovery with a CAD prior (a4).** Identifying on the reporter's full
10-trajectory dataset with a light MAP prior (inertia toward CAD, armature
toward its theoretical value, damping and frictionloss free) gives a clean
recovery: data residual `~0.45`, dynamics within `~10%` on most joints (joint5
damping is the outlier at `~57%`), and the inertials sit near the prior as
expected (mass within `~37%`, center of mass within `~0.03 m`). The inertials
remain non-identifiable in absolute terms; the prior provides the outside
information that anchors them. This recovery requires initializing damping
and frictionloss in the interior of their bounds (section 4) on either
backend.

> **Correction.** An earlier version of this section claimed the reporter's
> original trajectories under-excite the distal joints (joint7 RMS =
> `0.0009 rad/s`, "400 times smaller than joint1") and argued that a
> designed slow+fast excitation was needed to recover them. That was an
> artifact of `a4` and `a5` defaulting to a small `N_TRAJ` which loaded
> only the slow half of the dataset (`traj1.npz`-`traj5.npz`). The
> reporter's full set (`traj1.npz`-`traj10.npz`) is a deliberate mix of
> slow and fast trajectories; joint7 baseline RMS on the full set is
> `0.57 rad/s`, comparable to the other joints. The script defaults now
> load `N_TRAJ=10`.

`a5` and `a6` remain as supplementary scripts: `a5` reports per-joint
sensitivity gains between the baseline and a designed multisine, plus a
sensitivity-vs-speed sweep; `a6` compares all-fast / all-slow / mix designs
by worst-determined parameter and re-identifies with the mix.

### 4. Initialization at the parameter boundary breaks the solve

Both the default mujoco backend and `scipy_parallel_fd` fail on some trajectory
realizations when damping and frictionloss are initialized at their `1e-8` lower
bound: the residual plateaus far from the optimum and one or more joints are
mis-identified. Initializing those parameters in the interior of their bounds
removes the failure on both backends.

```
seed     start at 1e-8 bound       interior start
  0     34/21 % (r=2.69)          4/9  % (r=0.39)
  1      6/22 % (r=0.40)          6/21 % (r=0.40)
  2     14/30 % (r=0.60)         15/23 % (r=0.58)
  3      3/13 % (r=0.38)          3/13 % (r=0.38)
  4     82/100% (r=3.36)         14/8  % (r=0.39)
```

(per-seed worst-joint damping/frictionloss error, residual in parentheses)

On any single seed the two backends are indistinguishable once the start is
interior, while both fail at the boundary:

```
backend                 1e-8 bound        interior
mujoco                  34 % / r=2.69     4 % / r=0.39
scipy_parallel_fd       42 % / r=2.39     4 % / r=0.39
```

The mechanism: at the boundary corner of the bound box, damping and frictionloss
are essentially zero, the friction term is absent from the predicted dynamics,
and on some realizations the gradient does not reliably pull those parameters
up. Any non-degenerate interior start escapes the corner; we use 5 percent of
the upper bound, and the result is insensitive to the exact value.

The reporter's setup in #3284 also starts the identified parameter at `1e-8`, so
this boundary-initialization fragility plausibly contributes to the instability
observed there, on top of the conditioning in (1).

The symptom in #3284, a final residual changing dramatically with
`OPENBLAS_NUM_THREADS`, is the compound expression of the singular `J^T J` in
(1) and the boundary initialization here in (3); either fix reduces it
independently, and together they remove it.
