# Analysis of mujoco/issues/3284

This repository analyzes the system-identification issue reported in
[google-deepmind/mujoco#3284](https://github.com/google-deepmind/mujoco/issues/3284):
the `mujoco.sysid` optimizer's final residual depends substantially on the
OpenBLAS thread count.

We find four largely independent things:

1. The thread dependence is a consequence of numerical ill-conditioning, not a
   defect in OpenBLAS. The Gauss-Newton Hessian is numerically singular at every
   operating point we tested, because the inertial parameters are not identifiable
   from joint data (point 2), so its factorization is governed by floating-point
   rounding, which the thread count perturbs.
2. The benchmark is ill-posed independently of the optimizer: the true parameters
   lie outside the imposed bounds, and the inertial parameters are not
   identifiable from joint measurements.
3. The reporter's original trajectories under-excite the distal joints (joint7's
   root-mean-square velocity is `~0.0009 rad/s`), so they cannot be identified
   from that data; a designed slow+fast excitation recovers them.
4. Starting damping and frictionloss at their `1e-8` lower bound makes the
   optimizer fail on some trajectory realizations on both backends; an interior
   start fixes it and the two backends become indistinguishable.

The data are synthetic, generated from a known parameter set, so we report
parameter recovery against that ground truth. The residual is not a reliable
proxy for recovery: section 2 exhibits a near-zero residual together with
order-of-magnitude parameter errors. Identification below uses
`scipy_parallel_fd` for the parameter scaling that addresses (1); with the
interior initialization in (4), the default mujoco backend converges identically
(section 4).

## Usage

```bash
uv sync
uv run a1_conditioning.py  # optional size args, e.g. a3_recovery.py 5 1200 150
```

| script | result |
| --- | --- |
| `a1_conditioning.py` | the Gauss-Newton Hessian is numerically singular; the rounding-to-step mechanism |
| `a2_bounds.py` | the true inertials lie outside the optimization bounds |
| `a3_recovery.py` | with feasible bounds the residual reaches ~0 while the inertia is wrong by orders of magnitude |
| `a4_regularization.py` | a light prior toward CAD makes the inertials physical (realistic benchmark) |
| `a5_excitation.py` | the distal joints are under-excited; the friction/inertia speed tradeoff |
| `a6_excitation_design.py` | a combined slow+fast excitation recovers every parameter |
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
none            6.73e9      4.53e19       4.23e15
armature        6.73e9      4.53e19       3.98e15
damping         6.24e9      3.89e19       4.07e15
frictionloss    6.83e9      4.67e19       4.03e15
```

Double precision resolves condition numbers up to about `1e16`, so
`cond(J^T J) ~ 4.5e19` means `J^T J` is numerically singular and its
factorization is determined by rounding. Forming `J^T J` under two different
summation orders, which is what changing the BLAS thread count does, changes it
by `1.9e-15`, yet the regularized step changes by 12 percent at the small
regularizer used early in the solve:

```
mu        ||step||   relative step change (raw)   relative step change (scaled)
1e-6      3.2e3      1.2e-1                        1.9e-8
1e-3      1.9e1      2.0e-4                        1.7e-11
1.0       5.0e0      2.8e-7                        1.9e-14
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
bounds the optimizer fits the data to a residual of `~3e-11`, while the inertials
are wrong by orders of magnitude:

```
damping:       all 7 joints recovered to 0%
frictionloss:  all 7 joints recovered to 0%
armature:      joints 3-7 to 0%; joints 1 and 2 wrong by 83% and 29%
inertia trace: wrong by up to ~2300%
```

This is observational equivalence: many inertial parameter sets produce identical
joint trajectories (the serial-chain base-parameter result). The dissipative
terms (damping, friction) are independent of the inertial terms and are recovered
exactly; armature multiplies acceleration as inertia does, so on the proximal
joints it is confounded with the link inertia.

### 3. Identification on a realistic benchmark

The reporter's synthetic truth is far from CAD. `realistic.py` instead perturbs
CAD within physical tolerances, so CAD is a usable prior, as on a real robot. The
section builds up in three steps: a prior is necessary but not sufficient (a4),
the original trajectories are diagnosed as the limiting factor (a5), and a better
excitation closes the gap (a6).

**A prior is not enough on the original trajectories (a4).** Identifying on the
reporter's trajectories with a light MAP prior, inertia toward CAD, armature
toward its theoretical value, damping and frictionloss free, makes the inertials
physical instead of unconstrained and removes the armature/inertia confound, while
the data drives the well-excited proximal dynamics to the truth. But the distal
joints stay wrong (joint7 damping off by about 290 percent and frictionloss by
orders of magnitude), because the original trajectories barely move them. The
prior cannot manufacture information the data does not contain.

**Diagnosing the excitation (a5).** The original trajectories leave the distal
joints under-excited; joint7's root-mean-square velocity is `~0.0009 rad/s`, about
400 times smaller than joint1's. Different parameters also require different
motion: frictionloss is observable only where a joint passes through velocity
reversals at low speed, whereas damping and armature require fast motion. A speed
sweep confirms this, friction sensitivity falls and damping sensitivity rises with
speed. A single-speed experiment cannot make every parameter observable.

**Choosing a better excitation (a6).** This is design by selection among
candidate trajectories, not numerical optimal input design: we do not optimize a
trajectory parameterization against an information criterion (future work). We
instead construct
a few candidate designs (all-fast multisines, all-slow sinusoids, and a slow+fast
mix) and compare them by the parameter each one determines worst. The table
reports, per design, the smallest sensitivity among the friction, damping, and
armature parameters (higher is better; the best design is the one whose worst
entry is largest). An all-fast design starves friction, an all-slow design starves
damping and armature, and only the combination keeps all three observable:

```
design     min friction   min damping   min armature
all-fast   0.13           0.63          0.46      (friction starved)
all-slow   0.18           0.19          0.13      (damping and armature starved)
mix        0.16           0.61          0.45      (worst entry largest)
```

(The usual D-optimality criterion, which maximizes total information volume, is
misleading here: it is dominated by the well-excited directions and actually
prefers the all-fast design that leaves friction unidentified.)

Re-identifying with the selected slow+fast mix and the light prior, recovery
against the truth is consistent across five trajectory seeds (a7): worst-joint
damping `4-15 %` (mean 8), armature within `~10 %`, frictionloss `8-23 %`
(mean ~15), and the inertials physical (mass within `~30 %`, center of mass within
about `0.05 m`). The single-seed numbers in earlier drafts (8 / 11 / 19 percent)
were one favorable realization; the multi-seed figures above are the honest
picture. This recovery requires initializing damping and frictionloss in the
interior of their bounds (section 4) on either backend.

The dynamics, which the excitation targets, are recovered to within ~20 percent
on the worst joint and a few percent on most. The larger inertial errors are
expected rather than a failure of the estimator: those directions are weakly
identifiable and rely on the prior, which here is centered on CAD about one to
two standard deviations from the (synthetic) truth, so the gap measures that
prior-to-truth distance. On hardware the truth is unknown and CAD is the best
available estimate, so holding the unidentifiable directions at CAD is the
intended behavior.

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
(1) and the boundary initialization here in (4); either fix reduces it
independently, and together they remove it.
