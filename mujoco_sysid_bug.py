#!/usr/bin/env python3
"""Standalone reproducer for the MuJoCo sysid solver-degradation issue.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_XML = SCRIPT_DIR / "base.xml"
TARGET_XML = SCRIPT_DIR / "target.xml"
PARAMS_JSON = SCRIPT_DIR / "mujoco_model_parameters.json"

NUM_TRAJECTORIES = 10
MAX_STEPS = 4000
MAX_ITERS = 50
BOUND_MULT = 3.0
MASS_BOUND_MIN = 0.5
MASS_BOUND_MAX = 1.5
IPOS_BOUND = 0.05
STRETCH_BOUND_MIN = 0.5
STRETCH_BOUND_MAX = 1.5
SHEAR_BOUND = 0.25

# Bug knob: VALUE_POS or VALUE_ZERO
VALUE_POS = 1e-8
VALUE_ZERO = 0.0
DYNAMICS_FIELDS = ("armature", "damping", "frictionloss")
DYNAMICS_ALIASES = {"friction": "frictionloss", "zero": "none"}


@dataclass(frozen=True)
class BodyInertia:
    mass: float
    ipos: np.ndarray
    fullinertia: np.ndarray


@dataclass(frozen=True)
class JointDynamics:
    armature: float
    damping: float
    frictionloss: float


@dataclass(frozen=True)
class MuJoCoModelParameters:
    body_names: list[str]
    joint_names: list[str]
    body_inertias: dict[str, BodyInertia]
    joint_dynamics: dict[str, JointDynamics]


def _as_vector(payload: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != (size,):
        raise ValueError(f"Expected {key} to have shape ({size},), got {value.shape}")
    return value


def load_mujoco_model_parameters(path: Path) -> MuJoCoModelParameters:
    payload = json.loads(path.expanduser().resolve().read_text())
    body_names = list(payload["body_names"])
    joint_names = list(payload["joint_names"])

    body_payload = payload["body_inertias"]
    body_inertias = {
        name: BodyInertia(
            mass=float(body_payload[name]["mass"]),
            ipos=_as_vector(body_payload[name], "ipos", 3),
            fullinertia=_as_vector(body_payload[name], "fullinertia", 6),
        )
        for name in body_names
    }

    joint_payload = payload["joint_dynamics"]
    joint_dynamics = {
        name: JointDynamics(
            armature=float(joint_payload[name]["armature"]),
            damping=float(joint_payload[name]["damping"]),
            frictionloss=float(joint_payload[name]["frictionloss"]),
        )
        for name in joint_names
    }

    return MuJoCoModelParameters(body_names, joint_names, body_inertias, joint_dynamics)


def apply_mujoco_model_parameters(spec: mujoco.MjSpec, params: MuJoCoModelParameters) -> None:
    for body_name in params.body_names:
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"MuJoCo body not found: {body_name}")
        inertia = params.body_inertias[body_name]
        body.explicitinertial = True
        body.mass = inertia.mass
        body.ipos = inertia.ipos
        body.inertia[:] = 0.0
        body.iquat[:] = np.nan
        body.fullinertia[:] = inertia.fullinertia

    for joint_name in params.joint_names:
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        dynamics = params.joint_dynamics[joint_name]
        joint.armature = dynamics.armature
        joint.damping[0] = dynamics.damping
        joint.frictionloss = dynamics.frictionloss


def load_trajectory(path: Path, max_steps: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    q = np.asarray(data["q"], dtype=np.float64)
    dq = np.asarray(data["dq"], dtype=np.float64) if "dq" in data else np.zeros_like(q)
    n = min(q.shape[0], max_steps)
    return q[:n], dq[:n]


def make_param_modifier(joint_name: str, attr: str):
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MuJoCo sysid near-zero-bound reproducer."
    )
    parser.add_argument(
        "--positive-dynamics",
        choices=("none", "zero", "armature", "damping", "friction", "frictionloss"),
        default="armature",
        help=(
            "Joint dynamics parameter whose nominal and min_value are set to "
            "VALUE_POS. The other joint dynamics parameters use VALUE_ZERO. "
            "Use 'none' to set all joint dynamics parameters to VALUE_ZERO."
        ),
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=MAX_ITERS,
        help="Maximum optimizer iterations.",
    )
    return parser.parse_args(argv)


def canonical_dynamics_field(field: str) -> str:
    return DYNAMICS_ALIASES.get(field, field)


def compute_inverse_dynamics_torque(
    model: mujoco.MjModel,
    joint_names: list[str],
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    qacc_traj: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    dof_indices = np.asarray(
        [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT.value, jn)])
         for jn in joint_names],
        dtype=np.int64,
    )
    out = np.empty((qpos_traj.shape[0], len(dof_indices)), dtype=np.float64)
    for t in range(qpos_traj.shape[0]):
        data.qpos[:] = qpos_traj[t]
        data.qvel[:] = qvel_traj[t]
        data.qacc[:] = qacc_traj[t]
        mujoco.mj_inverse(model, data)
        out[t] = data.qfrc_inverse[dof_indices]
    return out


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    positive_dynamics = canonical_dynamics_field(args.positive_dynamics)
    if positive_dynamics != "none" and positive_dynamics not in DYNAMICS_FIELDS:
        raise ValueError(f"Unexpected joint dynamics field: {args.positive_dynamics}")

    model_params = load_mujoco_model_parameters(PARAMS_JSON)
    base_spec = mujoco.MjSpec.from_file(str(BASE_XML))
    target_spec = mujoco.MjSpec.from_file(str(TARGET_XML))
    base_spec.compile()
    apply_mujoco_model_parameters(target_spec, model_params)
    target_model = target_spec.compile()
    dt = float(target_model.opt.timestep)
    print(f"dt={dt}, nq={target_model.nq}, nu={target_model.nu}")
    print(
        "joint dynamics nominal/min_value: "
        + ", ".join(
            f"{field}={'VALUE_POS' if field == positive_dynamics else 'VALUE_ZERO'}"
            for field in DYNAMICS_FIELDS
        )
    )

    npz_paths = [SCRIPT_DIR / f"traj{i}.npz" for i in range(1, NUM_TRAJECTORIES + 1)]
    missing = [str(path) for path in npz_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing trajectory files: {missing}")
    print(f"using {len(npz_paths)} trajectories")

    actuator_names = [f"act_{jn}" for jn in model_params.joint_names]
    sensor_names_all = [f"{jn}_pos" for jn in model_params.joint_names] + [
        f"{jn}_vel" for jn in model_params.joint_names
    ]

    initial_states: list[np.ndarray] = []
    control_ts_list: list[sysid.TimeSeries] = []
    sensor_ts_list: list[sysid.TimeSeries] = []
    sequence_names: list[str] = []
    rolled_states: list[np.ndarray] = []

    for npz_path in npz_paths:
        q, dq = load_trajectory(npz_path, MAX_STEPS)
        n = q.shape[0]
        t = np.arange(n) * dt

        initial_state = sysid.create_initial_state(
            target_model, qpos=q[0], qvel=dq[0], act=np.zeros(target_model.na),
        )
        target_data = mujoco.MjData(target_model)
        state, sensordata = rollout.rollout(target_model, target_data, initial_state, q[:-1])
        state = np.squeeze(state, axis=0)
        sensordata = np.squeeze(sensordata, axis=0)

        control_ts = sysid.TimeSeries.from_control_names(t, q, target_model, names=actuator_names)
        sensor_ts = sysid.TimeSeries.from_names(
            state[:, 0], sensordata, target_model, names=sensor_names_all,
        )

        initial_states.append(initial_state)
        control_ts_list.append(control_ts)
        sensor_ts_list.append(sensor_ts)
        sequence_names.append(npz_path.stem)
        rolled_states.append(state)
        print(f"  rolled out {npz_path.name}: {n} steps")

    ms = sysid.ModelSequences(
        "arm7dof",
        base_spec.copy(),
        sequence_names,
        initial_states,
        control_ts_list,
        sensor_ts_list,
        allow_missing_sensors=True,
    )

    params = sysid.ParameterDict()

    param_spec = base_spec.copy()
    param_model = param_spec.compile()
    mass_bounds = np.array([MASS_BOUND_MIN, MASS_BOUND_MAX], dtype=np.float64)
    ipos_bounds = np.array([-IPOS_BOUND, IPOS_BOUND], dtype=np.float64)
    stretch_bounds = np.array([STRETCH_BOUND_MIN, STRETCH_BOUND_MAX], dtype=np.float64)
    shear_bounds = np.array([-SHEAR_BOUND, SHEAR_BOUND], dtype=np.float64)
    for body_name in model_params.body_names:
        params.add(
            sysid.body_inertia_param(
                param_spec, param_model, body_name,
                inertia_type=sysid.InertiaType.Pseudo,
                mass_bound_mult=mass_bounds,
                ipos_bound_off=ipos_bounds,
                stretch_bound_mult=stretch_bounds,
                shear_bound_off=shear_bounds,
                param_name=f"{body_name}_inertia",
            )
        )

    for name in model_params.joint_names:
        dynamics = model_params.joint_dynamics[name]
        armature_value = VALUE_POS if positive_dynamics == "armature" else VALUE_ZERO
        damping_value = VALUE_POS if positive_dynamics == "damping" else VALUE_ZERO
        frictionloss_value = VALUE_POS if positive_dynamics == "frictionloss" else VALUE_ZERO
        params.add(sysid.Parameter(
            f"{name}_armature", nominal=armature_value,
            min_value=armature_value, max_value=max(dynamics.armature * BOUND_MULT, 1e-3),
            modifier=make_param_modifier(name, "armature"),
        ))
        params.add(sysid.Parameter(
            f"{name}_damping", nominal=damping_value,
            min_value=damping_value, max_value=max(dynamics.damping * BOUND_MULT, 1.0),
            modifier=make_param_modifier(name, "damping"),
        ))
        params.add(sysid.Parameter(
            f"{name}_frictionloss", nominal=frictionloss_value,
            min_value=frictionloss_value, max_value=max(dynamics.frictionloss * BOUND_MULT, 1.0),
            modifier=make_param_modifier(name, "frictionloss"),
        ))

    print(f"optimizing {params.size} parameters with max_iters={args.max_iters}")
    residual_fn = sysid.build_residual_fn(models_sequences=[ms])
    opt_params, _ = sysid.optimize(
        initial_params=params, residual_fn=residual_fn, optimizer="mujoco",
        verbose=True, max_iters=args.max_iters,
    )

    recovered_spec = base_spec.copy()
    for _, p in opt_params.items():
        p.modifier(recovered_spec, p)
    recovered_model = recovered_spec.compile()

    n_joints = len(model_params.joint_names)
    total_sq = np.zeros(n_joints, dtype=np.float64)
    tau_min = np.full(n_joints, np.inf, dtype=np.float64)
    tau_max = np.full(n_joints, -np.inf, dtype=np.float64)
    total_steps = 0
    for state in rolled_states:
        nstep = state.shape[0]
        qpos_full = state[:, 1 : 1 + target_model.nq]
        qvel_full = state[:, 1 + target_model.nq : 1 + target_model.nq + target_model.nv]
        qacc_full = np.zeros_like(qvel_full)
        qacc_full[1:] = (qvel_full[1:] - qvel_full[:-1]) / dt
        qacc_full[0] = qacc_full[1]

        tau_target = compute_inverse_dynamics_torque(
            target_model, model_params.joint_names, qpos_full, qvel_full, qacc_full
        )
        tau_recovered = compute_inverse_dynamics_torque(
            recovered_model, model_params.joint_names, qpos_full, qvel_full, qacc_full
        )
        err = tau_target - tau_recovered
        total_sq += np.sum(err ** 2, axis=0)
        tau_min = np.minimum(tau_min, tau_target.min(axis=0))
        tau_max = np.maximum(tau_max, tau_target.max(axis=0))
        total_steps += nstep

    tau_range = tau_max - tau_min
    rmse_per_joint = np.sqrt(total_sq / max(total_steps, 1))
    nrmse_per_joint = np.where(tau_range > 1e-12, rmse_per_joint / tau_range * 100.0, np.nan)
    grand_rmse = float(np.sqrt(np.sum(total_sq) / max(total_steps * n_joints, 1)))
    grand_range = float(np.max(tau_max) - np.min(tau_min))
    grand_nrmse = (grand_rmse / grand_range * 100.0) if grand_range > 1e-12 else float("nan")

    print()
    print("Per-joint torque error over all rolled-out steps (NRMSE normalized by tau_max - tau_min):")
    print(f"{'joint':<14}{'rmse [N*m]':>14}{'tau_range [N*m]':>18}{'NRMSE [%]':>12}")
    for i, jn in enumerate(model_params.joint_names):
        print(f"{jn:<14}{rmse_per_joint[i]:>14.6f}{tau_range[i]:>18.6f}{nrmse_per_joint[i]:>12.4f}")
    print(f"{'TOTAL':<14}{grand_rmse:>14.6f}{grand_range:>18.6f}{grand_nrmse:>12.4f}")


if __name__ == "__main__":
    main()
