# MuJoCo Sysid Near-Zero Bound Reproducer

This folder is self-contained except for the Python environment. It expects MuJoCo with `mujoco.sysid` available.

Run:

```bash
python mujoco_sysid_bug.py
```

Docker:

The Dockerfile runs on `linux/amd64`. Set `OPENBLAS_NUM_THREADS=1` at `docker run`
time to compare the default multi-threaded OpenBLAS path against the
single-threaded OpenBLAS path.

```bash
docker build --platform linux/amd64 -t mujoco-sysid-bug .
```

Run the full reproducer test:

```bash
docker run --rm --platform linux/amd64 mujoco-sysid-bug | tee docker_output.txt
```

Run the single-threaded OpenBLAS variant:

```bash
docker run --rm --platform linux/amd64 -e OPENBLAS_NUM_THREADS=1 mujoco-sysid-bug | tee docker_openblas1_output.txt
```

On an ARM machine, such as Apple Silicon, Docker must have amd64 emulation
enabled for the `--platform linux/amd64` build and run commands.

`VALUE_POS` in `mujoco_sysid_bug.py` is the knob. Use `1e-8` for the degraded near-zero-bound case and `--positive-dynamics none` for the all-`VALUE_ZERO` groundtruth case.

Run the all-zero groundtruth case with default OpenBLAS threading:

```bash
docker run --rm --platform linux/amd64 mujoco-sysid-bug \
	python mujoco_sysid_bug.py --positive-dynamics none
```

Run the all-zero groundtruth case with single-threaded OpenBLAS:

```bash
docker run --rm --platform linux/amd64 -e OPENBLAS_NUM_THREADS=1 mujoco-sysid-bug \
	python mujoco_sysid_bug.py --positive-dynamics none
```

Files:

- `base.xml`: base MuJoCo model
- `target.xml`: target MuJoCo model
- `mujoco_model_parameters.json`: MuJoCo-native body and joint parameters loaded by the script
- `traj1.npz` ... `traj10.npz`: local trajectory inputs (5 fast with good conditioning, 5 slow with bad conditioning)
