# syntax=docker/dockerfile:1

# The host reproducer was captured on Linux x86_64 / amd64. Keep the image on
# the same architecture so MuJoCo, NumPy, and SciPy use the same wheel family.
ARG REPRO_PLATFORM=linux/amd64
FROM --platform=${REPRO_PLATFORM} ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV VIRTUAL_ENV=/opt/mujoco-sysid-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    PYTHONUNBUFFERED=1

RUN test "$(dpkg --print-architecture)" = "amd64" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libegl1 \
        libgl1 \
        libglfw3 \
        libglvnd0 \
        libosmesa6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxxf86vm1 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip==26.1.1 \
    && python -m pip install --no-cache-dir \
        absl-py==2.4.0 \
        colorama==0.4.6 \
        contourpy==1.3.3 \
        cycler==0.12.1 \
        etils==1.14.0 \
        fonttools==4.62.1 \
        fsspec==2026.4.0 \
        glfw==2.10.0 \
        Jinja2==3.1.6 \
        kiwisolver==1.5.0 \
        MarkupSafe==3.0.3 \
        matplotlib==3.10.9 \
        mujoco==3.8.1 \
        narwhals==2.21.0 \
        numpy==2.4.6 \
        packaging==26.2 \
        pillow==12.2.0 \
        plotly==6.7.0 \
        PyOpenGL==3.1.10 \
        pyparsing==3.3.2 \
        python-dateutil==2.9.0.post0 \
        PyYAML==6.0.3 \
        scipy==1.17.1 \
        six==1.17.0 \
        tabulate==0.10.0 \
        typing_extensions==4.15.0 \
        zipp==3.23.1 \
    && python -c "import jinja2, matplotlib, mujoco, numpy, plotly, scipy, yaml; from mujoco import sysid; assert mujoco.__version__ == '3.8.1'; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"

WORKDIR /app
COPY . /app/

CMD ["python", "mujoco_sysid_bug.py"]