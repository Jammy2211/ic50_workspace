# Activate the Python environment that has PyAutoFit installed.
#
# Resolving relative to this file (not the current directory) means it works both for a
# local `source activate.sh` and for the HPC scripts' `source $PROJECT_PATH/activate.sh`.
#
# Order of preference:
#   1. a project-local .venv
#   2. $PYAUTO_HPC_BASE — the directory holding a `PyAuto/` virtualenv alongside
#      editable PyAuto* source checkouts (on RAL: /mnt/ral/jnightin/PyAuto)
#   3. the RAL default, so an HPC job works even if PYAUTO_HPC_BASE is unset
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV=$HERE/.venv

if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
else
    BASE="${PYAUTO_HPC_BASE:-/mnt/ral/jnightin/PyAuto}"
    if [ -f "$BASE/PyAuto/bin/activate" ]; then
        source "$BASE/PyAuto/bin/activate"
        # PyAutoConf is deliberately NOT listed: the checkout at $BASE/PyAutoConf is
        # stale, and putting it on PYTHONPATH shadows the current pip-installed
        # package in the venv. PyAutoNerves must be listed — current PyAutoFit
        # depends on it. This mirrors slope_hierarchy_scale/activate.sh, which is
        # the configuration proven to work on RAL.
        export PYTHONPATH=$BASE:\
$BASE/PyAutoNerves:\
$BASE/PyAutoFit:\
$BASE/PyAutoArray:\
$BASE/PyAutoGalaxy:\
$BASE/PyAutoLens
        # Keep caches OFF $HOME on HPC (as in the assistant template): RAL's /home sits on each
        # node's small root disk, and tools that default to ~/.cache — the PyAutoNerves JAX compile
        # cache, pip, matplotlib, numba, CUDA/Triton kernels — fill it and break the node (RAL
        # admin, 2026-09-25). Only unset variables are filled, so a submit script's own setting
        # wins (and an explicitly EMPTY JAX_COMPILATION_CACHE_DIR still disables the JAX cache).
        export PYAUTO_HPC_CACHE="${PYAUTO_HPC_CACHE:-$(dirname "$BASE")/.cache}"
        mkdir -p "$PYAUTO_HPC_CACHE" 2>/dev/null || true
        export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PYAUTO_HPC_CACHE}"
        export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PYAUTO_HPC_CACHE/pip}"
        export MPLCONFIGDIR="${MPLCONFIGDIR:-$PYAUTO_HPC_CACHE/matplotlib}"
        export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PYAUTO_HPC_CACHE/numba}"
        export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$PYAUTO_HPC_CACHE/nv}"
        export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PYAUTO_HPC_CACHE/triton}"
        export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR-$PYAUTO_HPC_CACHE/pyauto_jax}"
        export ASTROPY_CACHE_DIR="${ASTROPY_CACHE_DIR:-$PYAUTO_HPC_CACHE/astropy}"
        # RAL compute-node hygiene (2026-11-13, as in the assistant template): temp files off
        # the node's small local /tmp (RAL admin, 2026-11-12), and JAX pinned to the CPU when the
        # job holds no GPU (its bundled CUDA plugin otherwise logs a cuInit ERROR + traceback).
        if [ -z "${TMPDIR:-}" ] || [ "$TMPDIR" = /tmp ]; then
            export TMPDIR="$(dirname "$BASE")/tmp"
        fi
        mkdir -p "$TMPDIR" 2>/dev/null || true
        if [ -z "${JAX_PLATFORMS:-}" ] && [ -z "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}${CUDA_VISIBLE_DEVICES:-}" ]; then
            export JAX_PLATFORMS=cpu
            echo "activate.sh: no GPU in this job, JAX_PLATFORMS=cpu" >&2
        fi
    else
        echo "No local .venv found and no PyAuto venv at $BASE (set PYAUTO_HPC_BASE)." >&2
    fi
fi
