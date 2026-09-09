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
    else
        echo "No local .venv found and no PyAuto venv at $BASE (set PYAUTO_HPC_BASE)." >&2
    fi
fi
