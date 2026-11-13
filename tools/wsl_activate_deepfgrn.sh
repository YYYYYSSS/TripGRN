#!/usr/bin/env bash
# Activate DeepFGRN WSL env with TF GPU library paths.
# Preferred entry on this machine: source /home/yhr/activate_deepfgrn_wsl.sh
export HOME=/home/yhr
export USER=yhr
# shellcheck disable=SC1091
source /home/yhr/miniconda3/etc/profile.d/conda.sh
conda activate deepfgrn_wsl
SP="$CONDA_PREFIX/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$(echo "$SP"/nvidia/*/lib | tr ' ' ':'):/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
if [[ -d /home/yhr/tripgrn ]]; then
  export PYTHONPATH="/home/yhr/tripgrn:${PYTHONPATH:-}"
  cd /home/yhr/tripgrn || true
fi
