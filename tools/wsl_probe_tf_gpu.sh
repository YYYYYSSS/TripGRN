#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/tools/wsl_activate_deepfgrn.sh"
python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("tf", tf.__version__)
print("built_with_cuda", tf.test.is_built_with_cuda())
print("gpus", gpus)
assert gpus, "NO_GPU"
with tf.device("/GPU:0"):
    a = tf.random.normal([512, 512])
    b = tf.matmul(a, a)
    _ = b.numpy()
print("matmul_ok")
PY
