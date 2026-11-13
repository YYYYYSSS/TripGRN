#!/usr/bin/env python3
from pathlib import Path

root = Path("/mnt/f")
repo = None
for p in root.iterdir():
    if not p.is_dir():
        continue
    if "TripGRN" in p.name:
        repo = p
        break
    try:
        for c in p.iterdir():
            if c.is_dir() and "TripGRN" in c.name:
                repo = c
                break
    except PermissionError:
        continue
    if repo is not None:
        break

print("repo", repo)
if repo is None:
    raise SystemExit("TripGRN repo not found under /mnt/f")

link = Path("/home/yhr/tripgrn")
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(repo)
print("symlink", link, "->", repo)

act = Path("/home/yhr/activate_deepfgrn_wsl.sh")
act.write_text(
    """#!/usr/bin/env bash
export HOME=/home/yhr
export USER=yhr
source /home/yhr/miniconda3/etc/profile.d/conda.sh
conda activate deepfgrn_wsl
SP="$CONDA_PREFIX/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="$(echo "$SP"/nvidia/*/lib | tr ' ' ':'):/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
export PYTHONPATH="/home/yhr/tripgrn:${PYTHONPATH:-}"
cd /home/yhr/tripgrn || true
""",
    encoding="utf-8",
)

probe = Path("/home/yhr/probe_tf_gpu.sh")
probe.write_text(
    """#!/usr/bin/env bash
set -euo pipefail
source /home/yhr/activate_deepfgrn_wsl.sh
python -c 'import tensorflow as tf; g=tf.config.list_physical_devices("GPU"); print("tf", tf.__version__); print("gpus", g); assert g, "NO_GPU"; a=tf.random.normal([512,512]); b=tf.matmul(a,a); _=b.numpy(); print("matmul_ok")'
""",
    encoding="utf-8",
)

smoke = Path("/home/yhr/smoke_deepfgrn.sh")
smoke.write_text(
    """#!/usr/bin/env bash
set -euo pipefail
source /home/yhr/activate_deepfgrn_wsl.sh
python -u fig01_pancancer_benchmark/scripts/run_fig01_deepfgrn_kegg.py smoke
python -c 'import tensorflow as tf; print("gpus", tf.config.list_physical_devices("GPU"))'
""",
    encoding="utf-8",
)

for p in (act, probe, smoke):
    p.chmod(0o755)
    print("wrote", p)
print("done")
