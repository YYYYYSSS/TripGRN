#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.1 launchers: DeepFGRN KEGG epithelial 10CV (+ path smoke check).

Protocol matches TripGRN evaluation setting:
  - shared ``cv_splits_10fold.json`` (5:2:3 unseen cells)
  - held-out gene pairs (~64/16/20 stratified)
  - dGGAN gold graph = train positives only (architecture unchanged)
DeepFGRN additionally subsample_to_common_length for equal-length CorrResNet inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_deepfgrn_root(root: Path) -> Path | None:
    env = os.environ.get("DEEPFGRN_ROOT", "").strip()
    if env and Path(env).is_dir():
        return Path(env).resolve()
    candidates = [root.parent / "project_DeepFGRN"]
    legacy_txt = root / "LEGACY_ROOT.txt"
    if legacy_txt.is_file():
        try:
            legacy = Path(legacy_txt.read_text(encoding="utf-8").strip())
            candidates.append(legacy.parent / "project_DeepFGRN")
        except Exception:
            pass
    candidates.extend(
        [
            Path(r"E:\project\project_DeepFGRN"),
            Path("/mnt/e/project/project_DeepFGRN"),
        ]
    )
    for path in candidates:
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            continue
    return None


def cmd_smoke() -> int:
    root = _repo_root()
    sys.path.insert(0, str(root))
    from grn.data.pancancer_paths import (
        EXCLUDED_GSE_IDS,
        PANCANCER_GSE_IDS,
        cv_splits_path,
        gse_npy_triplet,
        scRNA_layout,
        step01_grn_csv,
    )

    layout = scRNA_layout(root)
    deep_root = _default_deepfgrn_root(root)
    checks = {
        "grn_csv": str(step01_grn_csv(root)),
        "epithelial": str(layout["epithelial"]),
        "experiments_deepfgrn": str(layout["experiments_deepfgrn"]),
        "deepfgrn_root": str(deep_root) if deep_root else None,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"PANCANCER_GSE_IDS ({len(PANCANCER_GSE_IDS)}): {PANCANCER_GSE_IDS}")
    print(f"EXCLUDED_GSE_IDS: {EXCLUDED_GSE_IDS}")
    assert "GSE145825" not in PANCANCER_GSE_IDS
    if not Path(checks["grn_csv"]).is_file():
        raise FileNotFoundError(f"Missing GRN: {checks['grn_csv']}")
    if deep_root is None:
        raise FileNotFoundError(
            "project_DeepFGRN not found. Set DEEPFGRN_ROOT or place at E:\\project\\project_DeepFGRN"
        )
    missing = []
    for gse in PANCANCER_GSE_IDS:
        v, g, c = gse_npy_triplet("epithelial", gse, root)
        cv = cv_splits_path(gse, root)
        for p in (v, g, c, cv):
            if not p.is_file():
                missing.append(str(p))
        # Same CV file TripGRN uses
        payload = json.loads(cv.read_text(encoding="utf-8"))
        if list(payload.get("sample_split", [])) != [5, 2, 3]:
            raise ValueError(f"{gse} sample_split != [5,2,3]: {payload.get('sample_split')}")
        if int(payload.get("n_folds", 0)) != 10:
            raise ValueError(f"{gse} n_folds != 10")
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print("[fig01-deepfgrn-smoke] OK: paths + shared 9 GSE cv_splits (5:2:3) + DeepFGRN root")
    return 0


def cmd_run(gse: str, fold: str, force: bool, keras_batch_size_cap: int, deepfgrn_root: str | None) -> int:
    root = _repo_root()
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.run_deepfgrn_pancancer_10cv",
        "--gse",
        gse,
        "--fold",
        fold,
        "--keras-batch-size-cap",
        str(keras_batch_size_cap),
    ]
    if force:
        cmd.append("--force")
    resolved = deepfgrn_root or (str(_default_deepfgrn_root(root)) if _default_deepfgrn_root(root) else None)
    if resolved:
        cmd.extend(["--deepfgrn-root", resolved])
    print("[fig01-deepfgrn] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def cmd_summarize(gse: str) -> int:
    root = _repo_root()
    cmd = [sys.executable, "-u", "-m", "grn.scripts.summarize_deepfgrn_pancancer_10cv", "--gse", gse]
    print("[fig01-deepfgrn] RUN", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.1 DeepFGRN KEGG helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Validate paths / shared CV splits / DeepFGRN root")

    p_run = sub.add_parser("run", help="Run DeepFGRN 10CV (shared TripGRN splits)")
    p_run.add_argument("--gse", default="GSE156632", help="GSE id or ALL")
    p_run.add_argument("--fold", default="1", help="fold 1..10 or ALL")
    p_run.add_argument("--keras-batch-size-cap", type=int, default=256)
    p_run.add_argument("--deepfgrn-root", default=None)
    p_run.add_argument("--force", action="store_true")

    p_sum = sub.add_parser("summarize", help="Aggregate DeepFGRN fold metrics")
    p_sum.add_argument("--gse", default="ALL")

    args = ap.parse_args()
    if args.cmd == "smoke":
        raise SystemExit(cmd_smoke())
    if args.cmd == "run":
        raise SystemExit(
            cmd_run(args.gse, args.fold, args.force, args.keras_batch_size_cap, args.deepfgrn_root)
        )
    if args.cmd == "summarize":
        raise SystemExit(cmd_summarize(args.gse))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
