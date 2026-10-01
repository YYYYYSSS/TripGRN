#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-export pancancer test confusion matrices (step01 paths).

Expression: data/exp/pancancer_scdata_step5/npy/<GSE>_{values,genes}.npy
（原始 CSV 缺失时自动回退 npy；DeepFGRN cm-only 亦同。）

  cd project_GRN
  python -m grn.scripts.export_pancancer_confusion_matrices --method all --gse ALL
  python -m grn.scripts.export_pancancer_confusion_matrices --method grn deepfgrn --gse ALL
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from grn.scripts.pancancer_step01_paths import (
    export_grn_test_cm_from_metrics,
    find_latest_deeptgi_run,
    find_latest_fgrn_run_with_ckpt,
    grn_run_dir_for_gse,
    grn_run_has_checkpoint,
    grn_root_from_script,
    list_fgrn_gse_with_checkpoint,
    list_npy_gse_ids,
    npy_ready_for_gse,
    step01_layout,
)


def _resolve_gses(grn_root: Path, gse_arg: str, out_fgrn: Path) -> List[str]:
    gse_arg = str(gse_arg).strip().upper()
    if gse_arg == "ALL":
        return list_npy_gse_ids(grn_root)
    if gse_arg == "ALL_STEP01":
        from grn.scripts.pancancer_step01_paths import list_pancancer_gse_ids

        return list_pancancer_gse_ids(grn_root)
    if gse_arg == "ALL_FGRN_CKPT":
        return list_fgrn_gse_with_checkpoint(out_fgrn)
    return [str(gse_arg).strip()]


def _run_cmd(cmd: List[str], *, cwd: Path) -> int:
    print("[export-cm]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd))


def _export_grn(grn_root: Path, gse: str) -> None:
    run_dir = grn_run_dir_for_gse(gse, grn_root)
    if not grn_run_has_checkpoint(run_dir):
        raise FileNotFoundError(f"GRN checkpoint missing under {run_dir}")
    json_p, png_p = export_grn_test_cm_from_metrics(run_dir)
    print(f"[export-cm] GRN {gse} -> {json_p.name} , {png_p.name}")


def _export_deepfgrn(
    grn_root: Path,
    gse: str,
    out_base: Path,
    deepfgrn_root: Path | None,
    run_dir: Path | None,
    *,
    retrain_on_cm_fail: bool = True,
) -> int:
    def _invoke(cm_only: bool) -> int:
        cmd = [
            sys.executable,
            "-m",
            "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
            "--output-base",
            str(out_base),
            "--gse-id",
            gse,
            "--n-folds",
            "1",
        ]
        if cm_only:
            cmd.append("--cm-only")
        if run_dir is not None:
            cmd.extend(["--run-dir", str(run_dir)])
        elif cm_only and find_latest_fgrn_run_with_ckpt(out_base, gse) is None:
            raise FileNotFoundError(f"No DeepFGRN checkpoint for {gse} under {out_base}")
        if deepfgrn_root is not None:
            cmd.extend(["--deepfgrn-root", str(deepfgrn_root)])
        return _run_cmd(cmd, cwd=grn_root)

    rc = _invoke(cm_only=True)
    if rc == 0:
        return 0
    if not retrain_on_cm_fail:
        return rc
    print(f"[export-cm] DeepFGRN {gse} cm-only failed — retraining from npy ...", flush=True)
    return _invoke(cm_only=False)


def _export_deeptgi(
    grn_root: Path,
    gse: str,
    layout,
    deeptgi_root: Path | None,
    force_train: bool,
) -> int:
    run_dir = find_latest_deeptgi_run(layout["out_tgi"], gse)
    if run_dir is None and not npy_ready_for_gse(gse, grn_root):
        raise FileNotFoundError(f"No DeepTGI run and no npy for {gse}")

    if run_dir is not None and not force_train:
        cmd = [
            sys.executable,
            "-m",
            "grn.scripts.run_deeptgi_pancancer_one",
            "--gse-id",
            gse,
            "--reuse-run-dir",
            str(run_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "grn.scripts.run_deeptgi_pancancer_one",
            "--gse-id",
            gse,
            "--exp-npy-values",
            str(layout["npy_dir"] / f"{gse}_values.npy"),
            "--exp-npy-genes",
            str(layout["npy_dir"] / f"{gse}_genes.npy"),
        ]
    if deeptgi_root is not None:
        cmd.extend(["--deeptgi-root", str(deeptgi_root)])
    return _run_cmd(cmd, cwd=grn_root)


def _export_scmgatgrn(
    grn_root: Path,
    gse: str,
    layout,
    force_train: bool,
    epochs: int,
    device: str,
) -> int:
    out_art = layout["out_scmgatgrn"] / gse
    cmd = [
        sys.executable,
        "-m",
        "grn.scripts.run_scmgatgrn_on_grn_pancancer_gse",
        "--gse",
        gse,
        "--npy-dir",
        str(layout["npy_dir"]),
        "--epochs",
        str(epochs),
        "--device",
        device,
    ]
    if out_art.joinpath("best_model.pth").is_file() and not force_train:
        cmd.append("--cm-only")
    return _run_cmd(cmd, cwd=grn_root)


def _export_genelink(
    grn_root: Path,
    gse: str,
    layout,
    force_train: bool,
    epochs: int,
    device: str,
) -> int:
    out_art = layout["out_genelink"] / gse
    cmd = [
        sys.executable,
        "-m",
        "grn.scripts.run_genelink_on_grn_pancancer_gse",
        "--gse",
        gse,
        "--npy-dir",
        str(layout["npy_dir"]),
        "--epochs",
        str(epochs),
        "--device",
        device,
    ]
    if out_art.joinpath("best_model.pth").is_file() and not force_train:
        cmd.append("--cm-only")
    return _run_cmd(cmd, cwd=grn_root)


def _expand_methods(methods: Sequence[str]) -> List[str]:
    if "all" in methods:
        return ["grn", "deepfgrn", "deeptgi", "scmgatgrn", "genelink"]
    return list(methods)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pancancer test confusion matrices (step01 npy paths).")
    default_root = grn_root_from_script()
    layout = step01_layout(default_root)
    ap.add_argument("--grn-root", type=Path, default=default_root)
    ap.add_argument(
        "--method",
        nargs="+",
        choices=("grn", "deepfgrn", "deeptgi", "scmgatgrn", "genelink", "all"),
        default=["all"],
        help="Methods to export (default: all). Uses checkpoint cm-only when available.",
    )
    ap.add_argument(
        "--gse",
        default="ALL",
        help="GSE id, ALL (npy list), ALL_STEP01, or ALL_FGRN_CKPT.",
    )
    ap.add_argument("--output-base-fgrn", type=Path, default=None)
    ap.add_argument("--run-dir-fgrn", type=Path, default=None, help="Fixed DeepFGRN run dir (single GSE).")
    ap.add_argument("--deepfgrn-root", type=Path, default=None)
    ap.add_argument("--deeptgi-root", type=Path, default=None)
    ap.add_argument("--force-train", action="store_true", help="Retrain even if checkpoint exists.")
    ap.add_argument(
        "--no-retrain-on-cm-fail",
        action="store_true",
        help="DeepFGRN: do not fall back to full training when cm-only fails.",
    )
    ap.add_argument("--epochs", type=int, default=20, help="Epochs when training scMGATGRN/GENELink.")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    grn_root = Path(args.grn_root).resolve()
    layout = step01_layout(grn_root)
    out_fgrn = Path(args.output_base_fgrn or layout["out_fgrn"]).resolve()
    methods = _expand_methods([m.lower() for m in args.method])

    gses = _resolve_gses(grn_root, args.gse, out_fgrn)
    if not gses:
        raise SystemExit("No GSE to process (check pancancer_scdata_step5/npy).")

    deepfgrn_root = args.deepfgrn_root
    if deepfgrn_root is None and os.environ.get("DEEPFGRN_ROOT"):
        deepfgrn_root = Path(os.environ["DEEPFGRN_ROOT"])
    deeptgi_root = args.deeptgi_root
    if deeptgi_root is None and os.environ.get("DEEPTGI_ROOT"):
        deeptgi_root = Path(os.environ["DEEPTGI_ROOT"])

    failed: List[str] = []
    for gse in gses:
        print(f"\n[export-cm] ===== {gse} =====", flush=True)
        if not npy_ready_for_gse(gse, grn_root) and "grn" not in methods:
            print(f"[export-cm] WARN: npy missing for {gse}", flush=True)
        for method in methods:
            try:
                if method == "grn":
                    _export_grn(grn_root, gse)
                elif method == "deepfgrn":
                    if args.force_train:
                        cmd = [
                            sys.executable,
                            "-m",
                            "grn.scripts.run_deepfgrn_pancancer_cell_unseen",
                            "--output-base",
                            str(out_fgrn),
                            "--gse-id",
                            gse,
                            "--n-folds",
                            "1",
                        ]
                        if deepfgrn_root is not None:
                            cmd.extend(["--deepfgrn-root", str(deepfgrn_root)])
                        rc = _run_cmd(cmd, cwd=grn_root)
                    else:
                        rc = _export_deepfgrn(
                            grn_root,
                            gse,
                            out_fgrn,
                            deepfgrn_root,
                            args.run_dir_fgrn,
                            retrain_on_cm_fail=not args.no_retrain_on_cm_fail,
                        )
                    if rc != 0:
                        failed.append(f"{gse}/deepfgrn")
                elif method == "deeptgi":
                    rc = _export_deeptgi(
                        grn_root, gse, layout, deeptgi_root, args.force_train
                    )
                    if rc != 0:
                        failed.append(f"{gse}/deeptgi")
                elif method == "scmgatgrn":
                    rc = _export_scmgatgrn(
                        grn_root, gse, layout, args.force_train, args.epochs, args.device
                    )
                    if rc != 0:
                        failed.append(f"{gse}/scmgatgrn")
                elif method == "genelink":
                    rc = _export_genelink(
                        grn_root, gse, layout, args.force_train, args.epochs, args.device
                    )
                    if rc != 0:
                        failed.append(f"{gse}/genelink")
            except Exception as exc:
                print(f"[export-cm] ERROR {gse}/{method}: {exc}", flush=True)
                failed.append(f"{gse}/{method}")

    if failed:
        raise SystemExit("Failed: " + ", ".join(failed))
    print(f"\n[export-cm] done — {len(gses)} GSE × methods {methods}")


if __name__ == "__main__":
    main()
