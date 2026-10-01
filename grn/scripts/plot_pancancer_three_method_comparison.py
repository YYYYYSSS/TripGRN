# -*- coding: utf-8 -*-
"""
汇总 pancancer 上三条流水线指标并导出 CSV（供 R 绘图）：GRN Step5、DeepFGRN、DeepTGI。
R 脚本 ``plot_pancancer_three_method_comparison.R`` 会额外合并（若存在）``experiments/scmgatgrn_pancancer/scmgatgrn_kegg_binary_metrics.csv``、``experiments/genelink_pancancer/genelink_kegg_binary_metrics.csv`` 与 ``experiments/genie3_pancancer/genie3_kegg_binary_metrics.csv``。

DeepFGRN 的 ``*_metrics.xlsx`` 依赖 ``openpyxl``（``pip install openpyxl``）。未安装时仍导出 CSV，但
DeepFGRN 各指标为 NaN，并打印警告。

默认不再生成 matplotlib 图；运行 R 脚本绘制统一调色板热图与柱状图。

用法：
  cd project_GRN
  python -m grn.scripts.plot_pancancer_three_method_comparison --out-dir experiments/pancancer_three_method_compare
  Rscript plot_pancancer_three_method_comparison.R experiments/pancancer_three_method_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_grn_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _macro_specificity_from_cm(cm: Any) -> float:
    arr = np.asarray(cm, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return float(np.nan)
    specs: List[float] = []
    for i in range(arr.shape[0]):
        tp = float(arr[i, i])
        fn = float(arr[i, :].sum() - tp)
        fp = float(arr[:, i].sum() - tp)
        tn = float(arr.sum() - tp - fn - fp)
        denom = tn + fp
        specs.append(float(tn / denom) if denom > 0 else float(np.nan))
    return float(np.nanmean(specs)) if specs else float(np.nan)


def _parse_grn_row(j: Dict[str, Any]) -> Dict[str, float]:
    auprc = j.get("auprc")
    mean_auprc = float(np.nanmean(np.asarray(auprc, dtype=float))) if isinstance(auprc, list) else np.nan
    cm = j.get("confusion_matrix")
    return {
        "macro_auc": float(j.get("macro_auc", np.nan)),
        "aupr": mean_auprc,
        "accuracy": float(j.get("accuracy", np.nan)),
        "f1": float(j.get("f1_macro", np.nan)),
        "mcc": float(j.get("mcc", np.nan)),
        "sensitivity": float(j.get("recall_macro", np.nan)),
        "specificity": _macro_specificity_from_cm(cm) if isinstance(cm, list) else float(np.nan),
        "acc_no_reg": float(j.get("无调控_accuracy", np.nan)),
        "acc_activate": float(j.get("激活_accuracy", np.nan)),
        "acc_repress": float(j.get("抑制_accuracy", np.nan)),
    }


def _latest_fgrn_xlsx(gse_dir: Path) -> Optional[Path]:
    if not gse_dir.is_dir():
        return None
    xs = list(gse_dir.rglob("*_metrics.xlsx"))
    if not xs:
        return None
    return max(xs, key=lambda p: p.stat().st_mtime)


def _parse_fgrn_xlsx(path: Path) -> Dict[str, float]:
    """Parse DeepFGRN *_metrics.xlsx (needs ``openpyxl``: ``pip install openpyxl``)."""
    df = pd.read_excel(path, sheet_name="metrics")
    df = df[df["Fold"].astype(str) != "Mean"]
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "macro_auc": float(r.get("Macro_AUC_OVR", np.nan)),
        "aupr": float(r.get("AUPR", np.nan)),
        "accuracy": float(np.nanmean([r.get("ACC_无调控"), r.get("ACC_激活"), r.get("ACC_抑制")])),
        "f1": float(r.get("F1", np.nan)),
        "mcc": float(r.get("MCC", np.nan)),
        "sensitivity": float(r.get("Recall", np.nan)),
        "specificity": float(np.nan),
        "acc_no_reg": float(r.get("ACC_无调控", np.nan)),
        "acc_activate": float(r.get("ACC_激活", np.nan)),
        "acc_repress": float(r.get("ACC_抑制", np.nan)),
    }


def _load_tgi_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["gse_id"], keep="last")
    return df


def _collect_all_gse(grn_root: Path, fgrn_root: Path, tgi_df: pd.DataFrame) -> List[str]:
    gses = set()
    if grn_root.is_dir():
        for d in grn_root.iterdir():
            if d.is_dir() and (d / "unseen_samples_test_metrics.json").is_file():
                gses.add(d.name)
    if fgrn_root.is_dir():
        for d in fgrn_root.iterdir():
            if d.is_dir() and _latest_fgrn_xlsx(d):
                gses.add(d.name)
    gses.update(tgi_df["gse_id"].astype(str).tolist())
    return sorted(gses)


def build_tables(
    grn_root: Path,
    fgrn_root: Path,
    tgi_summary: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    tgi = _load_tgi_summary(tgi_summary)
    gses = _collect_all_gse(grn_root, fgrn_root, tgi)
    macro_rows: List[Dict[str, Any]] = []
    perclass_rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    tgi_map = tgi.set_index("gse_id")

    for gse in gses:
        row_m: Dict[str, Any] = {"gse_id": gse}
        row_p: Dict[str, Any] = {"gse_id": gse}

        gj = grn_root / gse / "unseen_samples_test_metrics.json"
        if gj.is_file():
            gr = _parse_grn_row(_load_grn_json(gj))
            row_m["GRN_macro_auc"] = gr["macro_auc"]
            row_m["GRN_aupr"] = gr["aupr"]
            row_m["GRN_accuracy"] = gr["accuracy"]
            row_m["GRN_f1"] = gr["f1"]
            row_m["GRN_mcc"] = gr["mcc"]
            row_m["GRN_sensitivity"] = gr["sensitivity"]
            row_m["GRN_specificity"] = gr["specificity"]
            row_p["GRN_acc_无调控"] = gr["acc_no_reg"]
            row_p["GRN_acc_激活"] = gr["acc_activate"]
            row_p["GRN_acc_抑制"] = gr["acc_repress"]
        else:
            row_m["GRN_macro_auc"] = np.nan
            row_m["GRN_aupr"] = np.nan
            row_m["GRN_accuracy"] = np.nan
            row_m["GRN_f1"] = np.nan
            row_m["GRN_mcc"] = np.nan
            row_m["GRN_sensitivity"] = np.nan
            row_m["GRN_specificity"] = np.nan
            row_p["GRN_acc_无调控"] = np.nan
            row_p["GRN_acc_激活"] = np.nan
            row_p["GRN_acc_抑制"] = np.nan

        fx = _latest_fgrn_xlsx(fgrn_root / gse)
        fr: Optional[Dict[str, float]] = None
        if fx is not None:
            try:
                fr = _parse_fgrn_xlsx(fx)
            except ImportError as exc:
                print(
                    "[pancancer-compare] WARN: cannot read DeepFGRN xlsx (install openpyxl). "
                    f"Skipping {fx.name}: {exc}",
                    flush=True,
                )
                fr = {}
            except Exception as exc:
                print(f"[pancancer-compare] WARN: DeepFGRN xlsx parse failed ({fx}): {exc}", flush=True)
                fr = {}
        if fr:
            row_m["DeepFGRN_macro_auc"] = fr["macro_auc"]
            row_m["DeepFGRN_aupr"] = fr["aupr"]
            row_m["DeepFGRN_accuracy"] = fr["accuracy"]
            row_m["DeepFGRN_f1"] = fr["f1"]
            row_m["DeepFGRN_mcc"] = fr["mcc"]
            row_m["DeepFGRN_sensitivity"] = fr["sensitivity"]
            row_m["DeepFGRN_specificity"] = fr["specificity"]
            row_p["DeepFGRN_acc_无调控"] = fr["acc_no_reg"]
            row_p["DeepFGRN_acc_激活"] = fr["acc_activate"]
            row_p["DeepFGRN_acc_抑制"] = fr["acc_repress"]
        else:
            row_m["DeepFGRN_macro_auc"] = np.nan
            row_m["DeepFGRN_aupr"] = np.nan
            row_m["DeepFGRN_accuracy"] = np.nan
            row_m["DeepFGRN_f1"] = np.nan
            row_m["DeepFGRN_mcc"] = np.nan
            row_m["DeepFGRN_sensitivity"] = np.nan
            row_m["DeepFGRN_specificity"] = np.nan
            row_p["DeepFGRN_acc_无调控"] = np.nan
            row_p["DeepFGRN_acc_激活"] = np.nan
            row_p["DeepFGRN_acc_抑制"] = np.nan

        if gse in tgi_map.index:
            tr = tgi_map.loc[gse]
            row_m["DeepTGI_auc"] = float(tr["AUC_mean"])
            row_m["DeepTGI_aupr"] = float(tr["AUPR_mean"])
            row_m["DeepTGI_accuracy"] = float(tr["ACC_mean"])
            row_m["DeepTGI_f1"] = float(tr["F1_mean"])
            row_m["DeepTGI_mcc"] = float(tr["MCC_mean"])
            row_m["DeepTGI_sensitivity"] = float(tr["Recall_mean"])
            spec_col = "Specificity_mean" if "Specificity_mean" in tr.index else None
            row_m["DeepTGI_specificity"] = float(tr[spec_col]) if spec_col else np.nan
        else:
            row_m["DeepTGI_auc"] = np.nan
            row_m["DeepTGI_aupr"] = np.nan
            row_m["DeepTGI_accuracy"] = np.nan
            row_m["DeepTGI_f1"] = np.nan
            row_m["DeepTGI_mcc"] = np.nan
            row_m["DeepTGI_sensitivity"] = np.nan
            row_m["DeepTGI_specificity"] = np.nan

        macro_rows.append(row_m)
        perclass_rows.append(row_p)

        ok = (
            np.isfinite(row_m.get("GRN_macro_auc", np.nan))
            and np.isfinite(row_m.get("DeepFGRN_macro_auc", np.nan))
            and np.isfinite(row_m.get("DeepTGI_auc", np.nan))
        )
        if not ok:
            missing.append(gse)

    df_macro = pd.DataFrame(macro_rows).sort_values("gse_id").reset_index(drop=True)
    df_pc = pd.DataFrame(perclass_rows).sort_values("gse_id").reset_index(drop=True)
    return df_macro, df_pc, missing


def export_long_for_r(df_macro: pd.DataFrame, df_pc: pd.DataFrame, out_dir: Path) -> None:
    """Tidy tables for ggplot2: unified scales in R (limits 0–1)."""
    metric_map = [
        ("Macro-AUC", "GRN_macro_auc", "DeepFGRN_macro_auc", "DeepTGI_auc"),
        ("AUPR", "GRN_aupr", "DeepFGRN_aupr", "DeepTGI_aupr"),
        ("Accuracy", "GRN_accuracy", "DeepFGRN_accuracy", "DeepTGI_accuracy"),
        ("F1", "GRN_f1", "DeepFGRN_f1", "DeepTGI_f1"),
        ("MCC", "GRN_mcc", "DeepFGRN_mcc", "DeepTGI_mcc"),
        ("Sensitivity", "GRN_sensitivity", "DeepFGRN_sensitivity", "DeepTGI_sensitivity"),
        ("Specificity", "GRN_specificity", "DeepFGRN_specificity", "DeepTGI_specificity"),
    ]
    long_rows: List[Dict[str, Any]] = []
    for _, r in df_macro.iterrows():
        gse = str(r["gse_id"])
        for label, cg, cf, ct in metric_map:
            long_rows.append({"gse_id": gse, "method": "GRN", "metric": label, "value": r[cg]})
            long_rows.append({"gse_id": gse, "method": "DeepFGRN", "metric": label, "value": r[cf]})
            long_rows.append({"gse_id": gse, "method": "DeepTGI", "metric": label, "value": r[ct]})
    pd.DataFrame(long_rows).to_csv(out_dir / "plot_inputs_macro_long.csv", index=False)

    cls_map = [
        ("No regulation", "GRN_acc_无调控", "DeepFGRN_acc_无调控"),
        ("Activation", "GRN_acc_激活", "DeepFGRN_acc_激活"),
        ("Repression", "GRN_acc_抑制", "DeepFGRN_acc_抑制"),
    ]
    pc_rows: List[Dict[str, Any]] = []
    for _, r in df_pc.iterrows():
        gse = str(r["gse_id"])
        for cls_label, cg, cf in cls_map:
            pc_rows.append({"gse_id": gse, "method": "GRN", "class": cls_label, "value": r[cg]})
            pc_rows.append({"gse_id": gse, "method": "DeepFGRN", "class": cls_label, "value": r[cf]})
    pd.DataFrame(pc_rows).to_csv(out_dir / "plot_inputs_perclass_long.csv", index=False)


def export_macro_means_for_bar(df_macro: pd.DataFrame, out_dir: Path) -> None:
    """One row per method x metric for bar chart."""
    metric_map = [
        ("Macro-AUC", "GRN_macro_auc", "DeepFGRN_macro_auc", "DeepTGI_auc"),
        ("AUPR", "GRN_aupr", "DeepFGRN_aupr", "DeepTGI_aupr"),
        ("Accuracy", "GRN_accuracy", "DeepFGRN_accuracy", "DeepTGI_accuracy"),
        ("F1", "GRN_f1", "DeepFGRN_f1", "DeepTGI_f1"),
        ("MCC", "GRN_mcc", "DeepFGRN_mcc", "DeepTGI_mcc"),
        ("Sensitivity", "GRN_sensitivity", "DeepFGRN_sensitivity", "DeepTGI_sensitivity"),
        ("Specificity", "GRN_specificity", "DeepFGRN_specificity", "DeepTGI_specificity"),
    ]
    rows = []
    for label, cg, cf, ct in metric_map:
        rows.append({"metric": label, "method": "GRN", "mean_value": float(df_macro[cg].mean(skipna=True))})
        rows.append({"metric": label, "method": "DeepFGRN", "mean_value": float(df_macro[cf].mean(skipna=True))})
        rows.append({"metric": label, "method": "DeepTGI", "mean_value": float(df_macro[ct].mean(skipna=True))})
    pd.DataFrame(rows).to_csv(out_dir / "plot_inputs_macro_means_long.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export pancancer three-method tables for R plots.")
    grn_default = Path("experiments/pancancer_scdata_step5")
    fgrn_default = Path("experiments/deepfgrn_baseline_pancancer")
    tgi_default = Path("experiments/deeptgi_pancancer/cross_cancer_summary.csv")
    out_default = Path("experiments/pancancer_three_method_compare")

    ap.add_argument("--grn-root", type=Path, default=grn_default)
    ap.add_argument("--fgrn-root", type=Path, default=fgrn_default)
    ap.add_argument("--tgi-summary", type=Path, default=tgi_default)
    ap.add_argument("--out-dir", type=Path, default=out_default)
    args = ap.parse_args()

    grn_root = Path(args.grn_root).resolve()
    fgrn_root = Path(args.fgrn_root).resolve()
    tgi_path = Path(args.tgi_summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df_macro, df_pc, missing = build_tables(grn_root, fgrn_root, tgi_path)

    df_macro.to_csv(out_dir / "pancancer_three_method_macro.csv", index=False, encoding="utf-8-sig")
    df_pc.to_csv(out_dir / "pancancer_grn_fgrn_perclass_accuracy.csv", index=False, encoding="utf-8-sig")
    export_long_for_r(df_macro, df_pc, out_dir)
    export_macro_means_for_bar(df_macro, out_dir)

    status_lines = [
        f"gse_count={len(df_macro)}",
        f"grn_root={grn_root}",
        f"fgrn_root={fgrn_root}",
        f"tgi_summary={tgi_path}",
    ]
    if missing:
        status_lines.append("incomplete_gse_missing_any_method=" + ",".join(sorted(missing)))
    else:
        status_lines.append("all_listed_gse_have_three_methods=True")
    status_lines.append("next: Rscript plot_pancancer_three_method_comparison.R <out-dir>")
    (out_dir / "collection_status.txt").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    print("[pancancer-compare] exported CSV ->", out_dir)
    print("[pancancer-compare]", "\n  ".join(status_lines))


if __name__ == "__main__":
    main()
