#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export TripGRN vs DeepFGRN repression-class comparison for current workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from grn.data.pancancer_paths import grn_root_from_here

REPRESS = "抑制"
ACT = "激活"


def _repress_col(df: pd.DataFrame, prefix: str) -> str:
    for c in df.columns:
        if c == f"{prefix}_{REPRESS}" or (str(c).startswith(f"{prefix}_") and REPRESS in str(c)):
            return c
    raise KeyError(f"{prefix}_*{REPRESS}* not in {list(df.columns)}")


def _act_col(df: pd.DataFrame, prefix: str) -> str:
    for c in df.columns:
        if c == f"{prefix}_{ACT}" or (str(c).startswith(f"{prefix}_") and ACT in str(c)):
            return c
    raise KeyError(f"{prefix}_*{ACT}*")


def _load_pair(root: Path, exp_name: str):
    t = pd.read_csv(root / "experiments" / exp_name / "summary" / "tripgrn_10cv_fold_metrics.csv")
    d = pd.read_csv(root / "experiments" / exp_name / "summary" / "deepfgrn_10cv_fold_metrics.csv")
    return t, d


def _gse_table(t: pd.DataFrame, d: pd.DataFrame, network: str) -> pd.DataFrame:
    rows = []
    for gse in sorted(t.gse_id.unique()):
        tt = t[t.gse_id == gse]
        dd = d[d.gse_id == gse]
        row = {"network": network, "gse_id": gse}
        for prefix, key in (("auc_ovr", "auc"), ("acc_class", "acc"), ("auprc", "auprc")):
            tc = _repress_col(tt, prefix)
            dc = _repress_col(dd, prefix)
            row[f"trip_{key}"] = float(tt[tc].mean())
            row[f"deep_{key}"] = float(dd[dc].mean())
            row[f"delta_{key}"] = row[f"trip_{key}"] - row[f"deep_{key}"]
            row[f"trip_{key}_std"] = float(tt[tc].std(ddof=0))
            row[f"deep_{key}_std"] = float(dd[dc].std(ddof=0))
        for prefix, key in (("auc_ovr", "act_auc"), ("acc_class", "act_acc")):
            row[f"trip_{key}"] = float(tt[_act_col(tt, prefix)].mean())
            row[f"deep_{key}"] = float(dd[_act_col(dd, prefix)].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    root = grn_root_from_here()
    out_dir = root / "experiments" / "_reports" / "repression_trip_vs_deep"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gse = []
    pan_rows = []
    for network, exp in (
        ("KEGG", "step01_pancancer_epithelial_10cv"),
        ("TRRUST", "step01_pancancer_epithelial_10cv_trrust"),
    ):
        t, d = _load_pair(root, exp)
        gdf = _gse_table(t, d, network)
        all_gse.append(gdf)
        pan_rows.append(
            {
                "network": network,
                "n_folds": len(t),
                "n_gse": len(gdf),
                "trip_auc": float(t[_repress_col(t, "auc_ovr")].mean()),
                "deep_auc": float(d[_repress_col(d, "auc_ovr")].mean()),
                "trip_acc": float(t[_repress_col(t, "acc_class")].mean()),
                "deep_acc": float(d[_repress_col(d, "acc_class")].mean()),
                "trip_auprc": float(t[_repress_col(t, "auprc")].mean()),
                "deep_auprc": float(d[_repress_col(d, "auprc")].mean()),
                "wins_auc": int((gdf["delta_auc"] > 0).sum()),
                "wins_acc": int((gdf["delta_acc"] > 0).sum()),
                "wins_auprc": int((gdf["delta_auprc"] > 0).sum()),
                "trip_auc_gse_median": float(gdf["trip_auc"].median()),
                "deep_auc_gse_median": float(gdf["deep_auc"].median()),
                "trip_acc_gse_median": float(gdf["trip_acc"].median()),
                "deep_acc_gse_median": float(gdf["deep_acc"].median()),
            }
        )
        pan_rows[-1]["delta_auc"] = pan_rows[-1]["trip_auc"] - pan_rows[-1]["deep_auc"]
        pan_rows[-1]["delta_acc"] = pan_rows[-1]["trip_acc"] - pan_rows[-1]["deep_acc"]

    gse_df = pd.concat(all_gse, ignore_index=True)
    pan_df = pd.DataFrame(pan_rows)
    gse_df.to_csv(out_dir / "repression_by_gse.csv", index=False, encoding="utf-8-sig")
    pan_df.to_csv(out_dir / "repression_pancancer_summary.csv", index=False, encoding="utf-8-sig")

    seed_rows = []
    t42, d42 = _load_pair(root, "step01_pancancer_epithelial_10cv_trrust")
    seed_rows.append(
        {
            "seed": 42,
            "complete": True,
            "network": "TRRUST",
            "trip_auc": float(t42[_repress_col(t42, "auc_ovr")].mean()),
            "deep_auc": float(d42[_repress_col(d42, "auc_ovr")].mean()),
            "trip_acc": float(t42[_repress_col(t42, "acc_class")].mean()),
            "deep_acc": float(d42[_repress_col(d42, "acc_class")].mean()),
        }
    )
    seed_rows[-1]["delta_auc"] = seed_rows[-1]["trip_auc"] - seed_rows[-1]["deep_auc"]
    seed_rows[-1]["delta_acc"] = seed_rows[-1]["trip_acc"] - seed_rows[-1]["deep_acc"]

    for p in sorted((root / "experiments").glob("step01_pancancer_epithelial_10cv_trrust_seed*")):
        seed = int(p.name.split("seed")[-1])
        tp = p / "summary" / "tripgrn_10cv_fold_metrics.csv"
        dp = p / "summary" / "deepfgrn_10cv_fold_metrics.csv"
        if not (tp.is_file() and dp.is_file()):
            seed_rows.append({"seed": seed, "complete": False, "n_t": 0, "n_d": 0})
            continue
        t = pd.read_csv(tp)
        d = pd.read_csv(dp)
        if len(t) < 100 or len(d) < 100:
            seed_rows.append({"seed": seed, "complete": False, "n_t": len(t), "n_d": len(d)})
            continue
        row = {
            "seed": seed,
            "complete": True,
            "network": "TRRUST",
            "trip_auc": float(t[_repress_col(t, "auc_ovr")].mean()),
            "deep_auc": float(d[_repress_col(d, "auc_ovr")].mean()),
            "trip_acc": float(t[_repress_col(t, "acc_class")].mean()),
            "deep_acc": float(d[_repress_col(d, "acc_class")].mean()),
        }
        row["delta_auc"] = row["trip_auc"] - row["deep_auc"]
        row["delta_acc"] = row["trip_acc"] - row["deep_acc"]
        seed_rows.append(row)

    seed_df = pd.DataFrame(seed_rows).sort_values("seed")
    seed_df.to_csv(out_dir / "repression_trrust_seedscan.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 抑制类性能比对：TripGRN vs DeepFGRN",
        "",
        "范围：流程中**同时具备两种方法**的步骤（泛癌上皮 10CV）。",
        "消融 / bulk / 细胞类型 / 迁移 / 脑转移案例仅有 TripGRN，不纳入双方法比对。",
        "",
        "主指标：抑制类 **OvR-AUROC**、**类准确率**。",
        "说明：DeepFGRN 汇总 JSON 中逐类 AUPRC 为宏平均占位，不作逐类解读。",
        "",
        "## 1. 泛癌总览（全部 fold 均值）",
        "",
        "| 金标准 | Trip 抑制 AUC | Deep 抑制 AUC | ΔAUC | Trip 抑制 Acc | Deep 抑制 Acc | ΔAcc | AUC 胜场 | Acc 胜场 |",
        "|--------|---------------|---------------|------|---------------|---------------|------|----------|----------|",
    ]
    for _, r in pan_df.iterrows():
        lines.append(
            "| {net} | {ta:.3f} | {da:.3f} | {dau:+.3f} | {tac:.3f} | {dac:.3f} | {dacc:+.3f} | {wa}/{n} | {wc}/{n} |".format(
                net=r["network"],
                ta=r["trip_auc"],
                da=r["deep_auc"],
                dau=r["delta_auc"],
                tac=r["trip_acc"],
                dac=r["deep_acc"],
                dacc=r["delta_acc"],
                wa=int(r["wins_auc"]),
                wc=int(r["wins_acc"]),
                n=int(r["n_gse"]),
            )
        )

    lines += ["", "## 2. 逐队列（10 折均值）", ""]
    for network in ("KEGG", "TRRUST"):
        lines += [
            f"### {network}",
            "",
            "| GSE | Trip AUC | Deep AUC | ΔAUC | Trip Acc | Deep Acc | ΔAcc |",
            "|-----|----------|----------|------|----------|----------|------|",
        ]
        sub = gse_df[gse_df.network == network]
        for _, r in sub.iterrows():
            lines.append(
                "| {g} | {ta:.3f} | {da:.3f} | {dau:+.3f} | {tac:.3f} | {dac:.3f} | {dacc:+.3f} |".format(
                    g=r["gse_id"],
                    ta=r["trip_auc"],
                    da=r["deep_auc"],
                    dau=r["delta_auc"],
                    tac=r["trip_acc"],
                    dac=r["deep_acc"],
                    dacc=r["delta_acc"],
                )
            )
        lines.append("")

    lines += [
        "## 3. TRRUST 多种子（抑制类，已完成）",
        "",
        "| seed | Trip AUC | Deep AUC | ΔAUC | Trip Acc | Deep Acc | ΔAcc |",
        "|------|----------|----------|------|----------|----------|------|",
    ]
    for _, r in seed_df[seed_df["complete"] == True].iterrows():  # noqa: E712
        lines.append(
            "| {s} | {ta:.3f} | {da:.3f} | {dau:+.3f} | {tac:.3f} | {dac:.3f} | {dacc:+.3f} |".format(
                s=int(r["seed"]),
                ta=r["trip_auc"],
                da=r["deep_auc"],
                dau=r["delta_auc"],
                tac=r["trip_acc"],
                dac=r["deep_acc"],
                dacc=r["delta_acc"],
            )
        )

    lines += [
        "",
        "## 4. 结论要点",
        "",
        "- **KEGG**：抑制类 Trip 全面更强（AUC/Acc 均为 10/10）；Deep 抑制 Acc 均值约 0.62，明显偏弱。",
        "- **TRRUST**：抑制类 Deep 在 AUC 上更占优；Acc 接近；已完成多种子方向一致。",
        "- 后续仅 Trip 的步骤无法做同设定双方法抑制类对照。",
        "",
    ]
    (out_dir / "REPRESSION_Trip_vs_Deep.md").write_text("\n".join(lines), encoding="utf-8")
    print(pan_df.to_string(index=False))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
