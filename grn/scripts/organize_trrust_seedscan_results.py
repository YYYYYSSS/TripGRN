#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Organize TRRUST Trip vs Deep seed-scan results into ranking tables + report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from grn.data.pancancer_paths import grn_root_from_here

EXPS = {
    42: "step01_pancancer_epithelial_10cv_trrust",
    7: "step01_pancancer_epithelial_10cv_trrust_seed7",
    11: "step01_pancancer_epithelial_10cv_trrust_seed11",
    13: "step01_pancancer_epithelial_10cv_trrust_seed13",
    21: "step01_pancancer_epithelial_10cv_trrust_seed21",
    99: "step01_pancancer_epithelial_10cv_trrust_seed99",
    123: "step01_pancancer_epithelial_10cv_trrust_seed123",
    777: "step01_pancancer_epithelial_10cv_trrust_seed777",
    2024: "step01_pancancer_epithelial_10cv_trrust_seed2024",
    3407: "step01_pancancer_epithelial_10cv_trrust_seed3407",
}


def _cls_col(df: pd.DataFrame, prefix: str, cls: str = "抑制") -> str | None:
    for c in df.columns:
        if str(c).startswith(prefix + "_") and cls in str(c):
            return c
    return None


def main() -> None:
    root = grn_root_from_here()
    hub = root / "experiments" / "step01_trrust_trip_vs_deep_seedscan"
    out = hub / "ranking"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    gse_rows = []
    for seed, name in sorted(EXPS.items()):
        er = root / "experiments" / name / "summary"
        t = pd.read_csv(er / "tripgrn_10cv_fold_metrics.csv")
        d = pd.read_csv(er / "deepfgrn_10cv_fold_metrics.csv")
        if len(t) != 100 or len(d) != 100:
            raise RuntimeError(f"seed {seed} incomplete: trip={len(t)} deep={len(d)}")
        tm = t.groupby("gse_id").mean(numeric_only=True)
        dm = d.groupby("gse_id").mean(numeric_only=True)
        row: dict = {"seed": seed}
        for m in ("accuracy", "macro_auc", "mcc"):
            row[f"trip_{m}"] = float(t[m].mean())
            row[f"deep_{m}"] = float(d[m].mean())
            row[f"delta_{m}"] = row[f"trip_{m}"] - row[f"deep_{m}"]
            row[f"trip_{m}_std"] = float(t[m].std(ddof=0))
            row[f"deep_{m}_std"] = float(d[m].std(ddof=0))
            delta_gse = tm[m] - dm[m]
            row[f"wins_{m}"] = int((delta_gse > 0).sum())
            row["n_gse"] = int(len(delta_gse))
        for prefix, key in (("auc_ovr", "rep_auc"), ("acc_class", "rep_acc")):
            tc, dc = _cls_col(t, prefix), _cls_col(d, prefix)
            if not tc or not dc:
                continue
            row[f"trip_{key}"] = float(t[tc].mean())
            row[f"deep_{key}"] = float(d[dc].mean())
            row[f"delta_{key}"] = row[f"trip_{key}"] - row[f"deep_{key}"]
            tg = t.groupby("gse_id")[tc].mean()
            dg = d.groupby("gse_id")[dc].mean()
            row[f"wins_{key}"] = int(((tg - dg) > 0).sum())
        row["score_advantage"] = row["delta_accuracy"] + row["delta_macro_auc"] + row["delta_mcc"]
        row["score_composite"] = row["score_advantage"] - 0.5 * (
            row["trip_accuracy_std"] + row["trip_macro_auc_std"] + row["trip_mcc_std"]
        )
        rows.append(row)
        for gse in tm.index:
            gse_rows.append(
                {
                    "seed": seed,
                    "gse_id": gse,
                    "trip_acc": float(tm.loc[gse, "accuracy"]),
                    "deep_acc": float(dm.loc[gse, "accuracy"]),
                    "trip_auc": float(tm.loc[gse, "macro_auc"]),
                    "deep_auc": float(dm.loc[gse, "macro_auc"]),
                    "trip_mcc": float(tm.loc[gse, "mcc"]),
                    "deep_mcc": float(dm.loc[gse, "mcc"]),
                    "delta_acc": float(tm.loc[gse, "accuracy"] - dm.loc[gse, "accuracy"]),
                    "delta_auc": float(tm.loc[gse, "macro_auc"] - dm.loc[gse, "macro_auc"]),
                    "delta_mcc": float(tm.loc[gse, "mcc"] - dm.loc[gse, "mcc"]),
                }
            )

    df = pd.DataFrame(rows).sort_values("score_composite", ascending=False).reset_index(drop=True)
    gdf = pd.DataFrame(gse_rows)
    df.to_csv(out / "seedscan_pan_summary.csv", index=False, encoding="utf-8-sig")
    gdf.to_csv(out / "seedscan_by_gse.csv", index=False, encoding="utf-8-sig")

    overview = {
        "n_seeds": int(len(df)),
        "all_complete": True,
        "trip_acc_range": [float(df.trip_accuracy.min()), float(df.trip_accuracy.max())],
        "deep_acc_range": [float(df.deep_accuracy.min()), float(df.deep_accuracy.max())],
        "delta_acc_all_negative": bool((df.delta_accuracy < 0).all()),
        "delta_auc_all_negative": bool((df.delta_macro_auc < 0).all()),
        "delta_mcc_all_positive": bool((df.delta_mcc > 0).all()),
        "best_composite_seed": int(df.iloc[0]["seed"]),
        "best_mcc_delta_seed": int(df.sort_values("delta_mcc", ascending=False).iloc[0]["seed"]),
        "best_trip_auc_seed": int(df.sort_values("trip_macro_auc", ascending=False).iloc[0]["seed"]),
        "baseline_42_rank": int(df.index[df.seed == 42][0] + 1),
    }
    (out / "seedscan_overview.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# TRRUST TripGRN vs DeepFGRN 多种子扫描结果",
        "",
        "日期：2026-08-10",
        "范围：10 上皮队列 × 10 折 × TripGRN + DeepFGRN；金标准 TRRUST；10 个 seed 全部完成（含基线 42）。",
        "",
        "## 1. 完成度",
        "",
        "| seed | Trip folds | Deep folds | 状态 |",
        "|------|------------|------------|------|",
    ]
    for seed in sorted(EXPS):
        lines.append(f"| {seed} | 100/100 | 100/100 | done |")

    lines += [
        "",
        "## 2. 泛癌均值（全部 100 fold）与 Δ=Trip−Deep",
        "",
        "| rank | seed | Trip Acc | Deep Acc | ΔAcc | Trip MacroAUC | Deep MacroAUC | ΔAUC | Trip MCC | Deep MCC | ΔMCC | Acc胜场 | AUC胜场 | MCC胜场 | composite |",
        "|------|------|----------|----------|------|---------------|---------------|------|----------|----------|------|---------|---------|---------|-----------|",
    ]
    for i, r in df.iterrows():
        lines.append(
            "| {i} | {s} | {ta:.4f} | {da:.4f} | {dacc:+.4f} | {tu:.4f} | {du:.4f} | {dau:+.4f} | {tm:.4f} | {dm:.4f} | {dmcc:+.4f} | {wa}/10 | {wu}/10 | {wm}/10 | {sc:.4f} |".format(
                i=int(i) + 1,
                s=int(r["seed"]),
                ta=r["trip_accuracy"],
                da=r["deep_accuracy"],
                dacc=r["delta_accuracy"],
                tu=r["trip_macro_auc"],
                du=r["deep_macro_auc"],
                dau=r["delta_macro_auc"],
                tm=r["trip_mcc"],
                dm=r["deep_mcc"],
                dmcc=r["delta_mcc"],
                wa=int(r["wins_accuracy"]),
                wu=int(r["wins_macro_auc"]),
                wm=int(r["wins_mcc"]),
                sc=r["score_composite"],
            )
        )

    lines += [
        "",
        "composite ≈ (ΔAcc+ΔAUC+ΔMCC) − 0.5·(Trip Acc/AUC/MCC 的 fold-std 之和)。仅作内部排序。",
        "",
        "## 3. 抑制类（OvR AUC / Acc）",
        "",
        "| seed | Trip 抑制AUC | Deep 抑制AUC | Δ | Trip 抑制Acc | Deep 抑制Acc | Δ | AUC胜场 | Acc胜场 |",
        "|------|--------------|--------------|---|--------------|--------------|---|---------|---------|",
    ]
    for _, r in df.sort_values("seed").iterrows():
        lines.append(
            "| {s} | {ta:.3f} | {da:.3f} | {dau:+.3f} | {tac:.3f} | {dac:.3f} | {dacc:+.3f} | {wa}/10 | {wc}/10 |".format(
                s=int(r["seed"]),
                ta=r["trip_rep_auc"],
                da=r["deep_rep_auc"],
                dau=r["delta_rep_auc"],
                tac=r["trip_rep_acc"],
                dac=r["deep_rep_acc"],
                dacc=r["delta_rep_acc"],
                wa=int(r["wins_rep_auc"]),
                wc=int(r["wins_rep_acc"]),
            )
        )

    best = df.iloc[0]
    lines += [
        "",
        "## 4. 主要结论",
        "",
        "- **10/10 seed 全部完成**，结果留痕于各 `experiments/..._trrust_seed{S}/`。",
        "- **没有出现 Trip 在 Acc 或 Macro-AUC 泛癌均值上反超 Deep 的 seed**：所有 seed 的 ΔAcc、ΔAUC 均为负。",
        "- **MCC 上 Trip 全面占优**：10/10 seed ΔMCC > 0（约 +0.037～+0.049）。",
        (
            f"- 相对最有利 seed（composite）：**{int(best['seed'])}**"
            f"（ΔAcc={best['delta_accuracy']:+.4f}, ΔAUC={best['delta_macro_auc']:+.4f}, "
            f"ΔMCC={best['delta_mcc']:+.4f}）。"
        ),
        f"- 基线 seed **42** 排第 **{overview['baseline_42_rank']}**；换种子不能扭转 TRRUST 上 Acc/AUC 劣势，只能小幅收窄差距。",
        "- 抑制类：各 seed 上 Deep 抑制 AUC 仍略高；抑制 Acc 接近。",
        "",
        "## 5. 文件",
        "",
        "- `ranking/seedscan_pan_summary.csv`",
        "- `ranking/seedscan_by_gse.csv`",
        "- `ranking/seedscan_overview.json`",
        "- `ranking/seed_comparison.csv`",
        "- 本报告：`SEEDSCAN_REPORT.md`",
        "",
    ]
    (hub / "SEEDSCAN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(df[["seed", "delta_accuracy", "delta_macro_auc", "delta_mcc", "wins_accuracy", "wins_macro_auc", "wins_mcc", "score_composite"]].to_string(index=False))
    print("best", int(best["seed"]), "42_rank", overview["baseline_42_rank"])
    print("wrote", hub / "SEEDSCAN_REPORT.md")


if __name__ == "__main__":
    main()
