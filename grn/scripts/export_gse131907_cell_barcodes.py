#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出 GSE131907 Step05 对比用的细胞 barcode：

- **tLung**（Tissue origins）：肺部原发肿瘤样本来源细胞
- **mBrain**：脑转移样本来源细胞

列名匹配规则与 ``train_metastasis_*`` / ``materialize_gse131907_brain_nonbrain_npys`` 一致
（细胞 barcode 后缀 ``_<Samples>``）。

说明：Step05 bat 里 historical 变量名 non-mBrain / mBrain 的 **non-mBrain 含淋巴结、胸水等全部非脑部位**；
若虚拟敲除要对齐「原发 vs 脑转移」生物学对比，应使用本脚本导出的 **tLung vs mBrain** 列表。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from grn.data.gse131907_feature_summary import (
    load_sample_tissue_pairs,
    match_column_to_sample_id,
)


def split3(n: int, w1: int, w2: int, w3: int) -> tuple[int, int, int]:
    w = w1 + w2 + w3
    a = n * w1 // w
    b = n * w2 // w
    c = n - a - b
    return a, b, c


def add_column_split(cells: Sequence[str], split: str = "1,2,7") -> pd.DataFrame:
    w1, w2, w3 = [int(x.strip()) for x in split.split(",")]
    c_tr, c_va, c_te = split3(len(cells), w1, w2, w3)
    part = ["train"] * c_tr + ["val"] * c_va + ["test"] * c_te
    return pd.DataFrame({"cell_barcode": list(cells), "column_split_1_2_7": part})


def _load_all_precomputed_columns(pre: Path) -> np.ndarray:
    nb = np.load(pre / "nonbrain_column_names.npy", allow_pickle=True)
    mb = np.load(pre / "brain_column_names.npy", allow_pickle=True)
    return np.concatenate([nb.astype(str), mb.astype(str)])


def _cells_by_tissue(
    columns: Sequence[str],
    sample_to_tissue: Dict[str, str],
    sample_ids: Sequence[str],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for col in columns:
        sid = match_column_to_sample_id(str(col), sample_ids)
        if sid is None:
            out["__UNMAPPED__"].append(str(col))
            continue
        tissue = sample_to_tissue.get(sid, "__UNKNOWN__")
        out[tissue].append(str(col))
    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export GSE131907 tLung (primary lung) vs mBrain (brain met) cell barcodes."
    )
    root = Path(__file__).resolve().parents[2]
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "other" / "GSE131907" / "cell_barcodes_for_virtual_ko",
    )
    ap.add_argument(
        "--precomputed-dir",
        type=Path,
        default=root / "data" / "exp" / "GSE131907" / "precomputed_brain_nonbrain_npy",
    )
    ap.add_argument(
        "--feature-xlsx",
        type=Path,
        default=root / "data" / "other" / "GSE131907" / "GSE131907_Lung_Cancer_Feature_Summary.xlsx",
    )
    ap.add_argument("--sample-split", default="1,2,7", help="与 step05 bat FT_SAMPLE_SPLIT 一致")
    ap.add_argument(
        "--split-part",
        choices=["all", "train", "val", "test"],
        default="all",
        help="仅导出指定列划分子集；test 与 step05 case study --expression-column-split-part test 一致",
    )
    ap.add_argument(
        "--primary-tissue",
        default="tLung",
        help="Feature Summary 中视为「肺部原发」的 Tissue origins（默认 tLung）",
    )
    ap.add_argument(
        "--met-tissue",
        default="mBrain",
        help="Feature Summary 中视为「脑转移」的 Tissue origins（默认 mBrain）",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = Path(args.precomputed_dir).resolve()
    xlsx = Path(args.feature_xlsx).resolve()

    pairs = load_sample_tissue_pairs(str(xlsx))
    sample_to_tissue = {s: t for s, t in pairs}
    sample_ids = sorted(sample_to_tissue.keys(), key=len, reverse=True)

    all_cols = _load_all_precomputed_columns(pre)
    by_tissue = _cells_by_tissue(all_cols, sample_to_tissue, sample_ids)

    primary_t = str(args.primary_tissue)
    met_t = str(args.met_tissue)
    primary_cells = by_tissue.get(primary_t, [])
    met_cells = by_tissue.get(met_t, [])
    if not primary_cells:
        raise SystemExit(f"未找到 Tissue origins == {primary_t!r} 的细胞列")
    if not met_cells:
        raise SystemExit(f"未找到 Tissue origins == {met_t!r} 的细胞列")

    primary_samples = sorted(s for s, t in pairs if t == primary_t)
    met_samples = sorted(s for s, t in pairs if t == met_t)

    df_pri = add_column_split(primary_cells, args.sample_split)
    df_pri.insert(0, "domain", "tLung_primary")
    df_pri.insert(1, "tissue_origin", primary_t)
    df_pri["sample_id"] = df_pri["cell_barcode"].map(
        lambda c: match_column_to_sample_id(c, sample_ids) or ""
    )

    df_met = add_column_split(met_cells, args.sample_split)
    df_met.insert(0, "domain", "mBrain_met")
    df_met.insert(1, "tissue_origin", met_t)
    df_met["sample_id"] = df_met["cell_barcode"].map(
        lambda c: match_column_to_sample_id(c, sample_ids) or ""
    )

    split_part = str(args.split_part)
    if split_part != "all":
        df_pri = df_pri[df_pri["column_split_1_2_7"] == split_part].reset_index(drop=True)
        df_met = df_met[df_met["column_split_1_2_7"] == split_part].reset_index(drop=True)
        suffix = f"_{split_part}"
    else:
        suffix = ""

    p_pri = out_dir / f"GSE131907_tLung_primary_cell_barcodes{suffix}.csv"
    p_met = out_dir / f"GSE131907_mBrain_met_cell_barcodes{suffix}.csv"
    p_all = out_dir / f"GSE131907_tLung_primary_vs_mBrain_met_cell_barcodes{suffix}.csv"
    df_pri.to_csv(p_pri, index=False, encoding="utf-8-sig")
    df_met.to_csv(p_met, index=False, encoding="utf-8-sig")
    pd.concat([df_pri, df_met], ignore_index=True).to_csv(p_all, index=False, encoding="utf-8-sig")

    meta = {
        "step05_intended_comparison": {
            "primary_lung": {
                "tissue_origin": primary_t,
                "label_zh": "肺部原发（tLung 样本来源细胞）",
                "n_samples": len(primary_samples),
                "sample_ids": primary_samples,
                "n_cells": len(primary_cells),
            },
            "brain_metastasis": {
                "tissue_origin": met_t,
                "label_zh": "脑转移（mBrain 样本来源细胞）",
                "n_samples": len(met_samples),
                "sample_ids": met_samples,
                "n_cells": len(met_cells),
            },
        },
        "note_non_mbrain_vs_mbrain": (
            "Step05 bat 历史路径名 non-mBrain 含 nLung/nLN/mLN/PE/tL/B 等全部非脑部位；"
            "虚拟敲除若对齐「原发 vs 脑转移」请用本目录 tLung_primary / mBrain_met 文件。"
        ),
        "column_split": args.sample_split,
        "split_part_exported": split_part,
        "source": {
            "precomputed_dir": str(pre),
            "feature_summary_xlsx": str(xlsx),
        },
        "tissue_cell_counts_all_precomputed": {
            k: len(v) for k, v in sorted(by_tissue.items(), key=lambda x: (-len(x[1]), x[0]))
        },
        "output_files": {
            "primary": str(p_pri),
            "metastasis": str(p_met),
            "combined": str(p_all),
        },
    }
    (out_dir / "README_cell_barcodes.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[export] wrote -> {out_dir} (split_part={split_part})")
    print(f"  tLung primary: {len(df_pri)} cells exported ({len(primary_samples)} samples total)")
    print(f"  mBrain met:    {len(df_met)} cells exported ({len(met_samples)} samples total)")


if __name__ == "__main__":
    main()
