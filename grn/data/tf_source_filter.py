"""
GRN 边表按「调控源基因」筛选：用于剔除 KEGG PPrel 中非典型转录调控的边。

说明：仓库内 KEGG CSV 已将多种 PPrel（含磷酸化 +p 等）粗归为 activation/inhibition，
无法仅凭 relation_subtype 区分「TF→靶基因转录调控」与「激酶磷酸化激活蛋白」等。
若仅保留 source 为人源注释转录因子的行，可去掉如 CHEK2→TP53 这类边。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series


def _normalize_symbol_token(sym: str) -> str:
    x = sym.strip()
    x = x.replace('"', "").replace("'", "")
    return x


def load_regulator_symbol_set(path: str | Path) -> set[str]:
    """
    从文本文件读取基因符号集合：每行一个符号；空行与 # 开头行忽略。
    若行中含逗号/制表符，取第一个字段（便于粘贴 CSV 首列）。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"regulator symbol list not found: {p.resolve()}")
    out: set[str] = set()
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split(",")[0].split("\t")[0].strip()
        if not token:
            continue
        out.add(_normalize_symbol_token(token))
    if not out:
        raise ValueError(f"no symbols loaded from {p}")
    return out


def filter_grn_by_source_symbols(
    grn_df: pd.DataFrame,
    tf_col: str,
    *,
    include_tf: set[str] | None = None,
    exclude_sources: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    按 source 列筛选 GRN 行。

    - exclude_sources: 若给定，先剔除 source ∈ exclude_sources
    - include_tf: 若给定，再仅保留 source ∈ include_tf
    """
    n0 = len(grn_df)
    df = grn_df
    if exclude_sources:
        src = normalize_gene_symbol_series(df[tf_col].astype(str))
        df = df.loc[~src.isin(exclude_sources)].copy()
    n_after_exclude = len(df)
    if include_tf is not None:
        src = normalize_gene_symbol_series(df[tf_col].astype(str))
        df = df.loc[src.isin(include_tf)].copy()
    stats = {
        "rows_input": n0,
        "rows_after_exclude": n_after_exclude,
        "rows_output": len(df),
    }
    return df, stats
