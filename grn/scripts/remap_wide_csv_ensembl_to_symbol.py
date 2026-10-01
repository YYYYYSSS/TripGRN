"""
宽大表达 CSV（首列基因名，其余列为细胞）中行名为 Ensembl（ENSG…）时，映射为 HGNC symbol 并按 symbol 合并行（表达求和）。

与 remap_wide_npy_ensembl_to_symbol 共用 Ensembl REST、`prepare_tcga_luad_aligned` 的批量 lookup 与同目录缓存 JSON，
便于与 Step5 npy 流程共享映射结果。

典型用法（补 pancancer / GSE156632）：
  python -m grn.scripts.remap_wide_csv_ensembl_to_symbol ^
    --input-csv data/exp/pancancer/GSE156632.csv ^
    --output-csv data/exp/pancancer/GSE156632_symbol.csv ^
    --cache-json data/exp/pancancer/ensembl_symbol_cache.json

大矩阵采用分块读入，在内存中为每个 symbol 维护长度 n_cells 的累加向量（约为 n_symbol × n_cells × 4 字节）。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from grn.data.preprocessing import normalize_gene_symbol_series
from grn.scripts.prepare_tcga_luad_aligned import (
    _chunks,
    _ensembl_lookup_id_batch,
    _strip_ensembl_version,
)


def _is_ensembl_row(g: str) -> bool:
    x = _strip_ensembl_version(str(g).strip())
    return x.upper().startswith("ENSG")


def _ensembl_fraction(genes: List[str], sample_n: int = 800) -> float:
    n = min(int(sample_n), len(genes))
    if n <= 0:
        return 0.0
    hits = sum(1 for i in range(n) if _is_ensembl_row(str(genes[i])))
    return hits / float(n)


def _map_gene_to_symbol(g: str, cache: Dict[str, str]) -> str | None:
    g = str(g).strip()
    if not g:
        return None
    if _is_ensembl_row(g):
        ens = _strip_ensembl_version(g)
        sym = (cache.get(ens) or "").strip()
        if not sym:
            return None
    else:
        sym = g
    sym_norm = str(normalize_gene_symbol_series(pd.Series([sym])).iloc[0])
    return sym_norm if sym_norm else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap Ensembl gene IDs in wide expression CSV to HGNC symbols (sum duplicates).")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument(
        "--cache-json",
        type=Path,
        default=None,
        help="Ensembl→symbol 缓存（默认：output-csv 同目录 ensembl_symbol_cache.json）",
    )
    ap.add_argument("--min-ensembl-fraction", type=float, default=0.25)
    ap.add_argument("--force", action="store_true", help="即使 Ensembl 抽检占比低也强制映射 ENSG 行")
    ap.add_argument("--gene-chunk-rows", type=int, default=384, help="每次读入的基因行数（列多时适当减小省内存）")
    ap.add_argument("--ensembl-species", type=str, default="homo_sapiens")
    ap.add_argument("--api-sleep-sec", type=float, default=0.05)
    ap.add_argument("--lookup-batch-size", type=int, default=200)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-if-newer",
        action="store_true",
        help="若 output 已存在且修改时间新于 input，则跳过（不重跑 API）",
    )
    args = ap.parse_args()

    inp = Path(args.input_csv).resolve()
    outp = Path(args.output_csv).resolve()
    if not inp.is_file():
        raise SystemExit(f"missing input: {inp}")

    if bool(args.skip_if_newer) and outp.is_file():
        if outp.stat().st_mtime >= inp.stat().st_mtime:
            print(f"[remap-csv] skip: {outp} is newer than input", flush=True)
            return

    cache_path = Path(args.cache_json).resolve() if args.cache_json else outp.parent / "ensembl_symbol_cache.json"
    cache: Dict[str, str] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    head = pd.read_csv(inp, nrows=0)
    if head.shape[1] < 2:
        raise SystemExit("CSV 至少需要两列：基因名 + 至少一个细胞/样本")
    gene_col = str(head.columns[0])
    cell_cols = [str(c) for c in head.columns[1:]]

    sample_genes = pd.read_csv(inp, usecols=[gene_col], nrows=800).iloc[:, 0].astype(str).tolist()
    frac = _ensembl_fraction(sample_genes)
    print(f"[remap-csv] sample Ensembl-like row fraction ≈ {frac:.3f}", flush=True)
    if frac < float(args.min_ensembl_fraction) and not bool(args.force):
        print("[remap-csv] skip: matrix does not look Ensembl-heavy (use --force to remap anyway).", flush=True)
        return

    unique_ens: set[str] = set()
    for chunk in pd.read_csv(inp, usecols=[gene_col], chunksize=50_000):
        for g in chunk[gene_col].astype(str):
            if _is_ensembl_row(g):
                unique_ens.add(_strip_ensembl_version(g))
    pending = sorted(e for e in unique_ens if e not in cache)
    batch_size = max(1, min(int(args.lookup_batch_size), 1000))
    print(
        f"[remap-csv] unique Ensembl in matrix: {len(unique_ens)} | cache hits: {len(unique_ens) - len(pending)} | pending API: {len(pending)}",
        flush=True,
    )

    if pending and not args.dry_run:
        for bi, chunk in enumerate(_chunks(pending, batch_size)):
            _ensembl_lookup_id_batch(
                chunk,
                cache=cache,
                sleep_s=float(args.api_sleep_sec),
                species=str(args.ensembl_species),
                request_timeout=float(args.request_timeout),
                max_retries=int(args.max_retries),
            )
            if (bi + 1) % 5 == 0 or (bi + 1) * batch_size >= len(pending):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[remap-csv] cache saved ({min((bi + 1) * batch_size, len(pending))}/{len(pending)})", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    n_cells = len(cell_cols)
    acc: Dict[str, np.ndarray] = {}
    dropped = 0
    non_ensembl = 0
    total_in = 0

    chunk_rows = max(8, int(args.gene_chunk_rows))
    if args.dry_run:
        print(f"[remap-csv] dry-run: would stream-merge into symbols (n_cells={n_cells})", flush=True)
        return

    usecols = [gene_col] + cell_cols
    for chunk in pd.read_csv(inp, usecols=usecols, chunksize=chunk_rows):
        chunk[gene_col] = chunk[gene_col].astype(str).str.strip()
        mat = chunk.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        genes = chunk[gene_col].tolist()
        total_in += len(genes)
        for i, g in enumerate(genes):
            if not _is_ensembl_row(str(g)):
                non_ensembl += 1
            sym = _map_gene_to_symbol(str(g), cache)
            if sym is None:
                dropped += 1
                continue
            row = acc.get(sym)
            if row is None:
                row = np.zeros(n_cells, dtype=np.float32)
                acc[sym] = row
            row += mat[i]

    n_out = len(acc)
    print(
        f"[remap-csv] rows_in={total_in} symbols_out={n_out} cells={n_cells} "
        f"dropped_unmapped={dropped} non_ensembl_row_events={non_ensembl}",
        flush=True,
    )
    if n_out < 1:
        raise SystemExit("[remap-csv] no symbols after mapping; check API / cache / species.")

    syms_sorted = sorted(acc.keys())
    big = np.vstack([acc[s] for s in syms_sorted]).astype(np.float32, copy=False)
    del acc
    gc.collect()

    out_df = pd.DataFrame(big, columns=cell_cols)
    out_df.insert(0, gene_col, syms_sorted)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(outp, index=False)
    print(f"[remap-csv] wrote {outp}", flush=True)

    mf = outp.parent / f"{outp.stem}_ensembl_remap_manifest.json"
    mf.write_text(
        json.dumps(
            {
                "input_csv": str(inp),
                "output_csv": str(outp),
                "gene_col": gene_col,
                "n_genes_in": int(total_in),
                "n_symbols_out": int(n_out),
                "n_cells": int(n_cells),
                "dropped_unmapped_rows": int(dropped),
                "ensembl_symbol_cache": str(cache_path),
                "species": str(args.ensembl_species),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[remap-csv] manifest -> {mf}", flush=True)


if __name__ == "__main__":
    main()
