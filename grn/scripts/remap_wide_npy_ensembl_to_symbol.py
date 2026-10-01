"""
将宽大表达 npy 的基因行从 Ensembl ID（ENSG…）映射为 HGNC symbol，并按 symbol 合并重复行（表达值按行求和）。

用于 pancancer 等矩阵首列为 Ensembl、而 GRN/CDS/PPI 使用 symbol 时交集过小的问题。
依赖 Ensembl REST `/lookup/id`（与 prepare_tcga_luad_aligned 相同逻辑），结果写入本地 cache JSON 可重复运行。

用法：
  python -m grn.scripts.remap_wide_npy_ensembl_to_symbol ^
    --genes-npy data/exp/pancancer_scdata_step5/npy/GSE156632_genes.npy ^
    --values-npy data/exp/pancancer_scdata_step5/npy/GSE156632_values.npy ^
    --cache-json data/exp/pancancer_scdata_step5/npy/ensembl_symbol_cache.json

默认就地覆盖 *_genes.npy / *_values.npy（可先备份）。非 Ensembl 为主的矩阵会自动跳过（exit 0）。

Windows：写入前会将表达矩阵整份读入内存并关闭 memmap，否则覆盖原 *_values.npy 可能 WinError 5。
临时文件名为 *_remaptemp.npy（不可用 *.npy.tmp，否则 np.save 会再追加 .npy）。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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


def _ensembl_fraction(genes: np.ndarray, sample_n: int = 800) -> float:
    n = min(int(sample_n), len(genes))
    if n <= 0:
        return 0.0
    hits = sum(1 for i in range(n) if _is_ensembl_row(str(genes[i])))
    return hits / float(n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap Ensembl gene rows in wide expression npy to HGNC symbols.")
    ap.add_argument("--genes-npy", type=Path, required=True)
    ap.add_argument("--values-npy", type=Path, required=True)
    ap.add_argument(
        "--cache-json",
        type=Path,
        default=None,
        help="读取/追加 Ensembl→symbol 缓存（默认与 genes.npy 同目录 ensembl_symbol_cache.json）。",
    )
    ap.add_argument(
        "--min-ensembl-fraction",
        type=float,
        default=0.25,
        help="抽检前 800 行中 ENSG 占比低于该阈值则跳过（默认 0.25）。",
    )
    ap.add_argument("--ensembl-species", type=str, default="homo_sapiens")
    ap.add_argument("--api-sleep-sec", type=float, default=0.05)
    ap.add_argument("--lookup-batch-size", type=int, default=200)
    ap.add_argument("--request-timeout", type=float, default=600.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="即使 Ensembl 占比不高也尝试映射 ENSG 行")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = ap.parse_args()

    gp = Path(args.genes_npy).resolve()
    vp = Path(args.values_npy).resolve()
    print(f"[remap-ensembl] running script: {Path(__file__).resolve()}", flush=True)
    if not gp.is_file() or not vp.is_file():
        raise SystemExit(f"missing genes or values npy: {gp} / {vp}")

    cache_path = Path(args.cache_json).resolve() if args.cache_json else gp.parent / "ensembl_symbol_cache.json"
    cache: Dict[str, str] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    g_arr = np.load(str(gp), allow_pickle=True)
    raw = np.load(str(vp), mmap_mode="r")
    if raw.shape[0] != len(g_arr):
        raise SystemExit(
            f"genes ({len(g_arr)}) vs values rows ({raw.shape[0]}) mismatch。\n"
            "常见原因：上次 remap 先替换了 *_values.npy，写入 *_genes.npy 失败，导致行列不一致。\n"
            "修复：请从原始 CSV 重新生成该 GSE 的 npy（删旧文件后跑）：\n"
            "  Recreate npy from the GSE wide CSV, then rerun step01_单细胞多方法基准_LUAD_TCGA_方法比对.bat (pancancer A–B) or grn.scripts.wide_expression_csv_to_npy.\n"
            "然后再运行 remap / train。"
        )

    frac = _ensembl_fraction(g_arr)
    print(f"[remap-ensembl] sample Ensembl-like row fraction ≈ {frac:.3f}")
    if frac < float(args.min_ensembl_fraction) and not bool(args.force):
        print("[remap-ensembl] skip: matrix does not look Ensembl-heavy (use --force to remap anyway).")
        del raw
        gc.collect()
        return

    raw_genes = [str(x) for x in g_arr.tolist()]
    unique_ens = sorted({_strip_ensembl_version(x) for x in raw_genes if _is_ensembl_row(x)})
    pending = [e for e in unique_ens if e not in cache]
    batch_size = max(1, min(int(args.lookup_batch_size), 1000))
    print(f"[remap-ensembl] unique Ensembl IDs in matrix: {len(unique_ens)} | cache hits: {len(unique_ens) - len(pending)} | pending API: {len(pending)}")

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
                print(f"[remap-ensembl] cache saved ({min((bi + 1) * batch_size, len(pending))}/{len(pending)} pending batches progressed)")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # row_index -> target symbol (normalized)
    groups: Dict[str, List[int]] = {}
    dropped = 0
    non_ens_named = 0
    for i, g in enumerate(raw_genes):
        if _is_ensembl_row(g):
            ens = _strip_ensembl_version(g)
            sym = (cache.get(ens) or "").strip()
            if not sym:
                dropped += 1
                continue
        else:
            sym = str(g).strip()
            non_ens_named += 1
        sym_norm = str(normalize_gene_symbol_series(pd.Series([sym])).iloc[0])
        if not sym_norm:
            dropped += 1
            continue
        groups.setdefault(sym_norm, []).append(i)

    symbols_sorted = sorted(groups.keys())
    n_out = len(symbols_sorted)
    n_cells = int(raw.shape[1])
    print(
        f"[remap-ensembl] rows_in={len(raw_genes)} rows_out={n_out} cells={n_cells} "
        f"dropped_unmapped={dropped} non_ensembl_rows_kept_as_symbol={non_ens_named}"
    )

    if args.dry_run:
        del raw
        gc.collect()
        return

    if n_out < 1:
        raise SystemExit("[remap-ensembl] no symbols after mapping; check API cache / species.")

    # Windows：mmap 打开的 *_values.npy 未关闭时 os.replace 覆盖同路径会 WinError 5。
    print("[remap-ensembl] materializing values into RAM and closing mmap (required for in-place replace on Windows)…", flush=True)
    mat = np.array(raw, dtype=np.float32, copy=True)
    del raw
    gc.collect()

    out_vals = np.zeros((n_out, n_cells), dtype=np.float32)
    for gi, sym in enumerate(symbols_sorted):
        idxs = groups[sym]
        out_vals[gi] = mat[idxs, :].sum(axis=0)

    # 临时文件必须仍以 .npy 结尾：numpy.save 遇其它后缀会自动再追加 .npy，
    # 会得到 *.npy.tmp.npy 导致 os.replace 找不到 *.npy.tmp（WinError 2）。
    tmp_v = vp.parent / f"{vp.stem}_remaptemp.npy"
    tmp_g = gp.parent / f"{gp.stem}_remaptemp.npy"
    np.save(str(tmp_g), np.array(symbols_sorted, dtype=object))
    if not tmp_g.is_file():
        raise SystemExit(f"[remap-ensembl] temp genes file missing after np.save: {tmp_g}")
    mm = np.lib.format.open_memmap(tmp_v, mode="w+", dtype=np.float32, shape=out_vals.shape)
    mm[:] = out_vals
    mm.flush()
    del mm
    if not tmp_v.is_file():
        raise SystemExit(f"[remap-ensembl] temp values file missing after memmap write: {tmp_v}")

    os.replace(str(tmp_v), str(vp))
    os.replace(str(tmp_g), str(gp))

    tag = gp.name[: -len("_genes.npy")] if gp.name.endswith("_genes.npy") else gp.stem
    legacy_manifest = gp.parent / f"{tag}_filter_manifest.json"
    if legacy_manifest.is_file():
        legacy_manifest.unlink()

    manifest = {
        "genes_npy": str(gp),
        "values_npy": str(vp),
        "n_genes_in": int(len(raw_genes)),
        "n_genes_out": int(n_out),
        "n_cells": int(n_cells),
        "dropped_unmapped_rows": int(dropped),
        "ensembl_symbol_cache": str(cache_path),
        "species": str(args.ensembl_species),
    }
    mf = gp.parent / f"{tag}_ensembl_remap_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[remap-ensembl] done. manifest -> {mf}")


if __name__ == "__main__":
    main()
