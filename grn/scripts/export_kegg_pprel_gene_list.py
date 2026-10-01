"""从 KEGG PPrel 边表收集全部出现过的基因 symbol，写出单列 CSV，供 case study --de-cartesian-csv + --de-use-all-rows。"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path


def _ts_log(msg: str) -> None:
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [export-kegg-genes] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="导出 KEGG PPrel 边表中的全部基因 symbol（单列 gene）")
    p.add_argument(
        "--kegg-csv",
        type=Path,
        default=Path("data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/network/grn/KEGG_hsa_PPrel_all_symbols_gene_list.csv"),
    )
    args = p.parse_args()
    _ts_log("start (use PYTHONUNBUFFERED=1 for live logs under conda)")
    t0 = time.monotonic()
    if not args.kegg_csv.is_file():
        raise SystemExit(f"缺少 KEGG 表: {args.kegg_csv}")

    syms: set[str] = set()
    with args.kegg_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            a = (row.get("source_symbol") or "").strip()
            b = (row.get("target_symbol") or "").strip()
            if a:
                syms.add(a)
            if b:
                syms.add(b)
    ordered = sorted(syms)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["gene"])
        for g in ordered:
            w.writerow([g])
    n = len(ordered)
    _ts_log(f"done in {time.monotonic() - t0:.2f}s: n={n} directed_pairs=n*(n-1)={n * (n - 1)} -> {args.out_csv}")


if __name__ == "__main__":
    main()
