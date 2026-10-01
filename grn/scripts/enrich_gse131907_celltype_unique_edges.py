#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GO/KEGG enrichment for step06 type-unique edge gene sets.

Default: local clusterProfiler (R) — no g:Profiler network required for GO.
Optional fallback: g:Profiler REST (--backend gprofiler).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

from grn.data.gse131907_celltype_paths import CELL_TYPES, gse131907_celltype_layout, grn_root_from_here


def _resolve_rscript(cli: str | None) -> Path:
    if cli:
        p = Path(cli)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Rscript not found: {cli}")
    env = os.environ.get("RSCRIPT", "").strip()
    if env and Path(env).is_file():
        return Path(env)
    for cand in (
        r"C:\Program Files\R\R-4.5.2\bin\x64\Rscript.exe",
        r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.4.2\bin\x64\Rscript.exe",
    ):
        if Path(cand).is_file():
            return Path(cand)
    return Path("Rscript")


def _genes_from_unique_csv(path: Path) -> List[str]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    genes: Set[str] = set()
    for col in ("TF", "Target"):
        if col in df.columns:
            genes.update(df[col].astype(str).tolist())
    return sorted(g for g in genes if g and g.lower() != "nan")


def _gprofiler_enrich(genes: List[str], *, organism: str = "hsapiens") -> Dict[str, Any]:
    payload = {
        "organism": organism,
        "query": genes,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG"],
        "user_threshold": 0.05,
        "all_results": False,
        "ordered": False,
        "no_iea": False,
        "domain_scope": "annotated",
    }
    req = urllib.request.Request(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "project_GRN-step06/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _run_clusterprofiler(type_out: Path, genes: List[str], rscript: Path, r_script: Path) -> bool:
    type_out.mkdir(parents=True, exist_ok=True)
    gene_file = type_out / "r_enrich_gene_symbols.txt"
    gene_file.write_text("\n".join(genes) + "\n", encoding="utf-8")
    # Also write as network-all so metastasis R script picks it up as query
    (type_out / "r_network_all_gene_symbols.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    cmd = [str(rscript), str(r_script), str(type_out)]
    print(f"[celltype-enrich] clusterProfiler: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as ex:
        print(f"[celltype-enrich] warning: clusterProfiler failed: {ex}")
        return False
    return (type_out / "enrichment_summary.json").is_file() or (type_out / "enrich_R_GO_BP.csv").is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich GSE131907 step06 unique edge gene sets")
    ap.add_argument("--backend", choices=("clusterprofiler", "gprofiler", "auto"), default="clusterprofiler")
    ap.add_argument("--organism", default="hsapiens")
    ap.add_argument("--rscript", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-genes", type=int, default=5)
    args = ap.parse_args()

    layout = gse131907_celltype_layout()
    in_dir = layout["unique_edges_dir"]
    out_dir = layout["enrichment_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    r_script = grn_root_from_here() / "grn" / "r" / "metastasis_case_study_enrich_clusterprofiler.R"
    rscript = _resolve_rscript(args.rscript)

    n_ok = 0
    n_fail = 0
    for tid, label in CELL_TYPES:
        uniq_csv = in_dir / f"unique_edges_{tid}.csv"
        if not uniq_csv.is_file():
            print(f"[celltype-enrich] skip missing {uniq_csv}")
            continue
        genes = _genes_from_unique_csv(uniq_csv)
        type_out = out_dir / tid
        summary_json = type_out / "enrichment_summary.json"
        gp_json = out_dir / f"enrichment_{tid}.json"
        gp_terms = out_dir / f"enrichment_{tid}_terms.csv"

        if not args.force and (summary_json.is_file() or (gp_json.is_file() and gp_terms.is_file())):
            print(f"[celltype-enrich] skip existing {tid}")
            n_ok += 1
            continue
        if len(genes) < int(args.min_genes):
            print(f"[celltype-enrich] {tid}: too few genes ({len(genes)}), skip")
            continue

        print(f"[celltype-enrich] {tid} ({label}): n_genes={len(genes)}")
        backend = args.backend
        done = False

        if backend in ("clusterprofiler", "auto"):
            if not r_script.is_file():
                print(f"[celltype-enrich] warning: missing R script {r_script}")
            else:
                done = _run_clusterprofiler(type_out, genes, rscript, r_script)

        if not done and backend in ("gprofiler", "auto"):
            try:
                result = _gprofiler_enrich(genes, organism=str(args.organism))
                gp_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                rows = result.get("result") or []
                if rows:
                    pd.DataFrame(rows).to_csv(gp_terms, index=False, encoding="utf-8-sig")
                else:
                    pd.DataFrame(columns=["source", "term_id", "term_name", "p_value"]).to_csv(
                        gp_terms, index=False, encoding="utf-8-sig"
                    )
                print(f"[celltype-enrich] g:Profiler wrote {gp_json}")
                done = True
            except (urllib.error.URLError, TimeoutError, OSError) as ex:
                print(f"[celltype-enrich] warning: g:Profiler failed for {tid}: {ex}")

        if done:
            n_ok += 1
        else:
            n_fail += 1
            print(
                f"[celltype-enrich] {tid}: enrichment unavailable "
                "(install clusterProfiler locally, or retry when network can reach g:Profiler)"
            )

    print(f"[celltype-enrich] done: ok={n_ok}, failed={n_fail} -> {out_dir}")
    # Do not fail the bat pipeline solely because enrichment APIs are unreachable.
    raise SystemExit(0)


if __name__ == "__main__":
    main()
