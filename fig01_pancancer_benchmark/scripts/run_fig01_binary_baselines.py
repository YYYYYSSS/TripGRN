#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.1 launchers for DeepTGI / scMGATGRN / GENELink.

Protocol: shared ``cv_splits_10fold.json`` (5:2:3); genes = current GRN-mapped set
(no random / no transcriptome fill); full CV cells (no subsample).
scMGATGRN / GENELink zero-pad shorter splits so GAT ``input_dim`` matches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


METHODS = {
    "deeptgi": {
        "run": "grn.scripts.run_deeptgi_pancancer_10cv",
        "summarize": "grn.scripts.summarize_deeptgi_pancancer_10cv",
        "label": "DeepTGI",
        "extra": ["--max-genes", "0", "--batch-size-cap", "512"],
        "project": ("DEEPTGI_ROOT", ("project_DeepTGI",)),
        "cli_rel": Path("program") / "run_unseen_cells_cli.py",
    },
    "scmgatgrn": {
        "run": "grn.scripts.run_scmgatgrn_pancancer_10cv",
        "summarize": "grn.scripts.summarize_scmgatgrn_pancancer_10cv",
        "label": "scMGATGRN",
        "extra": ["--max-genes", "0", "--batch-size-cap", "256"],
        "project": ("SCMGATGRN_CODE", ("project_scMGATGRN", "scMGATGRN-main", "Code")),
        "cli_rel": None,
    },
    "genelink": {
        "run": "grn.scripts.run_genelink_pancancer_10cv",
        "summarize": "grn.scripts.summarize_genelink_pancancer_10cv",
        "label": "GENELink",
        "extra": ["--max-genes", "0", "--batch-size-cap", "256"],
        "project": ("GENELINK_CODE", ("project_GENELink", "Code")),
        "cli_rel": None,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dorothea_csv(root: Path) -> Path:
    candidates = [
        root / "shared" / "data" / "network" / "grn" / "DoRothEA_human_ABC_activation_inhibition_with_symbols.csv",
        root / "shared" / "data" / "network" / "grn" / "dorothea_abc_activation_inhibition.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve()


def _ensure_repo_on_path(root: Path) -> None:
    """Launcher is under fig01_*/scripts/; ensure repo root is importable for ``grn``."""
    root_s = str(root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _apply_network_env(root: Path, network: str) -> Path:
    net = str(network).strip().lower()
    _ensure_repo_on_path(root)
    if net in ("kegg", "kegg_pprel"):
        os.environ.pop("STEP01_EXPERIMENTS_NAME", None)
        os.environ.pop("STEP01_GRN_CSV", None)
        os.environ["STEP01_EXPERIMENTS_NAME"] = "kegg_epithelial_10cv"
        from grn.data.pancancer_paths import step01_grn_csv

        return step01_grn_csv(root)
    if net in ("dorothea", "do", "dorothea_abc"):
        csv = _dorothea_csv(root)
        os.environ["STEP01_EXPERIMENTS_NAME"] = "dorothea_abc_epithelial_10cv"
        os.environ["STEP01_GRN_CSV"] = str(csv)
        return csv
    raise SystemExit(f"Unknown network: {network} (use kegg or dorothea)")


def _resolve_methods(method_arg: str) -> list[str]:
    raw = str(method_arg).strip().lower()
    if raw == "all":
        return list(METHODS)
    aliases = {"tgi": "deeptgi", "scmgat": "scmgatgrn", "gene": "genelink"}
    key = aliases.get(raw, raw)
    if key not in METHODS:
        raise SystemExit(f"Unknown method: {method_arg} (DeepTGI|scMGATGRN|GENELink|ALL)")
    return [key]


def cmd_smoke(method: str, network: str) -> int:
    root = _repo_root()
    grn_csv = _apply_network_env(root, network)
    from grn.data.pancancer_paths import (
        EXCLUDED_GSE_IDS,
        PANCANCER_GSE_IDS,
        cv_splits_path,
        gse_npy_triplet,
        resolve_external_project,
        scRNA_layout,
        step01_grn_csv,
    )

    layout = scRNA_layout(root)
    checks = {
        "network": network,
        "grn_csv": str(step01_grn_csv(root)),
        "expected": str(grn_csv),
        "epithelial": str(layout["epithelial"]),
        "experiments": str(layout["experiments_root"]),
        "gene_policy": "grn_mapped",
        "cell_policy": "full_cv_splits (scMGAT/GENELink pad-to-max; no downsample)",
        "max_genes": 0,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"PANCANCER_GSE_IDS ({len(PANCANCER_GSE_IDS)}): {PANCANCER_GSE_IDS}")
    print(f"EXCLUDED_GSE_IDS: {EXCLUDED_GSE_IDS}")
    assert "GSE145825" not in PANCANCER_GSE_IDS
    if not Path(checks["grn_csv"]).is_file():
        raise FileNotFoundError(f"Missing GRN: {checks['grn_csv']}")
    want_name = "kegg" if str(network).lower().startswith("kegg") else "dorothea"
    if want_name not in Path(checks["grn_csv"]).name.lower() and want_name != "kegg":
        # kegg file is KEGG_hsa_...
        pass
    missing = []
    present_gses = []
    skipped_gses = []
    for gse in PANCANCER_GSE_IDS:
        v, g, c = gse_npy_triplet("epithelial", gse, root)
        cv = cv_splits_path(gse, root)
        # Partial packs (e.g. Machine B split) may only ship a subset of GSEs.
        if not v.is_file() and not cv.is_file():
            skipped_gses.append(gse)
            continue
        present_gses.append(gse)
        for p in (v, g, c, cv):
            if not p.is_file():
                missing.append(str(p))
        if cv.is_file():
            payload = json.loads(cv.read_text(encoding="utf-8"))
            if list(payload.get("sample_split", [])) != [5, 2, 3]:
                raise ValueError(f"{gse} sample_split != [5,2,3]")
    if skipped_gses:
        print(f"[fig01-binary-smoke] skip missing GSE dirs ({len(skipped_gses)}): {skipped_gses}")
    if not present_gses:
        raise FileNotFoundError("No epithelial GSE expression/cv files found under pack.")
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print(f"[fig01-binary-smoke] validated GSEs ({len(present_gses)}): {present_gses}")

    for key in _resolve_methods(method):
        spec = METHODS[key]
        env_key, rel = spec["project"]
        found = resolve_external_project(env_key, *rel)
        print(f"[{spec['label']}] project={found}")
        if found is None:
            raise FileNotFoundError(f"Missing {spec['label']} at E:\\project\\{'/'.join(rel)}")
        if spec["cli_rel"] is not None and not (found / spec["cli_rel"]).is_file():
            raise FileNotFoundError(f"Missing {found / spec['cli_rel']}")
    print("[fig01-binary-smoke] OK: GRN-mapped genes, full CV cells, shared splits, baseline repos")
    return 0


def cmd_run(method: str, network: str, gse: str, fold: str, force: bool) -> int:
    root = _repo_root()
    grn_csv = _apply_network_env(root, network)
    keys = _resolve_methods(method)
    rc = 0
    for key in keys:
        spec = METHODS[key]
        if not force:
            n_done, n_total, nxt = _fold_progress(root, key)
            nxt_s = "COMPLETE" if nxt is None else f"{nxt[0]} fold {nxt[1]:02d}"
            print(
                f"[fig01-{spec['label']}] checkpoint {network}: {n_done}/{n_total}  next={nxt_s}",
                flush=True,
            )
            if n_done >= n_total:
                print(f"[fig01-{spec['label']}] skip run (all folds done)", flush=True)
                continue
        env = os.environ.copy()
        env["STEP01_GRN_CSV"] = str(grn_csv)
        env["STEP01_EXPERIMENTS_NAME"] = os.environ["STEP01_EXPERIMENTS_NAME"]
        cmd = [
            sys.executable,
            "-u",
            "-m",
            spec["run"],
            "--gse",
            gse,
            "--fold",
            fold,
            *spec["extra"],
        ]
        if force:
            cmd.append("--force")
        print(f"[fig01-{spec['label']}] GRN {grn_csv}", flush=True)
        print(f"[fig01-{spec['label']}] genes=GRN-mapped  cells=full_cv  RUN", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(root), env=env)
        if rc != 0:
            return rc
    return rc


def _fold_progress(root: Path, method_key: str) -> tuple[int, int, tuple[str, int] | None]:
    """Return (n_done, n_total, next_missing_or_None). Env must already select the network."""
    from grn.data.pancancer_paths import PANCANCER_GSE_IDS, method_fold_dir

    label = METHODS[method_key]["label"]
    n_done = 0
    n_total = len(PANCANCER_GSE_IDS) * 10
    nxt: tuple[str, int] | None = None
    for gse in PANCANCER_GSE_IDS:
        for fold_idx in range(1, 11):
            run_dir = method_fold_dir(label, gse, fold_idx, root)
            ok = (run_dir / "unseen_samples_test_metrics.json").is_file()
            if ok:
                n_done += 1
            elif nxt is None:
                nxt = (gse, fold_idx)
    return n_done, n_total, nxt


def cmd_status(method: str, network: str) -> int:
    root = _repo_root()
    nets = ["kegg", "dorothea"] if str(network).strip().lower() == "all" else [network]
    keys = _resolve_methods(method)
    print(f"{'method':<12} {'network':<10} {'done':>6}  next")
    for net in nets:
        _apply_network_env(root, net)
        for key in keys:
            n_done, n_total, nxt = _fold_progress(root, key)
            nxt_s = "COMPLETE" if nxt is None else f"{nxt[0]} fold {nxt[1]:02d}"
            print(f"{METHODS[key]['label']:<12} {net:<10} {n_done:3d}/{n_total:<2d}  {nxt_s}")
    return 0


def cmd_check(method: str, network: str) -> int:
    """Exit 0 if this method×network already has all 90 folds; else 1 (need run)."""
    root = _repo_root()
    _apply_network_env(root, network)
    keys = _resolve_methods(method)
    all_done = True
    for key in keys:
        n_done, n_total, nxt = _fold_progress(root, key)
        extra = "" if nxt is None else f" next={nxt[0]} fold {nxt[1]:02d}"
        print(f"[check] {METHODS[key]['label']} {network}: {n_done}/{n_total}{extra}", flush=True)
        if n_done < n_total:
            all_done = False
    return 0 if all_done else 1


def cmd_summarize(method: str, network: str, gse: str) -> int:
    root = _repo_root()
    grn_csv = _apply_network_env(root, network)
    keys = _resolve_methods(method)
    rc = 0
    for key in keys:
        spec = METHODS[key]
        env = os.environ.copy()
        env["STEP01_GRN_CSV"] = str(grn_csv)
        env["STEP01_EXPERIMENTS_NAME"] = os.environ["STEP01_EXPERIMENTS_NAME"]
        cmd = [sys.executable, "-u", "-m", spec["summarize"], "--gse", gse]
        print(f"[fig01-{spec['label']}] RUN", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(root), env=env)
        if rc != 0:
            return rc
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.1 DeepTGI / scMGATGRN / GENELink helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--method", default="DeepTGI", help="DeepTGI | scMGATGRN | GENELink | ALL")
        p.add_argument("--network", default="kegg", help="kegg | dorothea")

    p_smoke = sub.add_parser("smoke")
    _add_common(p_smoke)

    p_run = sub.add_parser("run")
    _add_common(p_run)
    p_run.add_argument("--gse", default="GSE156632")
    p_run.add_argument("--fold", default="1")
    p_run.add_argument("--force", action="store_true")

    p_sum = sub.add_parser("summarize")
    _add_common(p_sum)
    p_sum.add_argument("--gse", default="ALL")

    p_st = sub.add_parser("status", help="Print fold completion (resume checkpoint)")
    p_st.add_argument("--method", default="ALL")
    p_st.add_argument("--network", default="all", help="kegg | dorothea | all")

    p_ck = sub.add_parser("check", help="Exit 0 if method×network is fully done (90/90)")
    _add_common(p_ck)

    args = ap.parse_args()
    if args.cmd == "smoke":
        raise SystemExit(cmd_smoke(args.method, args.network))
    if args.cmd == "run":
        raise SystemExit(cmd_run(args.method, args.network, args.gse, args.fold, args.force))
    if args.cmd == "summarize":
        raise SystemExit(cmd_summarize(args.method, args.network, args.gse))
    if args.cmd == "status":
        raise SystemExit(cmd_status(args.method, args.network))
    if args.cmd == "check":
        raise SystemExit(cmd_check(args.method, args.network))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
