# -*- coding: utf-8 -*-
"""
在 pancancer 宽大表达 CSV 上运行 DeepFGRN（project_DeepFGRN）的「未见细胞」流程。

表达谱处理边界（重要）：
  - 输入仅需「宽大 CSV」原始数值矩阵（首列基因名，其余列为细胞/样本）；读入与后续
    log2(x+1)、训练集 MinMax、细胞 5:2:3 划分、子采样对齐、构图嵌入、CorrResNet 训练，
    一律使用 project_DeepFGRN 内的模块（gse253718_kegg_utils、dggan_embedding_param、
    corrresnet_pred_224、three_utils_2_single），不经过 project_GRN 的 harmonize / npy /
    train_unseen 等自有流水线。
  - 本仓库脚本只做：解析参数、路径、切换工作目录与 sys.path、把结果写到 --output-base。

细胞列 5:2:3（可接共享 cv_splits）+ log2(x+1)/训练集 MinMax + CorrResNet。
默认另对基因对做 stratified held-out（约 64/16/20，对齐 TripGRN）；`--same-pairs-all-splits` 可回退为
「同对复用、只换细胞」的旧协议。

依赖：需在包含 TensorFlow 的环境中运行（如 conda env project_GRN），且存在同级目录 project_DeepFGRN。

用法：
  python -m grn.scripts.run_deepfgrn_pancancer_cell_unseen ^
    --exp-csv E:\\project\\project_GRN\\data\\exp\\pancancer\\GSE148071.csv ^
    --kegg-csv E:\\project\\project_GRN\\data\\network\\grn\\KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv ^
    --output-base E:\\project\\project_GRN\\experiments\\deepfgrn_baseline_pancancer ^
    --gse-id GSE148071
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from grn.evaluation.confusion_matrix_io import export_confusion_matrix_artifacts
from grn.scripts.deepfgrn_step01_metrics import deepfgrn_metrics_to_step01_json
from grn.data.pancancer_cv import get_fold_split, load_cv_splits
from grn.utils.runtime import now_mono, write_runtime_json
from grn.scripts.pancancer_step01_paths import (
    find_latest_fgrn_run_with_ckpt,
    load_run_meta,
    npy_expression_paths,
    resolve_exp_csv,
    resolve_kegg_csv,
    step01_layout,
)

FGRN_CLASS_NAMES = ("No regulation", "Repression", "Activation")


def _sample_negatives_from_positives(
    positive_edges: List[Tuple[int, int, int]],
    num_negative: int,
    rng,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """Negatives from the positive-edge gene universe only (matched / TF-constrained).

    - source ``i`` ∈ positive sources (TFs that appear as regulators in gold edges)
    - target ``j`` ∈ genes that appear in any positive edge
    - ``(i, j)`` not a gold positive edge

    This avoids easy genome-wide negatives whose source is never a TF.
    """
    import numpy as np

    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(int(rng))

    positive_set = {(int(i), int(j)) for i, j, _ in positive_edges}
    sources = sorted({int(i) for i, _, _ in positive_edges})
    genes = sorted({int(i) for i, _, _ in positive_edges} | {int(j) for _, j, _ in positive_edges})
    if len(sources) < 1 or len(genes) < 2:
        raise RuntimeError(
            f"cannot sample matched negatives: n_sources={len(sources)} n_genes={len(genes)}"
        )

    neg_positions: List[Tuple[int, int]] = []
    neg_set: set = set()
    attempts = 0
    max_attempts = max(int(num_negative) * 500, 2_000_000)
    while len(neg_positions) < int(num_negative) and attempts < max_attempts:
        attempts += 1
        i = int(sources[int(rng.randint(0, len(sources)))])
        j = int(genes[int(rng.randint(0, len(genes)))])
        if i == j:
            continue
        if (i, j) in positive_set or (i, j) in neg_set:
            continue
        neg_set.add((i, j))
        neg_positions.append((i, j))
    if len(neg_positions) < int(num_negative):
        raise RuntimeError(
            f"无法从正例基因宇宙采满负例（得到 {len(neg_positions)} / {num_negative}；"
            f"sources={len(sources)}, genes={len(genes)}）。"
        )
    meta = {
        "negative_protocol": "from_positive_universe_tf_source",
        "n_positive_sources": len(sources),
        "n_positive_genes": len(genes),
        "n_negative": len(neg_positions),
        "n_attempts": int(attempts),
    }
    return neg_positions, meta


def _held_out_split_pairs(
    pos2: List[Tuple[int, int]],
    pos1: List[Tuple[int, int]],
    neg_positions: List[Tuple[int, int]],
    *,
    seed: int = 42,
) -> Tuple[
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]],
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]],
    Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]],
    Dict[str, Any],
]:
    """Stratified pair split ≈64/16/20 (same recipe as TripGRN legacy held-out pairs).

    Returns ((pos2,pos1,neg)_train, _val, _test, meta).
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    items: List[Tuple[int, int, int]] = []
    items.extend((2, i, j) for i, j in pos2)
    items.extend((1, i, j) for i, j in pos1)
    items.extend((0, i, j) for i, j in neg_positions)
    if len(items) < 10:
        raise RuntimeError(f"too few pairs to held-out split: n={len(items)}")
    labels = np.asarray([lab for lab, _, _ in items], dtype=np.int64)
    idx = np.arange(len(items))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=int(seed), stratify=labels
    )
    y_tr = labels[tr_idx]
    tr_idx2, va_idx = train_test_split(
        tr_idx, test_size=0.2, random_state=int(seed), stratify=y_tr
    )

    def _unpack(indices) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
        p2: List[Tuple[int, int]] = []
        p1: List[Tuple[int, int]] = []
        n0: List[Tuple[int, int]] = []
        for ii in indices:
            lab, a, b = items[int(ii)]
            if lab == 2:
                p2.append((a, b))
            elif lab == 1:
                p1.append((a, b))
            else:
                n0.append((a, b))
        return p2, p1, n0

    train_t = _unpack(tr_idx2)
    val_t = _unpack(va_idx)
    test_t = _unpack(te_idx)
    meta = {
        "pair_protocol": "held_out_pairs",
        "pair_split_seed": int(seed),
        "pair_split_ratios": "approx_64_16_20_stratified",
        "n_pairs_pool": {
            "activation": len(pos2),
            "repression": len(pos1),
            "noreg": len(neg_positions),
            "total": len(items),
        },
        "n_pairs_train": {
            "activation": len(train_t[0]),
            "repression": len(train_t[1]),
            "noreg": len(train_t[2]),
            "total": len(tr_idx2),
        },
        "n_pairs_val": {
            "activation": len(val_t[0]),
            "repression": len(val_t[1]),
            "noreg": len(val_t[2]),
            "total": len(va_idx),
        },
        "n_pairs_test": {
            "activation": len(test_t[0]),
            "repression": len(test_t[1]),
            "noreg": len(test_t[2]),
            "total": len(te_idx),
        },
    }
    return train_t, val_t, test_t, meta


def _resolve_deepfgrn_root(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.resolve()
    env = os.environ.get("DEEPFGRN_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    grn_root = Path(__file__).resolve().parents[2]
    candidates = [grn_root.parent / "project_DeepFGRN"]
    legacy_txt = grn_root / "LEGACY_ROOT.txt"
    if legacy_txt.is_file():
        try:
            legacy = Path(legacy_txt.read_text(encoding="utf-8").strip())
            candidates.append(legacy.parent / "project_DeepFGRN")
        except Exception:
            pass
    candidates.extend(
        [
            Path(r"E:\project\project_DeepFGRN"),
            Path("/mnt/e/project/project_DeepFGRN"),
            Path("/mnt/e/project/project_GRN/../project_DeepFGRN"),
        ]
    )
    for path in candidates:
        try:
            if path.is_dir():
                return path.resolve()
        except OSError:
            continue
    return candidates[0].resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepFGRN cell-unseen：细胞 5:2:3 划分 + CorrResNet（默认单次划分）。")
    grn_root = Path(__file__).resolve().parents[2]
    layout = step01_layout(grn_root)
    ap.add_argument(
        "--exp-csv",
        type=Path,
        default=None,
        help="宽大表达 CSV（step01: data/exp/pancancer/<GSE>.csv）。--cm-only 时可省略，从 run_meta.json 读取。",
    )
    ap.add_argument(
        "--kegg-csv",
        type=Path,
        default=None,
        help="KEGG 边 CSV（step01 默认 data/network/grn/KEGG_hsa_PPrel_activation_inhibition_with_symbols.csv）。",
    )
    ap.add_argument(
        "--output-base",
        type=Path,
        default=layout["out_fgrn"],
        help="DeepFGRN 输出根目录（step01: experiments/deepfgrn_baseline_pancancer）。",
    )
    ap.add_argument("--gse-id", type=str, required=True)
    ap.add_argument("--deepfgrn-root", type=Path, default=None, help="默认 DEEPFGRN_ROOT 环境变量或 <project_GRN>/../project_DeepFGRN")
    ap.add_argument(
        "--num-negative",
        type=int,
        default=0,
        help="负样本边数。<=0（默认）= 与正例 1:1（activate+inhibit）；"
        "历史 DeepFGRN_GSE253718_FCV 曾用固定 7000。",
    )
    ap.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="外层重复次数：每次用 split_cells_unseen_523(..., fold_idx=k, rng 种子 42+k)。"
        " 默认 1（单次划分）；设为 5 则与原 DeepFGRN_GSE253718_FCV 五折一致。",
    )
    ap.add_argument("--pair-seed", type=int, default=12345, help="Seed for building the fixed positive/negative pair pool.")
    ap.add_argument(
        "--pair-split-seed",
        type=int,
        default=42,
        help="Seed for stratified held-out pair split (default 42; TripGRN-aligned).",
    )
    ap.add_argument(
        "--held-out-pairs",
        dest="held_out_pairs",
        action="store_true",
        help="Held-out gene pairs (~64/16/20) + unseen cells (default ON).",
    )
    ap.add_argument(
        "--same-pairs-all-splits",
        dest="held_out_pairs",
        action="store_false",
        help="Legacy: same (i,j) pairs on train/val/test (only cells change).",
    )
    ap.set_defaults(held_out_pairs=True)
    ap.add_argument(
        "--verbose-training",
        action="store_true",
        help="完整日志：Keras 逐步进度条、模型 summary、dGGAN 详细迭代；默认关闭（仅关键信息与按 epoch 摘要）。",
    )
    ap.add_argument(
        "--cm-only",
        action="store_true",
        help="Skip training; load fold_*/best_model.keras under --run-dir (or latest run) and export test confusion matrix.",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing DeepFGRN run directory (for --cm-only). Default: newest run under output-base/gse-id/.",
    )
    ap.add_argument(
        "--export-test-probs",
        type=Path,
        default=None,
        help="With --cm-only: save test_y_true.npy and test_y_prob.npy here (no retraining).",
    )
    ap.add_argument(
        "--exp-npy-values",
        type=Path,
        default=None,
        help="Epithelial values.npy (step01); overrides --exp-csv when set.",
    )
    ap.add_argument(
        "--exp-npy-genes",
        type=Path,
        default=None,
        help="Epithelial genes.npy (required with --exp-npy-values).",
    )
    ap.add_argument(
        "--cv-splits-json",
        type=Path,
        default=None,
        help="Unified 10-fold CV manifest (step01 epithelial).",
    )
    ap.add_argument(
        "--fold-idx",
        type=int,
        default=0,
        help="1-based fold index when using --cv-splits-json (single fold). Ignored if --fold-list set.",
    )
    ap.add_argument(
        "--fold-list",
        type=str,
        default=None,
        help="Comma-separated 1-based folds (e.g. 1,2,3 or ALL). Requires --cv-splits-json + --output-gse-dir. "
        "Runs all listed folds in one process (shared TF/dGGAN/expression load).",
    )
    ap.add_argument(
        "--output-fold-dir",
        type=Path,
        default=None,
        help="Fixed per-fold output dir (step01: experiments/.../DeepFGRN/GSE/fold_XX).",
    )
    ap.add_argument(
        "--output-gse-dir",
        type=Path,
        default=None,
        help="GSE parent dir; with --fold-list writes fold_XX under this path.",
    )
    ap.add_argument(
        "--keras-batch-size",
        type=int,
        default=0,
        help="CorrResNet batch cap (0 = use DEEPFGRN_BATCH_SIZE env or 128).",
    )
    ap.add_argument(
        "--skip-summary-plots",
        action="store_true",
        help="Skip xlsx/radar/boxplots (step01 per-fold mode).",
    )
    args = ap.parse_args()

    if not args.verbose_training:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    keras_bs = int(args.keras_batch_size)
    if keras_bs > 0:
        os.environ["DEEPFGRN_BATCH_SIZE"] = str(keras_bs)
    elif "DEEPFGRN_BATCH_SIZE" not in os.environ:
        os.environ["DEEPFGRN_BATCH_SIZE"] = "128"

    fold_list_1based: List[int] | None = None
    if args.fold_list is not None:
        fl = str(args.fold_list).strip().upper()
        if fl == "ALL":
            fold_list_1based = list(range(1, 11))
        else:
            fold_list_1based = [int(x.strip()) for x in str(args.fold_list).split(",") if x.strip()]
        if not fold_list_1based or any(i < 1 for i in fold_list_1based):
            raise SystemExit("--fold-list must be ALL or comma-separated integers >= 1")
        if args.cv_splits_json is None or args.output_gse_dir is None:
            raise SystemExit("--fold-list requires --cv-splits-json and --output-gse-dir")
        if args.output_fold_dir is not None:
            raise SystemExit("use either --output-fold-dir (one fold) or --fold-list + --output-gse-dir")

    step01_multi_fold = fold_list_1based is not None
    step01_fold_mode = (args.output_fold_dir is not None and args.cv_splits_json is not None) or step01_multi_fold
    if args.cv_splits_json is not None and not step01_multi_fold and int(args.fold_idx) <= 0:
        raise SystemExit("--cv-splits-json requires --fold-idx >= 1 (or --fold-list)")

    grn_root = Path(__file__).resolve().parents[2]
    out_base = Path(args.output_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    gse_id = str(args.gse_id).strip()

    run_meta: dict | None = None
    if args.cm_only:
        if args.output_fold_dir is not None:
            # step01 recover: reuse fold_XX/best_model.keras and rewrite metrics JSON
            out_run = Path(args.output_fold_dir).resolve()
            run_meta = None
        elif args.run_dir is not None:
            out_run = Path(args.run_dir).resolve()
            run_meta = load_run_meta(out_run)
        else:
            out_run = find_latest_fgrn_run_with_ckpt(out_base, gse_id)
            if out_run is None:
                raise SystemExit(
                    "cm-only: no run with fold_1/best_model.keras under %s/%s"
                    % (out_base, gse_id)
                )
            run_meta = load_run_meta(out_run)
        print("[deepfgrn-pancancer] cm-only using run ->", out_run)
    else:
        out_run = None  # set after run_tag below
        run_meta = None

    if args.deepfgrn_root is not None:
        deep_root = Path(args.deepfgrn_root).resolve()
    elif run_meta and run_meta.get("deepfgrn_root"):
        deep_root = Path(str(run_meta["deepfgrn_root"])).resolve()
    else:
        deep_root = _resolve_deepfgrn_root(None)

    path_exp = Path(args.exp_csv).resolve() if args.exp_csv else None
    npy_values = Path(args.exp_npy_values).resolve() if args.exp_npy_values else None
    npy_genes = Path(args.exp_npy_genes).resolve() if args.exp_npy_genes else None
    if npy_values is not None:
        if npy_genes is None or not npy_genes.is_file():
            raise SystemExit("--exp-npy-genes required with --exp-npy-values")
        path_exp = None
    elif path_exp is None or not path_exp.is_file():
        try:
            path_exp = resolve_exp_csv(gse_id, grn_root, run_meta)
        except FileNotFoundError:
            if args.cm_only or args.exp_csv is None:
                v_p, g_p = npy_expression_paths(gse_id, grn_root)
                if v_p.is_file() and g_p.is_file():
                    path_exp = None  # signal npy load below
                else:
                    raise
            else:
                raise
    path_kegg = Path(args.kegg_csv).resolve() if args.kegg_csv else resolve_kegg_csv(grn_root, run_meta)
    if path_kegg and not path_kegg.is_file():
        raise SystemExit(f"missing KEGG CSV: {path_kegg}")

    if not deep_root.is_dir():
        raise SystemExit(f"project_DeepFGRN not found: {deep_root}（设置 --deepfgrn-root 或 DEEPFGRN_ROOT）")

    os.chdir(str(deep_root))
    if str(deep_root) not in sys.path:
        sys.path.insert(0, str(deep_root))

    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    import corrresnet_pred_224 as corrresnet_pred  # noqa: E402
    import dggan_embedding_param  # noqa: E402
    import gse253718_kegg_utils as ku  # noqa: E402
    import three_utils_2_single  # noqa: E402
    import dggan_config_224 as dggan_config_mod  # noqa: E402

    # Hot path: DeepFGRN's build_data_tensor does per-sample Python hstack (~33k pairs ×
    # train/val/test). That alone can burn 10+ min/fold with GPU/CPU idle. Vectorize.
    def _build_data_tensor_fast(dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, a, b, c, d, e, f):
        del a, b, c, d, e, f  # layout markers unused after concat
        return np.concatenate(
            [dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t],
            axis=-1,
        )

    ku.build_data_tensor = _build_data_tensor_fast  # type: ignore[attr-defined]

    if not args.verbose_training:
        dggan_config_mod.verbose = 0

    NB_CLASSES = 3
    DIM_NET = 224
    N_FOLDS = int(args.n_folds)
    NUM_NOT_REGULATE = int(run_meta.get("num_negative", args.num_negative)) if run_meta else int(args.num_negative)
    PAIR_SEED = int(run_meta.get("pair_seed", args.pair_seed)) if run_meta else int(args.pair_seed)
    PAIR_SPLIT_SEED = int(run_meta.get("pair_split_seed", args.pair_split_seed)) if run_meta else int(args.pair_split_seed)
    # dggan_dir / embedding cache paths are set after pair split (train-only gold graph).
    dggan_dir: Path | None = None
    emb_s_path: Path | None = None
    emb_t_path: Path | None = None

    if args.cm_only:
        assert out_run is not None
    elif step01_multi_fold:
        out_run = Path(args.output_gse_dir).resolve()
        out_run.mkdir(parents=True, exist_ok=True)
    elif step01_fold_mode:
        out_run = Path(args.output_fold_dir).resolve()
        out_run.mkdir(parents=True, exist_ok=True)
    else:
        log_time = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
        run_tag = (
            "%s_cellUnseen_single523_%s" % (gse_id, log_time)
            if N_FOLDS == 1
            else "%s_cellUnseen_%dFCV_%s" % (gse_id, N_FOLDS, log_time)
        )
        out_run = out_base / gse_id / run_tag
        out_run.mkdir(parents=True, exist_ok=True)
        meta_path = out_run / "run_meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "deepfgrn_root": str(deep_root),
                    "exp_csv": str(path_exp) if path_exp is not None else None,
                    "exp_npy_values": str(npy_expression_paths(gse_id, grn_root)[0])
                    if path_exp is None
                    else None,
                    "exp_npy_genes": str(npy_expression_paths(gse_id, grn_root)[1])
                    if path_exp is None
                    else None,
                    "kegg_csv": str(path_kegg),
                    "gse_id": gse_id,
                    "protocol": "DeepFGRN-compatible: 5:2:3 cells, log2+MinMax train-fit, fixed pairs, dggan emb; "
                    "outer_n_folds=%d (fit_5CV name is train+val early-stop only, not dataset CV)" % N_FOLDS,
                    "num_negative": NUM_NOT_REGULATE,
                    "n_folds": N_FOLDS,
                    "pair_seed": PAIR_SEED,
                    "verbose_training": bool(args.verbose_training),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def ensure_embeddings(net_tsv: Path, node_txt: Path, n_genes: int) -> Tuple[Any, Any]:
        assert emb_s_path is not None and emb_t_path is not None
        if emb_s_path.is_file() and emb_t_path.is_file():
            print("[deepfgrn-pancancer] load cached embeddings:", emb_s_path)
            return np.load(str(emb_s_path)), np.load(str(emb_t_path))
        print("[deepfgrn-pancancer] train dggan embeddings (cached after)...")
        emb_s, emb_t = dggan_embedding_param.dggan(str(net_tsv), str(node_txt))
        if emb_s.shape[0] != n_genes or emb_t.shape[0] != n_genes:
            raise RuntimeError("embedding nodes != n_genes")
        np.save(str(emb_s_path), emb_s)
        np.save(str(emb_t_path), emb_t)
        return emb_s, emb_t

    def slice_features(dataX, a, b, c, d, e, f):
        x = dataX
        return (
            x[:, :, 0:a],
            x[:, :, a:b],
            x[:, :, b:c],
            x[:, :, c:d],
            x[:, :, d:e],
            x[:, :, e:f],
        )

    def run_one_fold(
        fold_idx: int,
        gse_id_fold: str,
        exp_raw,
        pos2,
        pos1,
        neg_positions,
        emb_s,
        emb_t,
        dim_exp: int,
        a,
        b,
        c,
        d,
        e,
        f,
        out_run_inner: Path,
        skip_fit: bool = False,
        export_test_probs_dir: Path | None = None,
        cv_train: List[int] | None = None,
        cv_val: List[int] | None = None,
        cv_test: List[int] | None = None,
        fold_out_override: Path | None = None,
        fold_idx_1based: int = 0,
        pairs_train: Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]] | None = None,
        pairs_val: Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]] | None = None,
        pairs_test: Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]] | None = None,
        pair_split_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(42 + int(fold_idx))
        if cv_train is not None and cv_val is not None and cv_test is not None:
            train_idx = np.asarray(cv_train, dtype=np.int64)
            val_idx = np.asarray(cv_val, dtype=np.int64)
            test_idx = np.asarray(cv_test, dtype=np.int64)
        else:
            train_idx, val_idx, test_idx = ku.split_cells_unseen_523(exp_raw.shape[1], fold_idx, rng)
        tr_i, va_i, te_i, m = ku.subsample_to_common_length(train_idx, val_idx, test_idx, rng)
        if m != dim_exp:
            raise RuntimeError("dim_exp mismatch fold %s (got m=%d expected %d)" % (fold_idx, m, dim_exp))

        wall_t0 = now_mono()
        prep_t0 = now_mono()
        exp_tr, exp_va, exp_te = ku.preprocess_expression_train_val_test(exp_raw, tr_i, va_i, te_i)

        # Held-out pairs (default): train/val/test use disjoint (i,j).
        # Legacy same-pairs: reuse the full pool on every cell split.
        if pairs_train is not None and pairs_val is not None and pairs_test is not None:
            p2_tr, p1_tr, n0_tr = ku.pack_pairs_to_samples(
                exp_tr, pairs_train[0], pairs_train[1], pairs_train[2], emb_s, emb_t
            )
            p2_va, p1_va, n0_va = ku.pack_pairs_to_samples(
                exp_va, pairs_val[0], pairs_val[1], pairs_val[2], emb_s, emb_t
            )
            p2_te, p1_te, n0_te = ku.pack_pairs_to_samples(
                exp_te, pairs_test[0], pairs_test[1], pairs_test[2], emb_s, emb_t
            )
        else:
            p2_tr, p1_tr, n0_tr = ku.pack_pairs_to_samples(exp_tr, pos2, pos1, neg_positions, emb_s, emb_t)
            p2_va, p1_va, n0_va = ku.pack_pairs_to_samples(exp_va, pos2, pos1, neg_positions, emb_s, emb_t)
            p2_te, p1_te, n0_te = ku.pack_pairs_to_samples(exp_te, pos2, pos1, neg_positions, emb_s, emb_t)

        def build_and_stack(p2, p1, n0):
            dtf, dtt, ntfs, ntft, nts, ntt, ly, _ = ku.assemble_arrays(
                p2, p1, n0, three_utils_2_single.transform_data_single_net
            )
            dx = ku.build_data_tensor(dtf, dtt, ntfs, ntft, nts, ntt, a, b, c, d, e, f)
            return dx, ly

        trainX, trainY = build_and_stack(p2_tr, p1_tr, n0_tr)
        valX, valY = build_and_stack(p2_va, p1_va, n0_va)
        testX, testY = build_and_stack(p2_te, p1_te, n0_te)

        trainXX_tf, trainXX_tar, trainXX_ntfs, trainXX_ntft, trainXX_nts, trainXX_ntt = slice_features(
            trainX, a, b, c, d, e, f
        )
        valXX_tf, valXX_tar, valXX_ntfs, valXX_ntft, valXX_nts, valXX_ntt = slice_features(valX, a, b, c, d, e, f)
        testX_tf, testX_tar, testX_ntfs, testX_ntft, testX_nts, testX_ntt = slice_features(testX, a, b, c, d, e, f)
        prep_time_sec = float(now_mono() - prep_t0)
        print(
            "[deepfgrn-pancancer] feature prep %.1fs (train/val/test n=%d/%d/%d dim_exp=%d)"
            % (prep_time_sec, trainX.shape[0], valX.shape[0], testX.shape[0], dim_exp),
            flush=True,
        )

        fold_out = fold_out_override if fold_out_override is not None else out_run_inner / ("fold_%d" % (fold_idx + 1))
        fold_out.mkdir(parents=True, exist_ok=True)
        out_dir = str(fold_out) + os.sep
        n_cells_train_full = int(len(train_idx))
        n_cells_val_full = int(len(val_idx))
        n_cells_test_full = int(len(test_idx))

        classifier = corrresnet_pred.Classifier_corrResNET_pred(
            out_dir,
            NB_CLASSES,
            trainXX_tf,
            trainXX_tar,
            trainXX_ntfs,
            trainXX_ntft,
            trainXX_nts,
            trainXX_ntt,
            verbose=bool(args.verbose_training),
            patience=5,
        )

        ckpt = fold_out / "best_model.keras"
        train_t0 = now_mono()
        if skip_fit:
            if not ckpt.is_file():
                raise FileNotFoundError("Missing %s (cm-only requires saved best_model.keras)" % ckpt)
            score_1 = classifier.predict(
                testX_tf,
                testX_tar,
                testX_ntfs,
                testX_ntft,
                testX_nts,
                testX_ntt,
            )
            score_int = np.argmax(score_1, axis=1)
            train_time_sec = 0.0
        else:
            score_1, score_int = classifier.fit_5CV(
                trainXX_tf,
                trainXX_tar,
                trainXX_ntfs,
                trainXX_ntft,
                trainXX_nts,
                trainXX_ntt,
                trainY,
                valXX_tf,
                valXX_tar,
                valXX_ntfs,
                valXX_ntft,
                valXX_nts,
                valXX_ntt,
                valY,
                testX_tf,
                testX_tar,
                testX_ntfs,
                testX_ntft,
                testX_nts,
                testX_ntt,
            )
            train_time_sec = float(now_mono() - train_t0)

        eval_t0 = now_mono()
        metrics, cm = ku.compute_metrics_bundle(testY, score_1, score_int)
        metrics["Fold"] = fold_idx + 1
        preserve_metrics = str(os.environ.get("DEEPFGRN_PRESERVE_METRICS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if fold_out_override is not None and not preserve_metrics:
            # Write metrics ASAP: Windows/CUDA teardown often kills the process during plot export.
            step01 = deepfgrn_metrics_to_step01_json(
                metrics,
                cm,
                gse_id=gse_id_fold,
                fold_idx=int(fold_idx_1based or fold_idx + 1),
                y_true=testY,
                y_prob=score_1,
            )
            (fold_out / "unseen_samples_test_metrics.json").write_text(
                json.dumps(step01, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            try:
                write_runtime_json(
                    fold_out,
                    wall_time_sec=float(now_mono() - wall_t0),
                    train_time_sec=train_time_sec,
                    eval_time_sec=float(now_mono() - eval_t0),
                    n_genes_used=int(exp_raw.shape[0]),
                    gene_policy="all_genes",
                    extra={
                        "method": "DeepFGRN",
                        "gse_id": gse_id_fold,
                        "fold_idx": int(fold_idx_1based or fold_idx + 1),
                        "split_source": "shared_cv_splits_10fold",
                        "sample_split": [5, 2, 3],
                        "n_cells_train_full": n_cells_train_full,
                        "n_cells_val_full": n_cells_val_full,
                        "n_cells_test_full": n_cells_test_full,
                        "n_cells_train": int(len(tr_i)),
                        "n_cells_val": int(len(va_i)),
                        "n_cells_test": int(len(te_i)),
                        "dim_exp_aligned": int(m),
                        "pair_protocol": (
                            (pair_split_meta or {}).get("pair_protocol")
                            if pair_split_meta
                            else "same_pairs_all_splits"
                        ),
                        "pair_split": pair_split_meta,
                        "dggan_gold_graph": (
                            (pair_split_meta or {}).get("dggan_gold_graph")
                            if pair_split_meta
                            else "full_positives"
                        ),
                        "negative_protocol": neg_meta.get(
                            "negative_protocol", "from_positive_universe_tf_source"
                        ),
                        "note": (
                            "Cell columns from TripGRN-shared cv_splits_10fold.json; "
                            "DeepFGRN subsample_to_common_length for equal-length CorrResNet inputs. "
                            "Negatives sampled from positive TF/gene universe only. "
                            + (
                                "Gene pairs held-out (~64/16/20); dGGAN gold graph = TRAIN positives only "
                                "(val/test edges excluded from embedding graph). Architecture unchanged."
                                if pair_split_meta
                                else "Legacy: same gene pairs reused across train/val/test (cells only change); "
                                "dGGAN uses full positive graph."
                            )
                        ),
                        "prep_time_sec": float(prep_time_sec),
                        "source": "run_deepfgrn_pancancer_cell_unseen",
                    },
                )
            except Exception as exc:
                print(f"[deepfgrn-pancancer] warn: runtime.json write failed: {exc}", flush=True)
        if export_test_probs_dir is not None:
            export_test_probs_dir = Path(export_test_probs_dir)
            export_test_probs_dir.mkdir(parents=True, exist_ok=True)
            y_true_arr = np.asarray(testY)
            if y_true_arr.ndim > 1:
                y_true_arr = np.argmax(y_true_arr, axis=1)
            np.save(str(export_test_probs_dir / "test_y_true.npy"), y_true_arr.astype(np.int64))
            np.save(str(export_test_probs_dir / "test_y_prob.npy"), np.asarray(score_1, dtype=np.float32))
            print("[deepfgrn-pancancer] exported test probs ->", export_test_probs_dir)
        if not preserve_metrics:
            try:
                export_confusion_matrix_artifacts(
                    fold_out,
                    cm,
                    title="DeepFGRN test confusion matrix",
                    class_names=FGRN_CLASS_NAMES,
                    extra={"method": "DeepFGRN", "fold": int(fold_idx_1based or fold_idx + 1), "gse_id": gse_id_fold},
                )
            except Exception as exc:
                print(f"[deepfgrn-pancancer] warn: CM plot export failed: {exc}", flush=True)

        try:
            import tensorflow.keras as keras

            keras.backend.clear_session()
        except Exception:
            pass
        return metrics

    print("[deepfgrn-pancancer] load expression:", npy_values if npy_values is not None else (path_exp if path_exp is not None else npy_expression_paths(gse_id, grn_root)[0]))
    if npy_values is not None:
        gene_names = np.load(str(npy_genes), allow_pickle=True).astype(str).tolist()
        exp_raw = np.load(str(npy_values)).astype(np.float32)
        if exp_raw.shape[0] != len(gene_names):
            raise RuntimeError("npy genes/values row count mismatch")
    elif path_exp is not None:
        exp_raw, gene_names = ku.load_gse253718_expression(path_exp)
    else:
        v_p, g_p = npy_expression_paths(gse_id, grn_root)
        gene_names = np.load(str(g_p), allow_pickle=True).astype(str).tolist()
        exp_raw = np.load(str(v_p)).astype(np.float32)
        if exp_raw.shape[0] != len(gene_names):
            raise RuntimeError("npy genes/values row count mismatch")
    n_genes, n_cells = exp_raw.shape
    print("  genes=%d cells=%d" % (n_genes, n_cells))

    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    print("[deepfgrn-pancancer] load GRN:", path_kegg)
    pos_edges = ku.load_kegg_positive_edges(path_kegg, gene_to_idx)
    print("  positive edges (mapped):", len(pos_edges))
    if len(pos_edges) == 0:
        raise SystemExit(
            "GRN 边无法在表达矩阵基因上映射（检查基因是否为 HGNC symbol，而非 ENSG）。"
        )

    pair_rng = np.random.RandomState(PAIR_SEED)
    # Positives: same as DeepFGRN (label 2=activation, 1=repression).
    # Negatives: ONLY from positive-edge gene universe (TF source ∈ gold sources).
    pos2: List[Tuple[int, int]] = []
    pos1: List[Tuple[int, int]] = []
    for i, j, lab in pos_edges:
        if int(lab) == 2:
            pos2.append((int(i), int(j)))
        elif int(lab) == 1:
            pos1.append((int(i), int(j)))
    n_pos = len(pos2) + len(pos1)
    # Default (<=0): 1:1 with positives (TripGRN-aligned). Explicit >0 keeps a fixed count.
    n_neg = int(n_pos if int(NUM_NOT_REGULATE) <= 0 else NUM_NOT_REGULATE)
    neg_positions, neg_meta = _sample_negatives_from_positives(
        pos_edges, n_neg, pair_rng
    )
    print(
        "  fixed pairs: activate=%d inhibit=%d negative=%d (ratio neg:pos=%.3f; "
        "neg from positive TF/gene universe; sources=%d genes=%d)"
        % (
            len(pos2),
            len(pos1),
            len(neg_positions),
            (len(neg_positions) / max(n_pos, 1)),
            neg_meta["n_positive_sources"],
            neg_meta["n_positive_genes"],
        )
    )

    held_out_pairs = bool(args.held_out_pairs)
    pairs_train = pairs_val = pairs_test = None
    pair_split_meta: Dict[str, Any] | None = None
    # Architecture unchanged: still dGGAN→CorrResNet. Only the gold-standard graph
    # fed to dGGAN is restricted to train positives under held-out pairs.
    if held_out_pairs:
        pairs_train, pairs_val, pairs_test, pair_split_meta = _held_out_split_pairs(
            pos2, pos1, neg_positions, seed=PAIR_SPLIT_SEED
        )
        # DeepFGRN labels: 2=activation, 1=repression (same as build_fixed_pair_structure).
        dggan_edges: List[Tuple[int, int, int]] = [
            (i, j, 2) for i, j in pairs_train[0]
        ] + [(i, j, 1) for i, j in pairs_train[1]]
        graph_tag = "trainpairs"
        print(
            "  held-out pairs ON (seed=%d): train/val/test totals = %d/%d/%d"
            % (
                PAIR_SPLIT_SEED,
                pair_split_meta["n_pairs_train"]["total"],
                pair_split_meta["n_pairs_val"]["total"],
                pair_split_meta["n_pairs_test"]["total"],
            )
        )
        print(
            "  dGGAN gold graph = TRAIN positives only (%d edges); val/test edges excluded"
            % len(dggan_edges)
        )
        pair_split_meta["dggan_gold_graph"] = "train_positives_only"
        pair_split_meta["n_dggan_edges"] = int(len(dggan_edges))
        pair_split_meta["negative_protocol"] = neg_meta.get("negative_protocol")
        pair_split_meta["n_positive_sources"] = neg_meta.get("n_positive_sources")
        pair_split_meta["n_positive_genes"] = neg_meta.get("n_positive_genes")
        if step01_fold_mode and out_run is not None:
            (out_run / "pair_split.json").write_text(
                json.dumps(pair_split_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        dggan_edges = list(pos_edges)
        graph_tag = "fullgraph"
        print("  held-out pairs OFF: same (i,j) reused on train/val/test (legacy cell-only protocol)")
        print("  dGGAN gold graph = FULL positives (%d edges)" % len(dggan_edges))

    grn_stem = Path(path_kegg).stem.replace(" ", "_")[:48]
    # Suffix encodes matched-neg + 1:1 (or fixed count): changes held-out splits / train graph.
    neg_tag = "neg1to1" if int(NUM_NOT_REGULATE) <= 0 else ("neg%d" % int(n_neg))
    input_tag = "%s_%s_%s_ps%d_ss%d_%s" % (
        gse_id,
        grn_stem,
        graph_tag,
        int(PAIR_SEED),
        int(PAIR_SPLIT_SEED if held_out_pairs else 0),
        neg_tag,
    )
    dggan_dir = deep_root / "traindataHuman" / ("inputs_%s" % input_tag)
    emb_s_path = dggan_dir / "cached_GRN_embedding_s.npy"
    emb_t_path = dggan_dir / "cached_GRN_embedding_t.npy"
    dggan_dir.mkdir(parents=True, exist_ok=True)
    net_tsv, node_txt = ku.write_dggan_inputs(dggan_dir, gene_names, dggan_edges)
    emb_s, emb_t = ensure_embeddings(net_tsv, node_txt, n_genes)

    cv_payload = None
    if args.cv_splits_json is not None:
        cv_payload = load_cv_splits(Path(args.cv_splits_json))

    fold_rows: List[Dict[str, Any]] = []
    if step01_multi_fold:
        assert fold_list_1based is not None
        fold_list = [i - 1 for i in fold_list_1based]
        n_run_folds = len(fold_list)
    elif step01_fold_mode:
        fold_list = [int(args.fold_idx) - 1]
        n_run_folds = 1
    else:
        fold_list = list(range(N_FOLDS))
        n_run_folds = N_FOLDS

    for fold_i, fold in enumerate(fold_list):
        fold_idx_1based = fold + 1
        cv_tr = cv_va = cv_te = None
        fold_out_override = None

        if cv_payload is not None:
            fi = fold_idx_1based
            cv_fold = get_fold_split(cv_payload, fi)
            cv_tr = cv_fold["train_cols"]
            cv_va = cv_fold["val_cols"]
            cv_te = cv_fold["test_cols"]
            tr0 = np.asarray(cv_tr, dtype=np.int64)
            va0 = np.asarray(cv_va, dtype=np.int64)
            te0 = np.asarray(cv_te, dtype=np.int64)
            # Match run_one_fold: seed 42+(0-based fold), same rng for subsample.
            fold_rng = np.random.default_rng(42 + fold)
            if step01_multi_fold:
                fold_out_override = out_run / ("fold_%02d" % fi)
            elif step01_fold_mode:
                fold_out_override = out_run
        else:
            fold_rng = np.random.default_rng(42 + fold)
            tr0, va0, te0 = ku.split_cells_unseen_523(n_cells, fold, fold_rng)

        _, _, _, dim_exp = ku.subsample_to_common_length(tr0, va0, te0, fold_rng)
        a = dim_exp
        b = a + dim_exp
        c = b + DIM_NET
        d = c + DIM_NET
        e = d + DIM_NET
        f = e + DIM_NET
        if fold_i == 0 or step01_multi_fold:
            print("  fold %d dim_exp (min segment len) = %d" % (fold_idx_1based, dim_exp), flush=True)

        if args.verbose_training:
            print("\n" + "=" * 60)
            print("Fold %d / %d (cells 5:2:3 unseen)" % (fold_idx_1based, n_run_folds))
            print("=" * 60)
        else:
            print(
                "[deepfgrn-pancancer] Fold %d/%d — CorrResNet（batch=%s；加 --verbose-training 可看完整进度条）"
                % (fold_idx_1based, n_run_folds, os.environ.get("DEEPFGRN_BATCH_SIZE", "32")),
                flush=True,
            )

        row = run_one_fold(
            fold,
            gse_id,
            exp_raw,
            pos2,
            pos1,
            neg_positions,
            emb_s,
            emb_t,
            dim_exp,
            a,
            b,
            c,
            d,
            e,
            f,
            out_run,
            skip_fit=bool(args.cm_only),
            export_test_probs_dir=args.export_test_probs if args.cm_only else None,
            cv_train=cv_tr,
            cv_val=cv_va,
            cv_test=cv_te,
            fold_out_override=fold_out_override,
            fold_idx_1based=fold_idx_1based,
            pairs_train=pairs_train,
            pairs_val=pairs_val,
            pairs_test=pairs_test,
            pair_split_meta=pair_split_meta,
        )
        fold_rows.append(row)
        if args.cm_only:
            print("[deepfgrn-pancancer] exported CM ->", fold_out_override or (out_run / ("fold_%d" % fold_idx_1based)))
            if args.export_test_probs is not None:
                print("[deepfgrn-pancancer] test prob npy ->", args.export_test_probs)
            continue
        if args.verbose_training:
            print("test metrics:", row)
        else:
            print(
                "[deepfgrn-pancancer] fold %d test  AUC=%s  AUPR=%s  MCC=%s  F1=%s"
                % (fold_idx_1based, row.get("AUC"), row.get("AUPR"), row.get("MCC"), row.get("F1")),
                flush=True,
            )

    if args.cm_only:
        print("\n[deepfgrn-pancancer] cm-only done ->", out_run)
        return

    if step01_fold_mode or args.skip_summary_plots:
        print("\n[deepfgrn-pancancer] step01 fold done ->", out_run)
        return

    col_order = [
        "Fold",
        "AUC",
        "AUPR",
        "Macro_AUC_OVR",
        "ACC_无调控",
        "ACC_激活",
        "ACC_抑制",
        "Recall",
        "Precision",
        "F1",
        "MCC",
        "AUC_OVR_无调控",
        "AUC_OVR_激活",
        "AUC_OVR_抑制",
        "AUC_ovo_无调控_抑制",
        "AUC_ovo_无调控_激活",
        "AUC_ovo_抑制_激活",
    ]
    df = pd.DataFrame(fold_rows)
    for c in col_order:
        if c not in df.columns:
            df[c] = np.nan
    df = df[col_order]
    if N_FOLDS > 1:
        mean_series = df[col_order[1:]].mean(numeric_only=True)
        mean_row = {"Fold": "Mean"}
        for c in col_order[1:]:
            v = mean_series.get(c, np.nan)
            mean_row[c] = float(v) if pd.notna(v) else np.nan
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        mean_metrics = {k: mean_row[k] for k in col_order if k != "Fold"}
    else:
        mean_metrics = {
            k: float(fold_rows[0].get(k, np.nan)) for k in col_order if k != "Fold"
        }

    xlsx_path = out_run / ("%s_metrics.xlsx" % run_tag)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="metrics", index=False)

    rad_labels, rad_vals = ku.metrics_to_radar_axes(mean_metrics)
    radar_title = (
        "%s test metrics (MCC axis normalized)" % run_tag
        if N_FOLDS == 1
        else "%s mean across folds (MCC axis normalized)" % run_tag
    )
    ku.plot_radar(
        rad_labels,
        rad_vals,
        title=radar_title,
        out_path=out_run / ("%s_radar_mean.png" % run_tag),
    )
    ku.plot_boxplots(fold_rows, out_path=out_run / ("%s_boxplots.png" % run_tag))

    print("\n[deepfgrn-pancancer] done ->", out_run)
    print(" xlsx:", xlsx_path)


if __name__ == "__main__":
    main()
