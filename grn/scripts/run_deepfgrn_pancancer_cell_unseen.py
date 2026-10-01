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

与 DeepFGRN_GSE253718_FCV.py 一致——细胞列 5:2:3、log2(x+1)+训练集 MinMax、固定 TF–靶基因对与负例数、
KEGG 边构图嵌入 + CorrResNet。默认仅一次细胞划分；可通过 --n-folds 重复多种子划分（类原论文外层折）。

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
from grn.scripts.pancancer_step01_paths import (
    find_latest_fgrn_run_with_ckpt,
    load_run_meta,
    npy_expression_paths,
    resolve_exp_csv,
    resolve_kegg_csv,
    step01_layout,
)

FGRN_CLASS_NAMES = ("No regulation", "Repression", "Activation")


def _resolve_deepfgrn_root(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return cli_path.resolve()
    env = os.environ.get("DEEPFGRN_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    grn_root = Path(__file__).resolve().parents[2]
    return (grn_root.parent / "project_DeepFGRN").resolve()


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
    ap.add_argument("--num-negative", type=int, default=7000, help="负样本边数，与 DeepFGRN_GSE253718_FCV NUM_NOT_REGULATE 一致")
    ap.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="外层重复次数：每次用 split_cells_unseen_523(..., fold_idx=k, rng 种子 42+k)。"
        " 默认 1（单次划分）；设为 5 则与原 DeepFGRN_GSE253718_FCV 五折一致。",
    )
    ap.add_argument("--pair-seed", type=int, default=12345)
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
        help="1-based fold index when using --cv-splits-json (single fold per invocation).",
    )
    ap.add_argument(
        "--output-fold-dir",
        type=Path,
        default=None,
        help="Fixed per-fold output dir (step01: experiments/.../DeepFGRN/GSE/fold_XX).",
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

    step01_fold_mode = args.output_fold_dir is not None and args.cv_splits_json is not None
    if args.cv_splits_json is not None and int(args.fold_idx) <= 0:
        raise SystemExit("--cv-splits-json requires --fold-idx >= 1")

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

    if not args.verbose_training:
        dggan_config_mod.verbose = 0

    NB_CLASSES = 3
    DIM_NET = 224
    N_FOLDS = int(args.n_folds)
    NUM_NOT_REGULATE = int(run_meta.get("num_negative", args.num_negative)) if run_meta else int(args.num_negative)
    PAIR_SEED = int(run_meta.get("pair_seed", args.pair_seed)) if run_meta else int(args.pair_seed)
    input_tag = "%s_KEGG_pancancer" % gse_id

    dggan_dir = deep_root / "traindataHuman" / ("inputs_%s" % input_tag)
    emb_s_path = dggan_dir / "cached_GRN_embedding_s.npy"
    emb_t_path = dggan_dir / "cached_GRN_embedding_t.npy"

    if args.cm_only:
        assert out_run is not None
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

        exp_tr, exp_va, exp_te = ku.preprocess_expression_train_val_test(exp_raw, tr_i, va_i, te_i)

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

        fold_out = fold_out_override if fold_out_override is not None else out_run_inner / ("fold_%d" % (fold_idx + 1))
        fold_out.mkdir(parents=True, exist_ok=True)
        out_dir = str(fold_out) + os.sep

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

        metrics, cm = ku.compute_metrics_bundle(testY, score_1, score_int)
        metrics["Fold"] = fold_idx + 1
        if fold_out_override is not None:
            # Write metrics ASAP: Windows/CUDA teardown often kills the process during plot export.
            step01 = deepfgrn_metrics_to_step01_json(
                metrics,
                cm,
                gse_id=gse_id_fold,
                fold_idx=int(fold_idx_1based or fold_idx + 1),
            )
            (fold_out / "unseen_samples_test_metrics.json").write_text(
                json.dumps(step01, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if export_test_probs_dir is not None:
            export_test_probs_dir = Path(export_test_probs_dir)
            export_test_probs_dir.mkdir(parents=True, exist_ok=True)
            y_true_arr = np.asarray(testY)
            if y_true_arr.ndim > 1:
                y_true_arr = np.argmax(y_true_arr, axis=1)
            np.save(str(export_test_probs_dir / "test_y_true.npy"), y_true_arr.astype(np.int64))
            np.save(str(export_test_probs_dir / "test_y_prob.npy"), np.asarray(score_1, dtype=np.float32))
            print("[deepfgrn-pancancer] exported test probs ->", export_test_probs_dir)
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
    print("[deepfgrn-pancancer] load KEGG:", path_kegg)
    pos_edges = ku.load_kegg_positive_edges(path_kegg, gene_to_idx)
    print("  positive edges (mapped):", len(pos_edges))
    if len(pos_edges) == 0:
        raise SystemExit(
            "KEGG 边无法在表达矩阵基因上映射（检查基因是否为 HGNC symbol，而非 ENSG）。"
        )

    dggan_dir.mkdir(parents=True, exist_ok=True)
    net_tsv, node_txt = ku.write_dggan_inputs(dggan_dir, gene_names, pos_edges)
    emb_s, emb_t = ensure_embeddings(net_tsv, node_txt, n_genes)

    pair_rng = np.random.RandomState(PAIR_SEED)
    pos2, pos1, neg_positions = ku.build_fixed_pair_structure(n_genes, pos_edges, NUM_NOT_REGULATE, pair_rng)
    print(
        "  fixed pairs: activate=%d inhibit=%d negative=%d" % (len(pos2), len(pos1), len(neg_positions))
    )

    rng0 = np.random.default_rng(42)
    if args.cv_splits_json is not None:
        cv_payload = load_cv_splits(Path(args.cv_splits_json))
        cv_fold = get_fold_split(cv_payload, int(args.fold_idx))
        tr0 = np.asarray(cv_fold["train_cols"], dtype=np.int64)
        va0 = np.asarray(cv_fold["val_cols"], dtype=np.int64)
        te0 = np.asarray(cv_fold["test_cols"], dtype=np.int64)
        fold_rng = np.random.default_rng(42 + int(args.fold_idx) - 1)
    else:
        tr0, va0, te0 = ku.split_cells_unseen_523(n_cells, 0, rng0)
        fold_rng = rng0
    _, _, _, dim_exp = ku.subsample_to_common_length(tr0, va0, te0, fold_rng)
    print("  dim_exp (min segment len) =", dim_exp)

    a = dim_exp
    b = a + dim_exp
    c = b + DIM_NET
    d = c + DIM_NET
    e = d + DIM_NET
    f = e + DIM_NET

    fold_rows: List[Dict[str, Any]] = []
    if step01_fold_mode:
        fold_list = [int(args.fold_idx) - 1]
        n_run_folds = 1
    else:
        fold_list = list(range(N_FOLDS))
        n_run_folds = N_FOLDS

    for fold in fold_list:
        if args.verbose_training:
            print("\n" + "=" * 60)
            print("Fold %d / %d (cells 5:2:3 unseen)" % (fold + 1, n_run_folds))
            print("=" * 60)
        else:
            print(
                "[deepfgrn-pancancer] Fold %d/%d — CorrResNet（batch=%s；加 --verbose-training 可看完整进度条）"
                % (fold + 1, n_run_folds, os.environ.get("DEEPFGRN_BATCH_SIZE", "32")),
                flush=True,
            )
        cv_tr = cv_va = cv_te = None
        fold_out_override = None
        fold_idx_1based = fold + 1
        if args.cv_splits_json is not None:
            fi = int(args.fold_idx) if step01_fold_mode else (fold + 1)
            cv_fold = get_fold_split(cv_payload, fi)
            cv_tr = cv_fold["train_cols"]
            cv_va = cv_fold["val_cols"]
            cv_te = cv_fold["test_cols"]
            fold_idx_1based = fi
            if step01_fold_mode:
                fold_out_override = out_run

        row = run_one_fold(
            fold if not step01_fold_mode else int(args.fold_idx) - 1,
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
        )
        fold_rows.append(row)
        if args.cm_only:
            print("[deepfgrn-pancancer] exported CM ->", out_run / ("fold_%d" % (fold + 1)))
            if args.export_test_probs is not None:
                print("[deepfgrn-pancancer] test prob npy ->", args.export_test_probs)
            continue
        if args.verbose_training:
            print("test metrics:", row)
        else:
            print(
                "[deepfgrn-pancancer] fold %d test  AUC=%s  AUPR=%s  MCC=%s  F1=%s"
                % (fold + 1, row.get("AUC"), row.get("AUPR"), row.get("MCC"), row.get("F1")),
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
