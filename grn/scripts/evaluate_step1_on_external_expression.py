from __future__ import annotations

import argparse
import copy
import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from grn.evaluation.logit_calibration import (
    apply_temperature_and_bias,
    fit_temperature_and_bias,
    val_class_inverse_weights,
)
from grn.evaluation.metrics import calculate_metrics
from grn.evaluation.training_plots import (
    per_class_ovr_accuracy_precision_recall_from_cm,
    save_accuracy_bar_val_test,
    save_confusion_matrix_fig,
    save_grouped_metrics_val_test,
    save_ovr_roc_pr_curves,
    save_per_class_auprc_bar,
    save_per_class_metric_bars,
)
from grn.reproducibility import set_seed
from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import (
    _build_pipeline,
    _validate_args_like_train,
)
from grn.scripts.train_unseen_samples_no_cv import (
    _autocast_bf16,
    _evaluate,
    _forward_logits,
    _predict_arrays,
    _prepare_batch,
)


def _jsonify_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def _collect_logits_numpy(model, loader, device: torch.device, use_amp_bf16: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = _prepare_batch(batch, device)
            y = batch["label"].cpu().numpy()
            with _autocast_bf16(device, use_amp_bf16):
                lg = _forward_logits(model, x).float().cpu().numpy()
            logits_list.append(lg)
            y_list.append(y)
    return np.vstack(logits_list), np.concatenate(y_list)


def _metrics_from_logits(logits: np.ndarray, y: np.ndarray, temperature: float, bias: np.ndarray) -> Dict[str, Any]:
    prob = apply_temperature_and_bias(logits, temperature, bias)
    pred = np.argmax(prob, axis=1)
    return calculate_metrics(y, pred, prob)


def _resolve_run_dir(output_base_dir: str, experiment_name: str, no_timestamp: bool) -> Path:
    if no_timestamp:
        return Path(output_base_dir) / experiment_name
    return Path(output_base_dir) / experiment_name / datetime.now().strftime("%Y%m%d-%H%M%S")


def run_single_external_eval_from_cfg(
    *,
    step1_run_dir: Path,
    ckpt_path: Path,
    cfg: Dict[str, Any],
    norm_fit_all_columns: bool,
    run_dir: Path,
    meta_extra: Dict[str, Any] | None = None,
    posthoc_calibrate: bool = False,
    calibration_steps: int = 600,
    column_splits: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """cfg 须已设好 exp_csv 或 (exp_npy_values + exp_npy_genes)，与 train_unseen_samples_no_cv 一致。

    若 posthoc_calibrate：在验证集 logits 上拟合温度 T 与偏置 b，主输出 metrics 为校准后结果；
    返回 (val_cal, test_cal, val_pre, test_pre)，未校准时后两者为 None。
    """
    _validate_args_like_train(Namespace(**cfg))
    set_seed(int(cfg.get("seed", 42)))

    model, _, val_loader, test_loader, device, use_amp_bf16 = _build_pipeline(
        Namespace(**cfg),
        norm_fit_all_columns=norm_fit_all_columns,
        column_splits=column_splits,
    )
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    val_pre: Optional[Dict[str, Any]] = None
    test_pre: Optional[Dict[str, Any]] = None
    y_test_true: np.ndarray
    y_test_prob: np.ndarray
    if posthoc_calibrate:
        logits_v, y_v = _collect_logits_numpy(model, val_loader, device, use_amp_bf16)
        logits_t, y_t = _collect_logits_numpy(model, test_loader, device, use_amp_bf16)
        val_pre = _metrics_from_logits(logits_v, y_v, 1.0, np.zeros(3, dtype=np.float32))
        test_pre = _metrics_from_logits(logits_t, y_t, 1.0, np.zeros(3, dtype=np.float32))
        cw = val_class_inverse_weights(y_v, n_classes=int(logits_v.shape[1]))
        T, bias = fit_temperature_and_bias(
            logits_v, y_v, steps=int(calibration_steps), class_weights=cw
        )
        val_metrics = _metrics_from_logits(logits_v, y_v, T, bias)
        test_metrics = _metrics_from_logits(logits_t, y_t, T, bias)
        y_test_true = y_t
        y_test_prob = apply_temperature_and_bias(logits_t, T, bias)
        calib_record = {
            "temperature": float(T),
            "bias": bias.tolist(),
            "calibration_steps": int(calibration_steps),
            "class_weights_for_fit": cw.tolist(),
        }
    else:
        val_metrics = _evaluate(model, val_loader, device, use_amp_bf16=use_amp_bf16)
        test_metrics = _evaluate(model, test_loader, device, use_amp_bf16=use_amp_bf16)
        calib_record = None
        y_test_true, y_test_prob = _predict_arrays(model, test_loader, device, use_amp_bf16=use_amp_bf16)

    plot_dir = run_dir / "plots"
    os.makedirs(plot_dir, exist_ok=True)

    meta = {
        "step1_run_dir": str(step1_run_dir),
        "step1_checkpoint": str(ckpt_path),
        "device": str(device),
        "use_amp_bf16": bool(use_amp_bf16),
        "expression_norm_fit_all_columns": norm_fit_all_columns,
        "posthoc_calibrate": bool(posthoc_calibrate),
    }
    if calib_record is not None:
        meta["posthoc_calibration"] = calib_record
    if meta_extra:
        meta.update(meta_extra)

    if posthoc_calibrate and val_pre is not None and test_pre is not None:
        with (run_dir / "unseen_samples_val_metrics_pre_calibration.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonify_metrics(val_pre), f, ensure_ascii=False, indent=2)
        with (run_dir / "unseen_samples_test_metrics_pre_calibration.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonify_metrics(test_pre), f, ensure_ascii=False, indent=2)
        with (run_dir / "posthoc_calibration.json").open("w", encoding="utf-8") as f:
            json.dump(calib_record, f, ensure_ascii=False, indent=2)

    with (run_dir / "unseen_samples_val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonify_metrics(val_metrics), f, ensure_ascii=False, indent=2)
    with (run_dir / "unseen_samples_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonify_metrics(test_metrics), f, ensure_ascii=False, indent=2)
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    with (run_dir / "step3_eval_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_accuracy_bar_val_test(str(plot_dir), float(val_metrics["accuracy"]), float(test_metrics["accuracy"]))
    save_grouped_metrics_val_test(str(plot_dir), val_metrics, test_metrics)
    cm_v = np.asarray(val_metrics["confusion_matrix"])
    cm_t = np.asarray(test_metrics["confusion_matrix"])
    save_confusion_matrix_fig(str(plot_dir), cm_v, "Validation confusion matrix", "confusion_matrix_val.png")
    save_confusion_matrix_fig(str(plot_dir), cm_t, "Test confusion matrix", "confusion_matrix_test.png")
    disp_names = ["No regulation", "Activation", "Repression"]
    disp_idx = [0, 2, 1]
    acc_ovr, prec_pc, rec_pc = per_class_ovr_accuracy_precision_recall_from_cm(cm_t)
    save_per_class_metric_bars(
        str(plot_dir),
        disp_names,
        [float(acc_ovr[i]) for i in disp_idx],
        [float(prec_pc[i]) for i in disp_idx],
        [float(rec_pc[i]) for i in disp_idx],
    )
    save_per_class_auprc_bar(str(plot_dir), disp_names, [float(x) for x in test_metrics["auprc"]])
    save_ovr_roc_pr_curves(
        str(plot_dir),
        y_test_true,
        y_test_prob,
        disp_names,
        disp_idx,
        legend_roc_ovr=[float(x) for x in test_metrics["auc_ovr"]],
        legend_auprc=[float(x) for x in test_metrics["auprc"]],
    )

    print(f"[step3] run_dir={run_dir}")
    if posthoc_calibrate and test_pre is not None:
        print(
            f"[step3] pre-cal  test_acc={float(test_pre['accuracy']):.6f} macro_f1={float(test_pre['f1_macro']):.6f} "
            f"mcc={float(test_pre['mcc']):.6f}"
        )
    print(
        f"[step3] val_acc={float(val_metrics['accuracy']):.6f}, "
        f"test_acc={float(test_metrics['accuracy']):.6f}, test_f1_macro={float(test_metrics['f1_macro']):.6f}, "
        f"test_macro_auc={float(test_metrics['macro_auc']):.6f}, test_mcc={float(test_metrics['mcc']):.6f}"
        + ("  (posthoc calibrated)" if posthoc_calibrate else "")
    )
    return val_metrics, test_metrics, val_pre, test_pre


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate fixed step1 checkpoint on external expression with step1-like val/test outputs and plots."
    )
    ap.add_argument("--step1-run-dir", required=True, help="Directory containing run_config.json and best_model_fold0.pth")
    ap.add_argument("--external-exp-csv", default=None, help="External genes-by-cells CSV")
    ap.add_argument("--external-exp-npy-values", default=None, help="External expression npy values")
    ap.add_argument("--external-exp-npy-genes", default=None, help="External expression npy genes")
    ap.add_argument(
        "--gse131907-precomputed-dir",
        default=None,
        help="materialize_gse131907_brain_nonbrain_npys 目录：需含 genes.npy 与 brain_values_log_normalized.npy / "
        "nonbrain_values_log_normalized.npy（可用 gse131907_split_counts_npys_to_log_normalized 生成）。"
        "将分别写出 mbrain/ 与 nonbrain/ 子目录评估结果。",
    )
    ap.add_argument("--output-base-dir", default="experiments")
    ap.add_argument("--experiment-name", default="benchmark_luad_external_step3")
    ap.add_argument("--no-timestamp", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None, help="Override batch size (default from step1 config)")
    ap.add_argument("--seed", type=int, default=None, help="Override seed (default from step1 config)")
    ap.add_argument(
        "--expression-norm-fit-all-columns",
        action="store_true",
        help="log2-minmax 在表达矩阵全部列上拟合（外部单域评估建议开启，减轻预测塌缩到无调控）",
    )
    ap.add_argument(
        "--posthoc-calibrate",
        action="store_true",
        help="在验证集 logits 上拟合温度+偏置后再度量测试集（缓解跨域时 argmax 全判多数类；主 json 为校准后）",
    )
    ap.add_argument("--calibration-steps", type=int, default=600, help="事后校准 Adam 步数（仅 --posthoc-calibrate）")
    ap.add_argument(
        "--sample-split",
        default=None,
        metavar="W1,W2,W3",
        help="覆盖列划分 train/val/test（须与微调一致，如 GSE189357 step04 用 1,2,7）",
    )
    args = ap.parse_args()

    step1_run_dir = Path(args.step1_run_dir).resolve()
    cfg_path = step1_run_dir / "run_config.json"
    ckpt_path = step1_run_dir / "best_model_fold0.pth"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing step1 config: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing step1 checkpoint: {ckpt_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg_template: Dict[str, Any] = json.load(f)

    if args.batch_size is not None:
        cfg_template["batch_size"] = int(args.batch_size)
    if args.seed is not None:
        cfg_template["seed"] = int(args.seed)
    if args.sample_split:
        cfg_template["sample_split"] = str(args.sample_split).strip()

    norm_fit = bool(args.expression_norm_fit_all_columns)

    if args.gse131907_precomputed_dir:
        if args.external_exp_csv or args.external_exp_npy_values or args.external_exp_npy_genes:
            raise ValueError("Do not combine --gse131907-precomputed-dir with CSV/npy expression inputs.")
        split_dir = Path(args.gse131907_precomputed_dir).resolve()
        man_path = split_dir / "manifest.json"
        if not man_path.is_file():
            raise FileNotFoundError(f"Missing {man_path}")
        genes_npy = split_dir / "genes.npy"
        brain_log = split_dir / "brain_values_log_normalized.npy"
        nb_log = split_dir / "nonbrain_values_log_normalized.npy"
        if not genes_npy.is_file():
            raise FileNotFoundError(str(genes_npy))
        if not brain_log.is_file() or not nb_log.is_file():
            raise FileNotFoundError(
                f"Missing log-normalized npy under {split_dir}.\n"
                f"  Expected: {brain_log.name}, {nb_log.name}\n"
                f"  Run: python -m grn.scripts.gse131907_split_counts_npys_to_log_normalized --split-dir \"{split_dir}\""
            )

        parent_dir = _resolve_run_dir(args.output_base_dir, args.experiment_name, args.no_timestamp)
        summary: Dict[str, Any] = {
            "step1_run_dir": str(step1_run_dir),
            "gse131907_precomputed_dir": str(split_dir),
            "expression_norm_fit_all_columns": norm_fit,
            "posthoc_calibrate": bool(args.posthoc_calibrate),
            "sub_evaluations": {},
        }

        for sub, values_path, label in (
            ("mbrain", brain_log, "mBrain"),
            ("nonbrain", nb_log, "non-mBrain"),
        ):
            sub_dir = parent_dir / sub
            os.makedirs(sub_dir, exist_ok=True)
            cfg_sub = copy.deepcopy(cfg_template)
            cfg_sub["exp_npy_values"] = str(values_path)
            cfg_sub["exp_npy_genes"] = str(genes_npy)
            cfg_sub["exp_csv"] = None
            _, test_metrics, _, test_pre = run_single_external_eval_from_cfg(
                step1_run_dir=step1_run_dir,
                ckpt_path=ckpt_path,
                cfg=cfg_sub,
                norm_fit_all_columns=norm_fit,
                run_dir=sub_dir,
                meta_extra={
                    "gse131907_split_subset": sub,
                    "gse131907_precomputed_dir": str(split_dir),
                    "matrix_label": label,
                },
                posthoc_calibrate=bool(args.posthoc_calibrate),
                calibration_steps=int(args.calibration_steps),
            )
            row = {
                "run_dir": str(sub_dir.resolve()),
                "test_accuracy": float(test_metrics["accuracy"]),
                "test_macro_f1": float(test_metrics["f1_macro"]),
                "test_macro_auc": float(test_metrics["macro_auc"]),
                "test_mcc": float(test_metrics["mcc"]),
            }
            if test_pre is not None:
                row["test_accuracy_pre_calibration"] = float(test_pre["accuracy"])
                row["test_macro_f1_pre_calibration"] = float(test_pre["f1_macro"])
                row["test_mcc_pre_calibration"] = float(test_pre["mcc"])
            summary["sub_evaluations"][sub] = row

        parent_dir.mkdir(parents=True, exist_ok=True)
        with (parent_dir / "gse131907_dual_eval_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(
            f"[step3-dual] parent_dir={parent_dir}\n"
            f"  mbrain test_acc={summary['sub_evaluations']['mbrain']['test_accuracy']:.6f}\n"
            f"  nonbrain test_acc={summary['sub_evaluations']['nonbrain']['test_accuracy']:.6f}"
        )
        return

    if bool(args.external_exp_npy_values) ^ bool(args.external_exp_npy_genes):
        raise ValueError("--external-exp-npy-values and --external-exp-npy-genes must be provided together")
    if args.external_exp_csv and args.external_exp_npy_values:
        raise ValueError("Use either --external-exp-csv OR external npy pair, not both")
    if not args.external_exp_csv and not args.external_exp_npy_values:
        raise ValueError(
            "Provide --external-exp-csv, or --external-exp-npy-values/--external-exp-npy-genes, "
            "or --gse131907-precomputed-dir"
        )

    if args.external_exp_npy_values:
        cfg_template["exp_npy_values"] = args.external_exp_npy_values
        cfg_template["exp_npy_genes"] = args.external_exp_npy_genes
        cfg_template["exp_csv"] = None
    else:
        cfg_template["exp_csv"] = args.external_exp_csv
        cfg_template["exp_npy_values"] = None
        cfg_template["exp_npy_genes"] = None

    run_dir = _resolve_run_dir(args.output_base_dir, args.experiment_name, args.no_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    run_single_external_eval_from_cfg(
        step1_run_dir=step1_run_dir,
        ckpt_path=ckpt_path,
        cfg=cfg_template,
        norm_fit_all_columns=norm_fit,
        run_dir=run_dir,
        meta_extra=None,
        posthoc_calibrate=bool(args.posthoc_calibrate),
        calibration_steps=int(args.calibration_steps),
    )


if __name__ == "__main__":
    main()
