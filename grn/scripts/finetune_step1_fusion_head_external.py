"""
在固定 Step1 编码器（表达 / CDS / PPI）的前提下，仅微调 fc_fusion + classifier，
适配外部域（如 GSE131907）表达矩阵；数据管线与 evaluate_step1_on_external_expression 一致。

训练：外部矩阵 train 列 + 类别均衡采样（与 train_unseen_samples_no_cv 相同）。
早停：验证集 macro-F1 或 accuracy。
结束后在同一 output-dir 写出 best_model_fold0.pth 并可选跑事后校准评估（与 step3 脚本对齐）。

支持 --cv-splits-json + --fold-idx 做分域准确率 10-fold CV。
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from grn.models.fusion_model import FusionModelAblation
from grn.reproducibility import set_seed
from grn.scripts.evaluate_step1_on_external_expression import run_single_external_eval_from_cfg
from grn.scripts.regenerate_unseen_run_plots_from_checkpoint import _build_pipeline, _validate_args_like_train
from grn.scripts.train_unseen_samples_no_cv import (
    _autocast_bf16,
    _evaluate,
    _forward_logits,
    _prepare_batch,
)


def _freeze_backbone_train_fusion_only(model: FusionModelAblation) -> None:
    for name, p in model.named_parameters():
        p.requires_grad = ("fc_fusion" in name) or ("classifier" in name)


def _load_column_splits(
    cv_splits_json: Optional[Path],
    fold_idx: Optional[int],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if cv_splits_json is None:
        return None
    if fold_idx is None:
        raise ValueError("--fold-idx is required when --cv-splits-json is set")
    from grn.data.pancancer_cv import get_fold_split, load_cv_splits

    payload = load_cv_splits(cv_splits_json)
    fold = get_fold_split(payload, int(fold_idx))
    return (
        np.asarray(fold["train_cols"], dtype=np.int64),
        np.asarray(fold["val_cols"], dtype=np.int64),
        np.asarray(fold["test_cols"], dtype=np.int64),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Finetune Step1 fusion+classifier on external npy (backbone frozen).")
    ap.add_argument("--step1-run-dir", type=Path, required=True)
    ap.add_argument("--external-exp-npy-values", type=Path, required=True)
    ap.add_argument("--external-exp-npy-genes", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--expression-norm-fit-all-columns", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--early-stop-metric", choices=("macro_f1", "accuracy"), default="macro_f1")
    ap.add_argument("--batch-size", type=int, default=None, help="默认沿用 Step1 run_config.json")
    ap.add_argument(
        "--sample-split",
        default=None,
        metavar="W1,W2,W3",
        help='覆盖列划分（与 Step1 相同）：train/val/test 列数按 w1:w2:w3 比例切分。默认用 run_config.json 如 "5,2,3"=50%% 列用于微调。'
        '约 10%% 训练列可用 "1,2,7"；约 5%% 可用 "1,3,16"。',
    )
    ap.add_argument("--cv-splits-json", type=Path, default=None, help="10-fold CV splits JSON（显式列索引）。")
    ap.add_argument("--fold-idx", type=int, default=None, help="1-based fold when using --cv-splits-json.")
    ap.add_argument("--eval-posthoc-calibrate", action="store_true")
    ap.add_argument("--calibration-steps", type=int, default=600)
    ap.add_argument("--skip-plots", action="store_true", help="评估时尽量少写图（仍写 metrics json）。")
    args = ap.parse_args()

    step1_dir = Path(args.step1_run_dir).resolve()
    ckpt0 = step1_dir / "best_model_fold0.pth"
    cfg_path = step1_dir / "run_config.json"
    if not ckpt0.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"need {ckpt0} and {cfg_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    column_splits = _load_column_splits(args.cv_splits_json, args.fold_idx)

    with cfg_path.open(encoding="utf-8") as f:
        cfg: Dict[str, Any] = json.load(f)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.sample_split:
        cfg["sample_split"] = str(args.sample_split).strip()
    cfg["exp_npy_values"] = str(Path(args.external_exp_npy_values).resolve())
    cfg["exp_npy_genes"] = str(Path(args.external_exp_npy_genes).resolve())
    cfg["exp_csv"] = None
    # Backbone is frozen; checkpointing is unnecessary and breaks on some PyTorch 2.x builds
    # (RuntimeError: autograd.Function is missing clear_saved_tensors_on_access).
    cfg["expression_gradient_checkpointing"] = False

    _validate_args_like_train(Namespace(**cfg))
    set_seed(int(cfg.get("seed", 42)))

    model, train_loader, val_loader, test_loader, device, use_amp_bf16 = _build_pipeline(
        Namespace(**cfg),
        norm_fit_all_columns=bool(args.expression_norm_fit_all_columns),
        include_train_loader=True,
        column_splits=column_splits,
    )
    if train_loader is None:
        raise RuntimeError("internal error: train_loader missing")
    model.load_state_dict(torch.load(str(ckpt0), map_location=device))
    _freeze_backbone_train_fusion_only(model)
    for enc_name in ("exp_encoder_tf", "exp_encoder_tg"):
        enc = getattr(model, enc_name, None)
        if enc is not None and hasattr(enc, "gradient_checkpointing"):
            enc.gradient_checkpointing = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=float(args.lr))
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0
    bad_epochs = 0
    best_path = out_dir / "fusion_finetuned_best.pth"
    metric_key = "f1_macro" if args.early_stop_metric == "macro_f1" else "accuracy"

    for ep in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"fusion-ft {ep+1}/{args.epochs}"):
            x = _prepare_batch(batch, device)
            y = batch["label"].long().to(device)
            opt.zero_grad()
            with _autocast_bf16(device, use_amp_bf16):
                logits = _forward_logits(model, x).float()
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        model.eval()
        val_m = _evaluate(model, val_loader, device, use_amp_bf16=use_amp_bf16)
        score = float(val_m[metric_key])
        print(
            f"[fusion-finetune] epoch {ep+1}/{args.epochs} train_loss={total_loss/max(n_batches,1):.4f} "
            f"val_{metric_key}={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(model.state_dict(), str(best_path))
        else:
            bad_epochs += 1
            if bad_epochs >= int(args.patience):
                print(f"[fusion-finetune] early stop at epoch {ep+1}", flush=True)
                break

    fin_record = {
        "step1_run_dir": str(step1_dir),
        "init_checkpoint": str(ckpt0),
        "external_exp_npy_values": cfg["exp_npy_values"],
        "external_exp_npy_genes": cfg["exp_npy_genes"],
        "sample_split_effective": str(cfg.get("sample_split", "5,2,3")),
        "cv_splits_json": str(args.cv_splits_json) if args.cv_splits_json else None,
        "fold_idx": int(args.fold_idx) if args.fold_idx is not None else None,
        "expression_norm_fit_all_columns": bool(args.expression_norm_fit_all_columns),
        "epochs_ran": int(args.epochs),
        "lr": float(args.lr),
        "patience": int(args.patience),
        "early_stop_metric": str(args.early_stop_metric),
        "best_val_score": float(best_score),
        "best_val_metric_key": metric_key,
    }
    with (out_dir / "finetune_config.json").open("w", encoding="utf-8") as f:
        json.dump(fin_record, f, ensure_ascii=False, indent=2)

    model.load_state_dict(torch.load(str(best_path), map_location=device))
    torch.save(model.state_dict(), str(out_dir / "best_model_fold0.pth"))

    meta_extra: Dict[str, Any] = {
        "finetune_fusion_head": True,
        "init_checkpoint": str(ckpt0),
    }
    if args.fold_idx is not None:
        meta_extra["fold_idx"] = int(args.fold_idx)
    if args.cv_splits_json is not None:
        meta_extra["cv_splits_json"] = str(Path(args.cv_splits_json).resolve())

    run_single_external_eval_from_cfg(
        step1_run_dir=step1_dir,
        ckpt_path=out_dir / "best_model_fold0.pth",
        cfg=cfg,
        norm_fit_all_columns=bool(args.expression_norm_fit_all_columns),
        run_dir=out_dir,
        meta_extra=meta_extra,
        posthoc_calibrate=bool(args.eval_posthoc_calibrate),
        calibration_steps=int(args.calibration_steps),
        column_splits=column_splits,
    )
    print(f"[fusion-finetune] done out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
