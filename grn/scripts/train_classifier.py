import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from grn.configs.default import ClassifierConfig
from grn.data.datasets import GRNDataset
from grn.data.preprocessing import get_GRN, standard
from grn.evaluation.metrics import calculate_metrics
from grn.evaluation.visualize import plot_confusion_matrix
from grn.models.fusion_model import FusionModel
from grn.reproducibility import set_seed
from grn.training.classifier_trainer import ClassifierTrainer


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _reorder_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    # 从 [NoReg, Rep, Act] 到 [NoReg, Act, Rep]
    cm_reordered = cm[:, [0, 2, 1]]
    cm_reordered = cm_reordered[[0, 2, 1], :]
    return cm_reordered


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def _load_expression_with_optional_alignment(exp_csv: str, ref_gene_list_csv: str = None):
    """
    读取表达矩阵，支持可选“按参考 gene_list 对齐”：
    - 若 ref_gene_list_csv 提供：表达行重排到参考基因顺序，缺失基因补 0，重复基因保留首行。
    - 返回：(gene_list, raw_expression)
    """
    raw_df = pd.read_csv(exp_csv)
    if raw_df.shape[1] < 2:
        raise ValueError(f"表达文件列数不足，至少需要 [gene + >=1 sample]，当前文件: {exp_csv}")

    # 第一列视为 gene symbol，后续列为表达值
    gene_col = raw_df.columns[0]
    expr_cols = raw_df.columns[1:]

    expr_df = raw_df.copy()
    expr_df[gene_col] = expr_df[gene_col].astype(str).str.strip()
    expr_df = expr_df[expr_df[gene_col] != ""]

    # 重复基因保留首个，避免 index 对齐歧义
    expr_df = expr_df.drop_duplicates(subset=[gene_col], keep="first")
    expr_df = expr_df.set_index(gene_col)

    # 确保表达值可转 float
    expr_df = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    ref_indices = None
    if ref_gene_list_csv:
        ref_df = pd.read_csv(ref_gene_list_csv, usecols=[0])
        ref_genes = ref_df.iloc[:, 0].astype(str).str.strip().tolist()
        ref_genes = [g for g in ref_genes if g]
        ref_map = {g: i for i, g in enumerate(ref_genes)}

        # 表达谱做锚点：仅保留“可映射到参考 gene_list（用于 CDS 行定位）”的基因，不做补 0
        expr_genes_in_order = expr_df.index.tolist()
        kept_genes = [g for g in expr_genes_in_order if g in ref_map]
        dropped_expr = len(expr_genes_in_order) - len(kept_genes)
        if len(kept_genes) == 0:
            raise ValueError("表达谱与参考 gene_list 无交集，无法继续训练。")

        aligned = expr_df.loc[kept_genes]
        gene_list = kept_genes
        raw_expression = aligned.values.astype(np.float32)
        ref_indices = [ref_map[g] for g in kept_genes]

        print(
            f"[train_classifier] 表达谱锚点模式: 表达基因={len(expr_genes_in_order)}, "
            f"可映射到参考={len(gene_list)}, 已筛除={dropped_expr}（不补零）"
        )
    else:
        gene_list = expr_df.index.tolist()
        raw_expression = expr_df.values.astype(np.float32)
        print(f"[train_classifier] 表达按原文件顺序读取: genes={len(gene_list)}")

    return gene_list, raw_expression, ref_indices


def main():
    parser = argparse.ArgumentParser(description="Train GRN triple-classifier (TF-Target).")
    parser.add_argument("--exp-csv", required=True, help="Expression CSV path.")
    parser.add_argument(
        "--reference-gene-list-csv",
        default=None,
        help="可选：参考基因列表 CSV（第一列 gene），用于对齐表达行顺序并缺失补零。",
    )
    parser.add_argument("--grn-csv", required=True, help="Known GRN CSV path.")
    parser.add_argument("--cds-file", required=True, help="CDS feature CSV path (20-dim per gene).")
    parser.add_argument(
        "--ppi-csv",
        required=True,
        help="PPI 边表 CSV（from,to）；可用 build_ppi_csv_from_string 从 STRING 生成。",
    )

    parser.add_argument("--output-base-dir", default="experiments", help="Base output directory.")
    parser.add_argument("--experiment-name", default="triple_classifier", help="Experiment name.")
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="输出目录不追加时间戳，直接使用 output-base-dir/experiment-name。",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--neg-sample-ratio", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-cv", action="store_true", help="单次训练，不执行 K 折交叉验证。")
    parser.add_argument("--eval-only", action="store_true", help="仅评估：加载已有模型并导出结果，不重新训练。")
    parser.add_argument("--model-path", default=None, help="评估模式下的模型路径（.pth）。")

    parser.add_argument("--cv-excel-name", default="cross_validation_metrics_testbulk2.xlsx")
    parser.add_argument("--full-pt-name", default="full_dataset_prediction.pt")
    parser.add_argument("--full-csv-name", default="full_prediction_with_genes.csv")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true", default=False, help="是否在训练中弹窗显示图形（不建议一键运行时开启）")

    args = parser.parse_args()

    if args.no_timestamp:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name)
    else:
        run_dir = os.path.join(args.output_base_dir, args.experiment_name, _timestamp_str())
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    print(f"[train_classifier] run_dir={run_dir}")

    # 记录运行配置
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    set_seed(args.seed)

    # ====== 加载原始数据 ======
    gene_list, raw_expression, ref_indices = _load_expression_with_optional_alignment(
        args.exp_csv, ref_gene_list_csv=args.reference_gene_list_csv
    )

    known_GRN_df = pd.read_csv(args.grn_csv)
    gene_set = set(gene_list)
    grn_before = len(known_GRN_df)
    known_GRN_df = known_GRN_df[
        known_GRN_df["TF"].astype(str).isin(gene_set)
        & known_GRN_df["target"].astype(str).isin(gene_set)
    ].copy()
    print(f"[train_classifier] GRN按表达谱基因筛选: {grn_before} -> {len(known_GRN_df)}")
    tf_list = known_GRN_df["TF"].values
    target_list = known_GRN_df["target"].values
    re_type_list = known_GRN_df["Re_type"].values
    known_GRN = (tf_list, target_list, re_type_list)

    # ====== 预处理 ======
    exp_data = standard(raw_expression)
    GRN_matrix = get_GRN(known_GRN, gene_list)

    # ====== 构建数据集 ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cds_features_subset = None
    if ref_indices is not None:
        cds_df = pd.read_csv(args.cds_file, header=None, sep=",", skiprows=1)
        cds_all = cds_df.values.astype(np.float32)
        max_idx = max(ref_indices) if len(ref_indices) > 0 else -1
        if max_idx >= len(cds_all):
            raise ValueError(
                f"CDS行数不足：需要索引到 {max_idx}，但CDS仅有 {len(cds_all)} 行。"
            )
        cds_features_subset = cds_all[ref_indices]
        print(
            f"[train_classifier] CDS按表达谱锚点子集对齐: all={len(cds_all)} -> subset={len(cds_features_subset)}"
        )

    dataset = GRNDataset(
        exp_data=exp_data,
        GRN_matrix=GRN_matrix,
        cds_file=args.cds_file,
        gene_list=gene_list,
        ppi_file=args.ppi_csv,
        neg_sample_ratio=args.neg_sample_ratio,
        device=device,
        cds_features=cds_features_subset,
    )

    # ====== 初始化训练 ======
    cls_config = ClassifierConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        neg_sample_ratio=args.neg_sample_ratio,
        n_splits=args.n_splits,
    )

    trainer = ClassifierTrainer(
        FusionModel,
        dataset,
        cls_config,
        output_dir=run_dir,
        raw_expression=raw_expression,
    )

    if args.no_cv:
        print("\n=== 单次训练模式（no-cv）===")
        best_path = args.model_path or os.path.join(run_dir, "best_model_fold0.pth")
        if args.eval_only:
            if not os.path.exists(best_path):
                raise FileNotFoundError(f"[no-cv eval-only] 未找到模型：{best_path}")
            trainer.best_model_path = best_path
            print(f"[no-cv eval-only] 使用已有模型: {best_path}")
        else:
            model = trainer._train(dataset, fold=0)
            if os.path.exists(best_path):
                trainer.best_model_path = best_path
                print(f"[no-cv] best_model: {best_path}")
            else:
                # 兜底：如果未触发保存，手动保存当前模型
                torch.save(model.state_dict(), best_path)
                trainer.best_model_path = best_path
                print(f"[no-cv] best_model(backup): {best_path}")

        print("\n=== 评估（全数据集） ===")
        full_result = trainer.predict_all()
        torch.save(full_result, os.path.join(run_dir, args.full_pt_name))
        trainer.save_full_prediction_csv(full_result, os.path.join(run_dir, args.full_csv_name))

        metrics_json = os.path.join(run_dir, "no_cv_metrics.json")
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(full_result["metrics"]), f, ensure_ascii=False, indent=2)
        print(f"[no-cv] 已写入指标: {metrics_json}")

        cm = full_result["metrics"]["confusion_matrix"]
        cm_reordered = _reorder_confusion_matrix(cm)
        plot_path = os.path.join(run_dir, "plots", "confusion_matrix_no_cv_full.png")
        plot_confusion_matrix(
            cm_reordered,
            classes=["NoReg", "Activate", "Repress"],
            title="Confusion Matrix (No-CV Full Dataset)",
            save_path=plot_path if args.save_plots else None,
            show=args.show_plots,
        )

        print("\n=== no-cv 总体指标 ===")
        for k, v in full_result["metrics"].items():
            if k in ["confusion_matrix", "y_true", "y_pred", "y_prob"]:
                continue
            if isinstance(v, (list, np.ndarray)):
                print(f"{k}: {[round(x, 4) if x is not None else None for x in v]}")
            else:
                try:
                    print(f"{k}: {v:.4f}")
                except Exception:
                    print(f"{k}: {v}")
        return

    # ====== 交叉验证训练 ======
    results = trainer.cross_validate()

    # ====== 控制台输出（与原 invoking.py 保持一致） ======
    print("交叉验证结果:", results)

    class_names = ["NoReg", "Act", "Rep"]
    class_accs = {name: [] for name in class_names}

    for fold_result in results:
        cm = fold_result["confusion_matrix"]
        y_true = []
        y_pred = []
        for i in range(3):
            for j in range(3):
                y_true.extend([i] * cm[i, j])
                y_pred.extend([j] * cm[i, j])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc_order = [0, 2, 1]  # 新顺序对应原始索引
        for idx, cls in enumerate(acc_order):
            mask = y_true == cls
            if mask.sum() == 0:
                acc = float("nan")
            else:
                # 这里不依赖额外 accuracy_score，复用重建准确率逻辑
                acc = float((y_true[mask] == y_pred[mask]).mean())
            class_accs[class_names[idx]].append(acc)

    print("\n=== 各类别五折交叉验证平均准确率 ===")
    for name in ["NoReg", "Act", "Rep"]:
        mean_acc = np.nanmean(class_accs[name])
        print(f"{name}: {mean_acc:.4f}")

    if results:
        last_result = results[-1]
        cm = last_result["confusion_matrix"]
        cm_reordered = _reorder_confusion_matrix(cm)
        plot_path = os.path.join(run_dir, "plots", "confusion_matrix_last_fold.png")
        plot_confusion_matrix(
            cm_reordered,
            classes=["NoReg", "Act", "Rep"],
            title="Confusion Matrix (Last Fold)",
            save_path=plot_path if args.save_plots else None,
            show=args.show_plots,
        )

    # ====== 保存交叉验证指标到 Excel ======
    records = []
    for fold, res in enumerate(results):
        auc_ovr_list = res.get("auc_ovr", [None, None, None])
        auc_ovo_per_list = res.get("auc_ovo_per_class", [None, None, None])

        rec = {
            "Fold": fold + 1,
            "Accuracy": res["accuracy"],
            "Recall": res["recall"],
            "Precision": res["precision"],
            "F1": res["f1"],
            "MCC": res.get("mcc", None),
            "无调控_accuracy": res.get("无调控_accuracy"),
            "激活_accuracy": res.get("激活_accuracy"),
            "抑制_accuracy": res.get("抑制_accuracy"),
            "macro_auc": res.get("macro_auc"),
            "auc_ovo": res.get("auc_ovo"),
            "AUC_ovo_NoReg": auc_ovo_per_list[0] if len(auc_ovo_per_list) > 0 else None,
            "AUC_ovo_Activate": auc_ovo_per_list[2] if len(auc_ovo_per_list) > 2 else None,
            "AUC_ovo_Repress": auc_ovo_per_list[1] if len(auc_ovo_per_list) > 1 else None,
            "AUC_ovr_NoReg": auc_ovr_list[0] if len(auc_ovr_list) > 0 else None,
            "AUC_ovr_Activate": auc_ovr_list[2] if len(auc_ovr_list) > 2 else None,
            "AUC_ovr_Repress": auc_ovr_list[1] if len(auc_ovr_list) > 1 else None,
        }
        records.append(rec)

    valid_records = [r for r in records if r["Fold"] != "Mean"]
    mean_row = {
        "Fold": "Mean",
        "Accuracy": np.nanmean([r["Accuracy"] for r in valid_records]),
        "Recall": np.nanmean([r["Recall"] for r in valid_records]),
        "Precision": np.nanmean([r["Precision"] for r in valid_records]),
        "F1": np.nanmean([r["F1"] for r in valid_records]),
        "MCC": np.nanmean([r["MCC"] for r in valid_records]),
        "无调控_accuracy": np.nanmean([r["无调控_accuracy"] for r in valid_records]),
        "激活_accuracy": np.nanmean([r["激活_accuracy"] for r in valid_records]),
        "抑制_accuracy": np.nanmean([r["抑制_accuracy"] for r in valid_records]),
        "macro_auc": np.nanmean([r["macro_auc"] for r in valid_records]),
        "auc_ovo": np.nanmean([r["auc_ovo"] for r in valid_records]),
        "AUC_ovo_NoReg": np.nanmean([r["AUC_ovo_NoReg"] for r in valid_records]),
        "AUC_ovo_Activate": np.nanmean([r["AUC_ovo_Activate"] for r in valid_records]),
        "AUC_ovo_Repress": np.nanmean([r["AUC_ovo_Repress"] for r in valid_records]),
        "AUC_ovr_NoReg": np.nanmean([r["AUC_ovr_NoReg"] for r in valid_records]),
        "AUC_ovr_Activate": np.nanmean([r["AUC_ovr_Activate"] for r in valid_records]),
        "AUC_ovr_Repress": np.nanmean([r["AUC_ovr_Repress"] for r in valid_records]),
    }
    records.append(mean_row)

    df = pd.DataFrame(records)
    excel_path = os.path.join(run_dir, args.cv_excel_name)
    df.to_excel(excel_path, index=False)
    print(f"已生成 Excel：{excel_path}")

    # ====== 全数据集预测与落盘 ======
    print("\n=== 正在对整个数据集进行预测（训练+验证） ===")
    full_result = trainer.predict_all()

    torch.save(full_result, os.path.join(run_dir, args.full_pt_name))
    trainer.save_full_prediction_csv(full_result, os.path.join(run_dir, args.full_csv_name))

    cm = full_result["metrics"]["confusion_matrix"]
    cm_reordered = _reorder_confusion_matrix(cm)
    plot_path = os.path.join(run_dir, "plots", "confusion_matrix_full.png")
    plot_confusion_matrix(
        cm_reordered,
        classes=["无调控", "激活", "抑制"],
        title="Confusion Matrix (Full Dataset)",
        save_path=plot_path if args.save_plots else None,
        show=args.show_plots,
    )

    print("\n=== 总体指标 ===")
    for k, v in full_result["metrics"].items():
        if k in ["confusion_matrix", "y_true", "y_pred", "y_prob"]:
            continue
        if isinstance(v, (list, np.ndarray)):
            print(f"{k}: {[round(x, 4) if x is not None else None for x in v]}")
        else:
            try:
                print(f"{k}: {v:.4f}")
            except Exception:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()

