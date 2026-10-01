import os
import numpy as np
import torch
from typing import Optional
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from grn.evaluation.metrics import calculate_metrics
from grn.data.preprocessing import standard_with_train_fit


class ClassifierTrainer:
    def __init__(
        self,
        model_class,
        dataset,
        config,
        output_dir: str,
        raw_expression: Optional[np.ndarray] = None,
    ):
        self.model_class = model_class
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skf = StratifiedKFold(
            n_splits=self.config.n_splits, shuffle=True, random_state=self.config.random_state
        )

        first_sample = dataset[0]
        self.seq_dim = first_sample["seq_tf"].shape[0]

        self.best_model_path = None
        self.current_train_genes = None
        self.current_test_genes = None
        self.raw_expression = raw_expression
        print(f"[ClassifierTrainer] seq_dim={self.seq_dim}, num_classes={self.config.num_classes}")

    def _get_gene_indices_from_samples(self, sample_indices):
        gene_indices = set()
        for idx in sample_indices:
            i, j, _ = self.dataset.samples[idx]
            gene_indices.add(i)
            gene_indices.add(j)
        return sorted(list(gene_indices))

    def cross_validate(self):
        X = np.arange(len(self.dataset))
        y = self._get_labels()
        results = []

        for fold, (train_idx, test_idx) in enumerate(self.skf.split(X, y)):
            print(f"\n{'#' * 60}")
            print(f"### Fold {fold + 1}/{self.config.n_splits}")
            print(f"{'#' * 60}")

            train_genes = self._get_gene_indices_from_samples(train_idx)
            test_genes = self._get_gene_indices_from_samples(test_idx)

            # Inductive mask 假设：train genes 与 test genes 不应重叠。
            # 由于当前 K 折是按“样本/TF-Target 对”划分，可能会出现基因重叠。
            # 为了最大程度降低特征侧的泄露风险，这里把重叠基因从 test_mask 中剔除，
            # 并在控制台给出显式警告（但注意这会让“严格无基因重叠评估”变得不完全成立）。
            overlap_genes = set(train_genes).intersection(test_genes)
            if overlap_genes:
                overlap_list = sorted(overlap_genes)
                print(
                    f"[Leakage Risk Warning] Fold {fold + 1}: train/test genes overlap={len(overlap_list)} "
                    f"(例如：{overlap_list[:5]} ...)，将从 test_mask 中剔除重叠基因以降低 Inductive 泄露风险。"
                )
                test_genes_effective = sorted(set(test_genes) - overlap_genes)
            else:
                test_genes_effective = test_genes

            # 进一步降低“重叠基因样本泄露”：把 test_set 中包含重叠基因的 TF-Target 对也剔除。
            # 这样 test 的 inductive inference 才和评估样本集一致。
            test_gene_set_effective = set(test_genes_effective)
            test_idx_effective = []
            for idx in test_idx:
                i, j, _ = self.dataset.samples[idx]
                if i in test_gene_set_effective and j in test_gene_set_effective:
                    test_idx_effective.append(idx)

            if len(test_idx_effective) == 0:
                print(
                    f"[Leakage Risk Warning] Fold {fold + 1}: 剔除重叠基因后的 test 样本为空，回退为使用原始 test_idx "
                    f"（这会降低泄露控制的严格度）。"
                )
                test_idx_effective = list(test_idx)
            elif overlap_genes:
                print(
                    f"[Leakage Control] Fold {fold + 1}: test samples filtered "
                    f"{len(test_idx)} -> {len(test_idx_effective)}（剔除包含重叠基因的 TF-Target 对）"
                )

            # 表达分布标准化：每折仅用 train_genes 拟合 scaler，避免跨折 MinMax 泄露
            if self.raw_expression is not None:
                self.dataset.exp_data = standard_with_train_fit(self.raw_expression, train_genes)

            self._check_and_report_leakage(train_genes, test_genes_effective, fold)

            # 设置训练掩码用于 inductive 特征
            self.dataset.ppi_extractor.set_train_mask(train_genes)
            self.current_train_genes = train_genes
            self.current_test_genes = test_genes_effective

            # 划分数据集（保持原有逻辑，使用样本索引）
            train_set = torch.utils.data.Subset(self.dataset, train_idx)
            test_set = torch.utils.data.Subset(self.dataset, test_idx_effective)

            model = self._train(train_set, fold)
            metrics = self.evaluate(model, test_set, train_genes, test_genes_effective)
            results.append(metrics)

            # 保存最佳模型（以当前结果列表中的 max accuracy 为准）
            if metrics["accuracy"] == max([r["accuracy"] for r in results]):
                fold_path = os.path.join(self.output_dir, f"best_model_fold{fold}.pth")
                torch.save(model.state_dict(), fold_path)
                self.best_model_path = fold_path

            self.dataset.ppi_extractor.clear_train_mask()
            self.current_train_genes = None
            self.current_test_genes = None

        best_fold = int(np.argmax([r["accuracy"] for r in results]))
        print(f"[ClassifierTrainer] 最佳模型为第 {best_fold} 折，路径：{self.best_model_path}")
        return results

    def _check_and_report_leakage(self, train_genes, test_genes, fold):
        """检查并报告 PPI 数据泄露情况（基于基因索引）"""
        edge_index = self.dataset.ppi_extractor.graph.edge_index
        train_set = set(train_genes)
        test_set = set(test_genes)

        cross_edges = 0
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if (src in train_set and dst in test_set) or (src in test_set and dst in train_set):
                cross_edges += 1

        total_edges = edge_index.size(1)
        print(f"\n[Data Leakage Check] Fold {fold + 1}")
        print(f"  训练集基因数: {len(train_genes)}, 测试集基因数: {len(test_genes)}")
        print(f"  总边数: {total_edges}")
        print(f"  跨训练/测试集边数: {cross_edges} ({(cross_edges / total_edges * 100.0) if total_edges else 0:.1f}%)")

        if cross_edges > 0:
            # Windows 控制台默认编码在某些环境下不支持 emoji，避免 UnicodeEncodeError
            print(f"  [警告] 存在 {cross_edges} 条跨集边，将使用 Inductive 模式屏蔽")
        else:
            print("  [OK] 无跨集边")

    def _train(self, train_set, fold):
        """三分类单阶段训练"""
        if isinstance(train_set, torch.utils.data.Subset):
            train_sample_indices = train_set.indices
        else:
            train_sample_indices = list(range(len(train_set)))

        if not self.dataset.ppi_extractor._inductive_mode:
            train_genes = self._get_gene_indices_from_samples(train_sample_indices)
            self.dataset.ppi_extractor.set_train_mask(train_genes)

        labels = [self.dataset[i]["label"].item() for i in train_sample_indices]
        class_counts = np.bincount(labels, minlength=3)
        class_weights = 1.0 / torch.Tensor(class_counts + 1e-6)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, sampler=sampler)

        val_size = int(self.config.val_ratio * len(train_set))
        train_subset, val_subset = torch.utils.data.random_split(
            train_set, [len(train_set) - val_size, val_size]
        )
        val_loader = DataLoader(val_subset, batch_size=self.config.batch_size, shuffle=False)

        model = self.model_class(
            seq_dim=self.seq_dim,
            emb_dim=self.config.emb_dim,
            num_classes=self.config.num_classes,
            expr_d_model=self.config.expr_d_model,
            expr_pooled_len=self.config.expr_pooled_len,
            expr_nhead=self.config.expr_nhead,
            expr_num_layers=self.config.expr_num_layers,
            expr_dim_ff=self.config.expr_dim_ff,
            expr_dropout=self.config.expr_dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            threshold=0.0001,
            min_lr=1e-5,
        )

        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = self.config.patience

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                inputs = self._prepare_batch(batch)
                labels_batch = batch["label"].long().to(self.device)

                outputs = model(**inputs)
                loss = criterion(outputs, labels_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            val_acc, val_loss = self._validate(model, val_loader)
            scheduler.step(val_acc)

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {epoch_loss / max(len(train_loader), 1):.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                fold_path = os.path.join(self.output_dir, f"best_model_fold{fold}.pth")
                torch.save(model.state_dict(), fold_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  [Early Stop] 在 epoch {epoch + 1} 提前停止")
                    break

        return model

    def _validate(self, model, val_loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in val_loader:
                inputs = self._prepare_batch(batch)
                labels = batch["label"].long().to(self.device)

                outputs = model(**inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / max(total, 1), total_loss / max(len(val_loader), 1)

    def evaluate(self, model, test_set, train_genes=None, test_genes=None):
        """评估 Inductive 模型（测试时使用 inductive inference 重新计算 PPI 节点特征）"""
        if train_genes is not None and test_genes is not None:
            print("[Evaluate] 使用Inductive推理模式计算特征...")
            self.dataset.ppi_extractor.refresh_cache()
            inductive_features = self.dataset.ppi_extractor.extract_inductive_inference(train_genes, test_genes)
            self.dataset.ppi_extractor._cached_features = inductive_features

        test_loader = DataLoader(test_set, batch_size=self.config.batch_size)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = self._prepare_batch(batch)
                labels = batch["label"].cpu().numpy()

                outputs = model(**inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        if train_genes is not None and test_genes is not None:
            self.dataset.ppi_extractor.refresh_cache()

        metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.vstack(all_probs))
        return metrics

    def _prepare_batch(self, batch):
        return {
            "exp_tf": batch["tf"].to(self.device),
            "exp_target": batch["target"].to(self.device),
            "seq_tf": batch["seq_tf"].to(self.device),
            "seq_target": batch["seq_target"].to(self.device),
            "ppi_tf": batch["ppi_tf"].to(self.device),
            "ppi_target": batch["ppi_target"].to(self.device),
        }

    def _get_labels(self):
        return np.array([self.dataset[i]["label"].item() for i in range(len(self.dataset))])

    def predict_all(self):
        """全数据集预测（Transductive模式，使用全图信息）"""
        if self.best_model_path is None:
            raise FileNotFoundError("未找到最佳模型路径，请先运行 cross_validate() 训练模型。")

        self.dataset.ppi_extractor.clear_train_mask()
        self.dataset.ppi_extractor.refresh_cache()

        full_loader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=False)

        model = self.model_class(
            seq_dim=self.seq_dim,
            emb_dim=self.config.emb_dim,
            num_classes=self.config.num_classes,
            expr_d_model=self.config.expr_d_model,
            expr_pooled_len=self.config.expr_pooled_len,
            expr_nhead=self.config.expr_nhead,
            expr_num_layers=self.config.expr_num_layers,
            expr_dim_ff=self.config.expr_dim_ff,
            expr_dropout=self.config.expr_dropout,
        ).to(self.device)
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()

        all_preds, all_labels, all_probs, all_gene_pairs = [], [], [], []

        with torch.no_grad():
            for batch in full_loader:
                inputs = self._prepare_batch(batch)
                labels = batch["label"].cpu().numpy()

                tf_idx = batch["tf_idx"].cpu().numpy()
                target_idx = batch["target_idx"].cpu().numpy()
                tf_names = [self.dataset.gene_list[i] for i in tf_idx]
                target_names = [self.dataset.gene_list[i] for i in target_idx]
                gene_pairs = list(zip(tf_names, target_names))
                all_gene_pairs.extend(gene_pairs)

                outputs = model(**inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_probs.append(probs)
                all_preds.extend(preds)
                all_labels.extend(labels)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.vstack(all_probs)
        metrics = calculate_metrics(all_labels, all_preds, all_probs)

        return {
            "gene_pairs": all_gene_pairs,
            "y_true": all_labels,
            "y_pred": all_preds,
            "y_prob": all_probs,
            "metrics": metrics,
        }

    def save_full_prediction_csv(self, result_dict, csv_path: str):
        import pandas as pd

        df = pd.DataFrame(
            {
                "TF": [p[0] for p in result_dict["gene_pairs"]],
                "Target": [p[1] for p in result_dict["gene_pairs"]],
                "True_Label": result_dict["y_true"],
                "Pred_Label": result_dict["y_pred"],
                "Prob_NoReg": result_dict["y_prob"][:, 0],
                "Prob_Repress": result_dict["y_prob"][:, 1],
                "Prob_Activate": result_dict["y_prob"][:, 2],
            }
        )
        df.to_csv(csv_path, index=False)
        print(f"已保存完整预测结果：{csv_path}")

