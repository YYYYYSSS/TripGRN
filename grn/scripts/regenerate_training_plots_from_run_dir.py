"""
从已有 unseen_samples_{val,test}_metrics.json 重画 plots/（不重新训练）。

会更新：
  - metrics_val_vs_test.png（含 Macro F1：旧 json 会从 confusion_matrix 回填）
  - per_class_accuracy_precision_recall.png（文件名沿用旧习惯；三柱为 Recall / Precision / F1）

不会重画：learning_curve、accuracy_bar（需要 epoch 历史或仅标量参数）。
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

from grn.evaluation.training_plots import (
    per_class_ovr_accuracy_precision_recall_from_cm,
    save_grouped_metrics_val_test,
    save_per_class_metric_bars,
)


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_run_dirs(root: Path, max_depth: int = 4) -> List[Path]:
    """在 root 下 BFS，收集同时含 val/test unseen metrics 的目录（含单层 benchmark 与嵌套 ablation）。"""
    out: List[Path] = []
    seen: Set[Path] = set()
    dq: deque[tuple[Path, int]] = deque([(root.resolve(), 0)])
    while dq:
        p, d = dq.popleft()
        if not p.is_dir() or p in seen:
            continue
        seen.add(p)
        if (p / "unseen_samples_test_metrics.json").is_file() and (
            p / "unseen_samples_val_metrics.json"
        ).is_file():
            out.append(p)
        if d >= max_depth:
            continue
        for ch in sorted(p.iterdir()):
            if ch.is_dir():
                dq.append((ch, d + 1))
    return sorted(out, key=lambda x: str(x))


def regenerate_one(run_dir: Path) -> None:
    val_path = run_dir / "unseen_samples_val_metrics.json"
    te_path = run_dir / "unseen_samples_test_metrics.json"
    if not val_path.is_file() or not te_path.is_file():
        print(f"[regen-plots] skip (missing val/test metrics): {run_dir}")
        return
    val_m = _load_metrics(val_path)
    te_m = _load_metrics(te_path)
    plot_dir = str(run_dir / "plots")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    save_grouped_metrics_val_test(plot_dir, val_m, te_m)

    cm = te_m.get("confusion_matrix")
    if cm is None:
        print(f"[regen-plots] skip per-class (no confusion_matrix): {run_dir}")
        return
    acc_ovr, precision_pc, recall_pc = per_class_ovr_accuracy_precision_recall_from_cm(np.asarray(cm))
    disp_names = ["No regulation", "Activation", "Repression"]
    disp_idx = [0, 2, 1]
    save_per_class_metric_bars(
        plot_dir,
        disp_names,
        [float(acc_ovr[i]) for i in disp_idx],
        [float(precision_pc[i]) for i in disp_idx],
        [float(recall_pc[i]) for i in disp_idx],
    )
    print(f"[regen-plots] OK -> {plot_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="单个训练输出目录（可重复指定）。",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="扫描该目录树下（浅层 BFS）所有同时含 unseen_samples_val/test_metrics.json 的运行目录。",
    )
    args = ap.parse_args()

    todo: List[Path] = []
    for p in args.run_dir:
        todo.append(p.resolve())
    if args.root is not None:
        root = args.root.resolve()
        if not root.is_dir():
            raise SystemExit(f"--root 不是目录: {root}")
        todo.extend(_discover_run_dirs(root))

    if not todo:
        raise SystemExit("请指定 --run-dir 或 --root。")

    seen = set()
    for rd in todo:
        if rd in seen:
            continue
        seen.add(rd)
        print(f"[regen-plots] --- {rd}")
        regenerate_one(rd)


if __name__ == "__main__":
    main()
