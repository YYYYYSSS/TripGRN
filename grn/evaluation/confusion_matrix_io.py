# -*- coding: utf-8 -*-
"""Export confusion matrix JSON + figure for pancancer method benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from grn.evaluation.training_plots import save_confusion_matrix_fig


def cm_to_jsonable(cm: np.ndarray) -> List[List[int]]:
    arr = np.asarray(cm, dtype=np.int64)
    return [[int(x) for x in row] for row in arr.tolist()]


def export_confusion_matrix_artifacts(
    out_dir: Union[str, Path],
    cm: np.ndarray,
    *,
    stem: str = "confusion_matrix_test",
    title: str = "Test confusion matrix",
    class_names: Sequence[str],
    extra: Optional[Mapping[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write ``{stem}.json`` and ``{stem}.png`` (+ pdf) under *out_dir*."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm_arr = np.asarray(cm, dtype=np.int64)
    payload: Dict[str, Any] = {
        "confusion_matrix": cm_to_jsonable(cm_arr),
        "class_names": list(class_names),
        "n_samples": int(cm_arr.sum()),
    }
    if extra:
        payload.update(dict(extra))
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    png_path = Path(
        save_confusion_matrix_fig(
            str(out),
            cm_arr,
            title,
            f"{stem}.png",
            class_names=class_names,
        )
    )
    return json_path, png_path
