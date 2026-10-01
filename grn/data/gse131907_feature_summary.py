"""读取 GSE131907 Feature Summary xlsx：样本与 mBrain / 非脑转移标注（无 torch 依赖）。"""

from __future__ import annotations

import os
import tempfile
import zipfile
from contextlib import contextmanager
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import openpyxl


@contextmanager
def open_feature_summary_workbook(path: str) -> Iterator:
    """跳过部分 xlsx 中损坏的 docProps/custom.xml。"""
    fd, tmp_path = tempfile.mkstemp(suffix="_gse131907_fs.xlsx")
    os.close(fd)
    try:
        with zipfile.ZipFile(path, "r") as zin:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "docProps/custom.xml":
                        continue
                    zout.writestr(info, zin.read(info.filename))
        wb = openpyxl.load_workbook(tmp_path, read_only=True, data_only=True)
        try:
            yield wb
        finally:
            wb.close()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_sample_tissue_pairs(path: str) -> List[Tuple[str, str]]:
    """返回 (Samples, Tissue origins) 列表。"""
    out: List[Tuple[str, str]] = []
    with open_feature_summary_workbook(path) as wb:
        ws = wb.active
        rows_iter = ws.iter_rows(min_row=3, max_row=3, values_only=True)
        header_row = next(rows_iter)
        if not header_row:
            raise ValueError("empty header row (row 3)")
        headers = [str(h).strip() if h is not None else "" for h in header_row]
        try:
            i_samples = headers.index("Samples")
            i_tissue = headers.index("Tissue origins")
        except ValueError as e:
            raise ValueError(f"need columns 'Samples' and 'Tissue origins', got {headers}") from e
        for row in ws.iter_rows(min_row=4, values_only=True):
            if not row or row[0] is None:
                break
            first = str(row[0]).strip()
            if first.startswith("*"):
                break
            sample = row[i_samples]
            tissue = row[i_tissue]
            if sample is None or tissue is None:
                continue
            sample_s = str(sample).strip()
            tissue_s = str(tissue).strip()
            if sample_s:
                out.append((sample_s, tissue_s))
    return out


def mbrain_sample_ids_from_feature_summary(path: str) -> Set[str]:
    pairs = load_sample_tissue_pairs(path)
    return {s for s, t in pairs if t == "mBrain"}


def match_column_to_sample_id(col_name: str, sample_ids: Sequence[str]) -> Optional[str]:
    s = str(col_name).strip()
    ordered = sorted(sample_ids, key=len, reverse=True)
    for sid in ordered:
        suffix = "_" + sid
        if s.endswith(suffix):
            return sid
    return None
