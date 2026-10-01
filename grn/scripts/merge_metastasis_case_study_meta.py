"""
将 case_study 目录下 Python 的 case_study_meta.json 与 R 生成的
enrichment_summary.json、cellchat_partial_validation_summary.json 合并为 case_study_meta_merged.json。
由一站式 bat 在 R 步骤之后调用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Merge case study sidecar JSON into one manifest.")
    p.add_argument("--case-study-dir", required=True, help="例如 experiments/metastasis_case_study_brain_nb_full")
    args = p.parse_args()
    d = Path(args.case_study_dir).resolve()
    base_path = d / "case_study_meta.json"
    if not base_path.is_file():
        raise SystemExit(f"missing {base_path}")

    merged = json.loads(base_path.read_text(encoding="utf-8"))

    enr = d / "enrichment_summary.json"
    if enr.is_file():
        merged["enrichment"] = json.loads(enr.read_text(encoding="utf-8"))

    cc = d / "cellchat_partial_validation_summary.json"
    if cc.is_file():
        merged["cellchat_partial_validation"] = json.loads(cc.read_text(encoding="utf-8"))

    out = d / "case_study_meta_merged.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge-meta] wrote {out}")


if __name__ == "__main__":
    main()
