#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig.3 bulk singles (manuscript Figure 4 A–F style bars) as PDF."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "fig03_bulk_cross_cancer" / "figures" / "main",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "grn.scripts.plot_bulk_tcga_10cv_metric_bars",
        "--out-dir",
        str(args.out_dir),
    ]
    print("[fig03] RUN", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
