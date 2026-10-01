# -*- coding: utf-8 -*-
"""Full-cell Fig.6A UMAP without downsampling (memory-safe chunked PCA)."""
from __future__ import annotations

import gc
import sys
import types
from pathlib import Path

# Bypass umap -> torchvision CUDA mismatch in this env
sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))

import faulthandler

faulthandler.enable()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
NPY = ROOT / "data" / "exp" / "GSE131907" / "precomputed_tlung_mbrain_epi_npy"
EXP = ROOT / "experiments" / "gse131907_transfer_from_gse253718_epi_seed2"
FIG = EXP / "case_tlung_vs_mbrain_kegg_khop_k1" / "figures"


def main() -> None:
    print("load npy", flush=True)
    xt = np.load(NPY / "tlung_values_log_normalized.npy", mmap_mode="r")  # genes x cells
    xb = np.load(NPY / "mbrain_values_log_normalized.npy", mmap_mode="r")
    nt, nb = int(xt.shape[1]), int(xb.shape[1])
    n = nt + nb
    g = int(xt.shape[0])
    print(f"cells tLung={nt} mBrain={nb} total={n} genes={g}", flush=True)

    acc = np.zeros(g, dtype=np.float64)
    acc2 = np.zeros(g, dtype=np.float64)
    for src, ncell, tag in ((xt, nt, "tLung"), (xb, nb, "mBrain")):
        for j in range(0, ncell, 512):
            block = np.asarray(src[:, j : j + 512].T, dtype=np.float64)
            acc += block.sum(axis=0)
            acc2 += np.square(block).sum(axis=0)
        print(f"var done {tag}", flush=True)
    var = np.maximum(acc2 / n - (acc / n) ** 2, 0.0)
    top = np.argsort(var)[::-1][:2000]
    del acc, acc2
    gc.collect()

    print("assemble HVG matrix float32", flush=True)
    M = np.empty((n, 2000), dtype=np.float32)
    M[:nt] = np.asarray(xt[top, :].T, dtype=np.float32)
    M[nt:] = np.asarray(xb[top, :].T, dtype=np.float32)
    labs = np.array(["tLung"] * nt + ["mBrain"] * nb)
    names = np.array([f"c{i}" for i in range(n)])
    del xt, xb
    gc.collect()

    mu = M.mean(axis=0, dtype=np.float64)
    sd = M.std(axis=0, dtype=np.float64)
    sd[sd == 0] = 1.0
    M = (M - mu.astype(np.float32)) / sd.astype(np.float32)
    np.clip(M, -10, 10, out=M)
    print("scaled", M.shape, M.dtype, flush=True)

    print("chunked gene-covariance (torch CPU)", flush=True)
    import torch

    torch.set_num_threads(1)
    G = torch.zeros((2000, 2000), dtype=torch.float64)
    bs = 256
    for i in range(0, n, bs):
        xb = torch.from_numpy(np.asarray(M[i : i + bs], dtype=np.float64))
        G += xb.T @ xb
        if i % 2048 == 0:
            print(f"  gram {i}/{n}", flush=True)
    G /= max(n - 1, 1)
    print("top-k eigen via subspace iteration (torch matmul only)", flush=True)
    # Avoid broken LAPACK eigh/SVD in this env; use blocked subspace iteration on G.
    k = 50
    p = 2000
    rng = np.random.default_rng(2)
    Q = torch.from_numpy(rng.standard_normal((p, k)).astype(np.float64))
    Q, _ = torch.linalg.qr(Q, mode="reduced")
    for it in range(8):
        Z = G @ Q
        Q, _ = torch.linalg.qr(Z, mode="reduced")
        print(f"  subspace iter {it+1}/8", flush=True)
    # Rayleigh-Ritz in k-dim using torch (numpy.linalg.eigh crashes in this env)
    B = Q.T @ G @ Q
    evals, evecs = torch.linalg.eigh(B)
    order = torch.argsort(evals, descending=True)
    V = Q @ evecs[:, order]  # p x k torch
    print("project cells (torch)", flush=True)
    pcs = np.empty((n, k), dtype=np.float64)
    for i in range(0, n, bs):
        xb = torch.from_numpy(np.asarray(M[i : i + bs], dtype=np.float64))
        pcs[i : i + bs] = (xb @ V).numpy()
        if i % 2048 == 0:
            print(f"  proj {i}/{n}", flush=True)
    del M, G, Q, B, V, evals, evecs
    gc.collect()

    print("UMAP on all cells", flush=True)
    # Pass a/b explicitly: scipy.optimize.curve_fit crashes in this env (0xc06d007f).
    # Values approximate umap.find_ab_params(spread=1.0, min_dist=0.3).
    emb = UMAP(
        n_neighbors=15,
        min_dist=0.3,
        random_state=2,
        low_memory=True,
        a=1.8955812355331685,
        b=0.8007570911633017,
        init="random",  # skip spectral init (numpy.linalg.norm crashes in this env)
    ).fit_transform(pcs[:, :30].astype(np.float32))
    coords = pd.DataFrame(
        {
            "cell_idx": names,
            "compartment": labs,
            "umap1": emb[:, 0],
            "umap2": emb[:, 1],
        }
    )
    print(coords["compartment"].value_counts().to_string(), flush=True)
    coords_path = EXP / "umap_tlung_mbrain_epi_coords.csv"
    coords.to_csv(coords_path, index=False, encoding="utf-8-sig")
    print("wrote", coords_path, "n=", len(coords), flush=True)

    # Matplotlib savefig crashes in this env (broken numpy/LAPACK); draw via PIL.
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas as pdf_canvas

    W, H, margin = 1400, 1200, 80
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    xv = coords["umap1"].to_numpy()
    yv = coords["umap2"].to_numpy()
    xmin, xmax = float(xv.min()), float(xv.max())
    ymin, ymax = float(yv.min()), float(yv.max())
    dx, dy = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    xmin, xmax, ymin, ymax = xmin - dx, xmax + dx, ymin - dy, ymax + dy

    def to_px(xx: float, yy: float) -> tuple[int, int]:
        px = margin + int((xx - xmin) / (xmax - xmin + 1e-12) * (W - 2 * margin))
        py = H - margin - int((yy - ymin) / (ymax - ymin + 1e-12) * (H - 2 * margin))
        return px, py

    colors = {"tLung": (93, 173, 226), "mBrain": (231, 76, 60)}
    labs_arr = coords["compartment"].to_numpy()
    for i in np.random.default_rng(0).permutation(len(coords)):
        px, py = to_px(float(xv[i]), float(yv[i]))
        c = colors[str(labs_arr[i])]
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=c)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
        font_s = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        font_s = font
    draw.text(
        (margin, 20),
        "GSE131907 epithelial cells: tLung vs mBrain (Fig. 6A)",
        fill=(0, 0, 0),
        font=font,
    )
    lx, ly = margin, H - 55
    for lab, col in colors.items():
        nlab = int((labs_arr == lab).sum())
        draw.ellipse((lx, ly - 8, lx + 14, ly + 6), fill=col)
        draw.text((lx + 22, ly - 12), f"{lab} (n={nlab:,})", fill=(0, 0, 0), font=font_s)
        lx += 280
    draw.text((40, H // 2), "UMAP2", fill=(80, 80, 80), font=font_s)
    draw.text((W // 2 - 40, H - 30), "UMAP1", fill=(80, 80, 80), font=font_s)

    FIG.mkdir(parents=True, exist_ok=True)
    out_png = FIG / "fig6A_tlung_mbrain_epi_umap.png"
    out_pdf = FIG / "fig6A_tlung_mbrain_epi_umap.pdf"
    img.save(out_png, dpi=(300, 300))
    c = pdf_canvas.Canvas(str(out_pdf), pagesize=(620, 540))
    c.drawImage(ImageReader(str(out_png)), 10, 10, width=600, height=520, preserveAspectRatio=True, anchor="c")
    c.save()
    print("DONE", out_pdf, "n=", len(coords), flush=True)


if __name__ == "__main__":
    main()
