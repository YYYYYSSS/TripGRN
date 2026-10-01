import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from grn.scripts.train_metastasis_nonbrain_brain_eval import (
    FusionModel,
    FusionModelAblation,
    PairSubsetColumnDataset,
    _build_metastasis_eval_tensors,
    _collect_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    cfg_path = os.path.join(run_dir, "run_config_master.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("run_dir", None)

    eval_args = SimpleNamespace(**cfg)

    ctx = _build_metastasis_eval_tensors(eval_args, run_dir, write_subset_and_manifests=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModelAblation(
        seq_dim=ctx["seq_dim"],
        num_classes=3,
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        use_ppi=ctx["use_ppi"],
    ).to(device)
    ckpt = os.path.join(run_dir, "best_model_nonbrain_train_brain_extval_fold0.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    batch_size = int(cfg["batch_size"])

    ds_nb = PairSubsetColumnDataset(
        ctx["exp_te"],
        ctx["pairs_test"],
        ctx["seq_features"],
        ctx["ppi_extractor"],
        device=None,
        use_ppi=ctx["use_ppi"],
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        ppi_feat_dim=ctx["ppi_emb_dim"],
    )
    ds_br = PairSubsetColumnDataset(
        ctx["exp_brain_full"],
        ctx["pairs_test"],
        ctx["seq_features"],
        ctx["ppi_extractor"],
        device=None,
        use_ppi=ctx["use_ppi"],
        use_expression=ctx["use_expr"],
        use_cds=ctx["use_cds_modal"],
        ppi_feat_dim=ctx["ppi_emb_dim"],
    )

    ln = DataLoader(ds_nb, batch_size=batch_size, shuffle=False)
    lb = DataLoader(ds_br, batch_size=batch_size, shuffle=False)

    yt, pred_nb, prob_nb, tf_nb, tg_nb = _collect_predictions(
        model, ln, device, ctx["gene_list"]
    )
    yt2, pred_br, prob_br, tf_br, tg_br = _collect_predictions(
        model, lb, device, ctx["gene_list"]
    )

    assert np.array_equal(yt, yt2)
    assert tf_nb == tf_br and tg_nb == tg_br

    argmax_diff_edges = int((pred_nb != pred_br).sum())
    print(f"pairs_test={len(yt)} argmax_diff_edges={argmax_diff_edges}")

    abs_diff = np.abs(prob_nb - prob_br)
    max_diff = abs_diff.max(axis=1)  # per edge, max class-prob diff

    for th in [1e-3, 1e-2, 5e-2, 1e-1]:
        print(f"edges with max|p_nb-p_br| > {th}: {int((max_diff > th).sum())}")

    topk = min(args.topk, len(max_diff))
    idx = np.argsort(-max_diff)[:topk]
    for rank, i in enumerate(idx, start=1):
        print(
            f"[{rank}/{topk}] TF={tf_nb[i]} Target={tg_nb[i]} True={int(yt[i])} "
            f"Pred_nb={int(pred_nb[i])} Pred_br={int(pred_br[i])} max_diff={float(max_diff[i]):.6f}"
        )
        print(f"   p_nb={np.round(prob_nb[i], 6).tolist()}")
        print(f"   p_br={np.round(prob_br[i], 6).tolist()}")

    # Extra diagnostic: compare expression embeddings for a small gene subset.
    # If embeddings are (nearly) identical, then the downstream prediction parity is expected.
    n_check = min(200, ctx["exp_te"].shape[0])
    gene_idx = list(range(n_check))
    with torch.no_grad():
        exp_te_sub = ctx["exp_te"][gene_idx].to(device)
        exp_br_sub = ctx["exp_brain_full"][gene_idx].to(device)
        emb_nb = model.exp_encoder_tf(exp_te_sub)
        emb_br = model.exp_encoder_tf(exp_br_sub)
        emb_diff = (emb_nb - emb_br).abs().detach().cpu().numpy()
    print(f"\nEmbedding diff (Expression encoder TF) on first {n_check} genes:")
    print("  max_abs_diff:", float(emb_diff.max()))
    print("  mean_abs_diff:", float(emb_diff.mean()))

    # Additional diagnostic: expression ablation (zero out exp branch)
    ds_nb_zeroexp = PairSubsetColumnDataset(
        ctx["exp_te"],
        ctx["pairs_test"],
        ctx["seq_features"],
        ctx["ppi_extractor"],
        device=None,
        use_ppi=ctx["use_ppi"],
        use_expression=False,
        use_cds=ctx["use_cds_modal"],
        ppi_feat_dim=ctx["ppi_emb_dim"],
    )
    ds_br_zeroexp = PairSubsetColumnDataset(
        ctx["exp_brain_full"],
        ctx["pairs_test"],
        ctx["seq_features"],
        ctx["ppi_extractor"],
        device=None,
        use_ppi=ctx["use_ppi"],
        use_expression=False,
        use_cds=ctx["use_cds_modal"],
        ppi_feat_dim=ctx["ppi_emb_dim"],
    )
    ln2 = DataLoader(ds_nb_zeroexp, batch_size=batch_size, shuffle=False)
    lb2 = DataLoader(ds_br_zeroexp, batch_size=batch_size, shuffle=False)
    _, pred_nb_z, prob_nb_z, _, _ = _collect_predictions(model, ln2, device, ctx["gene_list"])
    _, pred_br_z, prob_br_z, _, _ = _collect_predictions(model, lb2, device, ctx["gene_list"])

    print("\nAblation check (expression inputs zeroed):")
    print("  argmax_diff_edges:", int((pred_nb_z != pred_br_z).sum()))
    abs_diff_z = np.abs(prob_nb_z - prob_br_z)
    print("  max|p_nb-p_br|:", float(abs_diff_z.max()))
    max_diff_z = abs_diff_z.max(axis=1)
    print("  edges with max|p_nb-p_br| > 1e-3:", int((max_diff_z > 1e-3).sum()))

    # Contribution check: does expression affect absolute predictions on each side?
    abs_diff_enabled_vs_zero_nb = np.abs(prob_nb - prob_nb_z)
    abs_diff_enabled_vs_zero_br = np.abs(prob_br - prob_br_z)
    print("\nContribution check (expression enabled vs zeroed on the SAME side):")
    print("  Non-brain enabled vs zeroed: max|Δp|=", float(abs_diff_enabled_vs_zero_nb.max()))
    print("  Brain enabled vs zeroed:      max|Δp|=", float(abs_diff_enabled_vs_zero_br.max()))

    # Parameter-level diagnostic: how much FusionModel's first fusion layer
    # uses expression vs seq vs ppi blocks.
    try:
        emb_dim = int(ctx["seq_dim"] * 0 + model.emb_dim)  # avoid accidental type issues
        w0 = model.fc_fusion[0].weight.detach().cpu()  # [hidden, 6*emb]
        # Feature concat order in FusionModel:
        # [exp_tf, exp_target, seq_tf, seq_target, ppi_tf, ppi_target]
        block = emb_dim
        expr_cols = w0[:, 0 : 2 * block]
        seq_cols = w0[:, 2 * block : 4 * block]
        ppi_cols = w0[:, 4 * block : 6 * block]

        expr_norm = float(expr_cols.norm().item())
        seq_norm = float(seq_cols.norm().item())
        ppi_norm = float(ppi_cols.norm().item())
        total = float(w0.norm().item())
        print("\nParameter diagnostic (fc_fusion first layer column norms):")
        print("  expr_norm=", expr_norm, " seq_norm=", seq_norm, " ppi_norm=", ppi_norm, " total_norm=", total)
        if seq_norm > 0:
            print("  expr/seq norm ratio=", expr_norm / seq_norm)
        if ppi_norm > 0:
            print("  expr/ppi norm ratio=", expr_norm / ppi_norm)
    except Exception as e:
        print("\nParameter diagnostic skipped due to:", repr(e))


if __name__ == "__main__":
    main()

