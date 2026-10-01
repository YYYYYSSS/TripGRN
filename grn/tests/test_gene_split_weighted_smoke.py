"""Smoke test for weighted gene split (run: conda run -n project_GRN python -m grn.tests.test_gene_split_weighted_smoke)."""
import os
import tempfile

import torch

from grn.data.gene_split_weighted import (
    assignment_to_gene_lists,
    load_gene_split_json,
    optimize_gene_split_weighted,
    regulated_edges_from_grn_matrix,
    save_gene_split_json,
)


def main():
    G = torch.zeros(8, 8, dtype=torch.long)
    G[0, 1] = 1
    G[0, 2] = 2
    G[3, 4] = 1
    G[5, 6] = 2
    G[6, 7] = 1
    tf_i, tg_i, is_r = regulated_edges_from_grn_matrix(G)
    asg, meta = optimize_gene_split_weighted(
        8, (3, 2, 3), tf_i, tg_i, is_r, 1.0, 0.25, seed=7, steps=2000, n_random_starts=2
    )
    tr, va, te = assignment_to_gene_lists(asg)
    assert len(tr) == 3 and len(va) == 2 and len(te) == 3
    assert len(set(tr) | set(va) | set(te)) == 8

    fd, p = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    genes = [f"G{i}" for i in range(8)]
    try:
        save_gene_split_json(p, genes, asg, (3, 2, 3), "3,2,3", extra=meta)
        tr2, va2, te2, _m2 = load_gene_split_json(p, genes, (3, 2, 3))
        assert set(tr2) == set(tr) and set(va2) == set(va) and set(te2) == set(te)
    finally:
        os.remove(p)
    print("gene_split_weighted smoke OK", meta)


if __name__ == "__main__":
    main()
