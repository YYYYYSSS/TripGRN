import torch

from grn.models.ppi_gcn_cascade import PpiGcnTwoStageCascade, build_inductive_message_edges


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    num_nodes = 8
    T = 6
    seq_dim = 20

    # 小模型，避免注意力开销
    model = PpiGcnTwoStageCascade(
        emb_dim=16,
        expr_d_model=16,
        expr_pooled_len=16,
        expr_nhead=4,
        expr_num_layers=1,
        expr_dim_ff=32,
        expr_dropout=0.1,
        gcn_layers=2,
        gcn_dropout=0.1,
        head_hidden_dim=32,
        head_dropout=0.1,
    ).to(device)

    exp_all = torch.randn(num_nodes, T, device=device)
    cds_all = torch.randn(num_nodes, seq_dim, device=device)

    # dummy ppi edge_index: [2, E]
    base_edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 0],
        ],
        dtype=torch.long,
        device=device,
    )

    # undirected edges
    edge_index = torch.cat([base_edge_index, base_edge_index.flip(0)], dim=1)

    train_nodes = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    src, dst, norm = build_inductive_message_edges(
        base_edge_index=edge_index,
        num_nodes=num_nodes,
        train_nodes=train_nodes,
        mode="train",
        add_self_loops=True,
    )

    h = model.encode_nodes(
        exp_all=exp_all,
        cds_all=cds_all,
        src=src,
        dst=dst,
        edge_norm=norm,
        chunk_size=4,
    )
    assert h.shape == (num_nodes, 16)

    # dummy edge list
    tf_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    tg_idx = torch.tensor([4, 5, 6, 7], dtype=torch.long, device=device)

    logitsA = model.stage_a_logits(h, tf_idx, tg_idx)
    assert logitsA.shape == (4, 2)

    logitsB = model.stage_b_logits(
        h,
        tf_idx=tf_idx,
        target_idx=tg_idx,
        logits_stage_a=logitsA,
    )
    assert logitsB.shape == (4, 2)

    print("ppi_gcn_cascade smoke OK")


if __name__ == "__main__":
    main()

