#!/usr/bin/env python3
"""Generate detailed TripGRN modality feature-extractor schematics as PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import rcParams

OUT_DIR = Path(r"C:\Users\14842\Desktop\示意图")

# Academic palette (avoid purple-on-white AI defaults)
C = {
    "bg": "#F7F8FA",
    "ink": "#1F2933",
    "muted": "#52606D",
    "line": "#3E4C59",
    "expr": "#0B6E4F",
    "expr_fill": "#E3F5EE",
    "cds": "#0B4F6C",
    "cds_fill": "#E6F2F7",
    "ppi": "#8A4B08",
    "ppi_fill": "#F8EEDF",
    "shared": "#334E68",
    "shared_fill": "#E8EEF5",
    "out": "#9B2226",
    "out_fill": "#FCE8E6",
    "note": "#F0F4F8",
    "white": "#FFFFFF",
}


def _setup_fonts() -> None:
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


def _new_fig(w: float = 14.0, h: float = 9.5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=C["bg"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    return fig, ax


def _box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    fc=C["white"],
    ec=C["line"],
    lw=1.4,
    fontsize=9,
    weight="normal",
    va="center",
    radius=0.012,
    text_color=None,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        mutation_aspect=0.35,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va=va,
        fontsize=fontsize,
        color=text_color or C["ink"],
        fontweight=weight,
        zorder=3,
        linespacing=1.25,
        wrap=False,
    )
    return patch


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.5):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color or C["line"],
        lw=lw,
        zorder=1,
    )
    ax.add_patch(arr)


def _vchain(ax, cx, tops, bottoms, color):
    for y1, y2 in zip(tops, bottoms):
        _arrow(ax, cx, y1, cx, y2, color=color)


def _title(ax, title: str, subtitle: str = ""):
    ax.text(50, 96.5, title, ha="center", va="top", fontsize=16, fontweight="bold", color=C["ink"])
    if subtitle:
        ax.text(50, 92.8, subtitle, ha="center", va="top", fontsize=9.5, color=C["muted"])


def _footer(ax, text: str):
    ax.text(50, 1.8, text, ha="center", va="bottom", fontsize=7.5, color=C["muted"])


def draw_overview(ax):
    _title(
        ax,
        "TripGRN 多模态特征提取器总览",
        "对每个候选调控对 (TF → Target)，三模态各产出 TF/Target 两条嵌入，再拼接融合",
    )

    # Input
    _box(ax, 4, 78, 18, 10, "候选基因对\n(i, j)\nTF → Target", fc=C["note"], ec=C["shared"], fontsize=10, weight="bold")

    # Three modality columns
    cols = [
        (28, "表达模态 Expression", C["expr"], C["expr_fill"], "x_g ∈ R^T\n变长细胞/样本表达", "ExpressionSampleSetEncoder\n(TF / Target 独立参数)", "e_exp ∈ R^160"),
        (52, "CDS 模态 Sequence", C["cds"], C["cds_fill"], "s_g ∈ R^20\n预计算 CDS 特征", "ResNetBranch\n(TF / Target 独立参数)", "e_CDS ∈ R^160"),
        (76, "PPI 模态 Topology", C["ppi"], C["ppi_fill"], "h_g ∈ R^160\nGCN 节点嵌入", "ResNetBranch\n(TF / Target 独立参数)", "e_PPI ∈ R^160"),
    ]
    for x, name, ec, fc, inp, enc, out in cols:
        ax.add_patch(Rectangle((x - 1, 28), 22, 52, facecolor=fc, edgecolor=ec, lw=1.6, alpha=0.55, zorder=0))
        ax.text(x + 10, 76.5, name, ha="center", va="center", fontsize=10.5, fontweight="bold", color=ec)
        _box(ax, x, 64, 20, 9, inp, fc=C["white"], ec=ec, fontsize=8.5)
        _arrow(ax, x + 10, 64, x + 10, 55.5, color=ec)
        _box(ax, x, 42, 20, 13, enc, fc=C["white"], ec=ec, fontsize=8.5, weight="bold")
        _arrow(ax, x + 10, 42, x + 10, 35.5, color=ec)
        _box(ax, x, 29.5, 20, 6, out, fc=C["out_fill"], ec=C["out"], fontsize=9, weight="bold")

    _arrow(ax, 22, 83, 27.5, 70, color=C["shared"])
    _arrow(ax, 22, 83, 51.5, 70, color=C["shared"])
    _arrow(ax, 22, 83, 75.5, 70, color=C["shared"])

    # Fusion strip
    _box(
        ax,
        18,
        12,
        64,
        12,
        "六路嵌入拼接  [e_exp^TF ‖ e_exp^Tar ‖ e_CDS^TF ‖ e_CDS^Tar ‖ e_PPI^TF ‖ e_PPI^Tar]  ∈ R^960\n"
        "→ Fusion MLP (960→256→160) → Conv1d 分类头 → {NoReg, Repress, Activate}",
        fc=C["shared_fill"],
        ec=C["shared"],
        fontsize=8.8,
        weight="bold",
    )
    for x in (38, 62, 86):
        _arrow(ax, x, 29.5, 50, 24.2, color=C["shared"], lw=1.2)

    _footer(ax, "对应实现：grn/models/fusion_model.py · ExpressionSampleSetEncoder · ResNetBranch · PPIFeatureExtractor")


def draw_expression(ax):
    _title(
        ax,
        "表达模态特征提取器 — ExpressionSampleSetEncoder",
        "输入变长表达向量 [B, T]（T 为细胞/样本数，训练时常 subsample 至 10000），输出固定 [B, 160]",
    )

    color = C["expr"]
    fill = C["expr_fill"]

    # Left: pipeline
    stages = [
        (82, 10, "输入 x ∈ R^{B×T}\n基因在各细胞上的表达\n(无需固定长度)", fill),
        (70, 9, "unsqueeze → [B, 1, T]\n视作单通道 1D 序列", C["white"]),
        (
            56,
            11,
            "Stem\nConv1d(1→128, k=5, pad=2)\n+ BatchNorm1d + GELU + Dropout(0.1)",
            fill,
        ),
        (
            42,
            11,
            "残差块 Res1\nh ← h + Conv1d(128→128, k=3)\n+ BN + GELU + Dropout",
            C["white"],
        ),
        (
            28,
            11,
            "残差块 Res2\nh ← h + Conv1d(128→128, k=3)\n+ BN + GELU + Dropout",
            fill,
        ),
        (
            15,
            10,
            "双全局池化\nAdaptiveAvgPool1d(1) ‖ AdaptiveMaxPool1d(1)\n→ [B, 256]",
            C["white"],
        ),
        (4, 8, "Linear(256→160)\n输出 e_exp ∈ R^{B×160}", C["out_fill"]),
    ]
    x0, w = 6, 40
    for y, h, text, fc in stages:
        ec = C["out"] if "输出" in text else color
        _box(ax, x0, y, w, h, text, fc=fc, ec=ec, fontsize=8.3, lw=1.5 if "输出" in text else 1.3)
    ys = [s[0] for s in stages]
    hs = [s[1] for s in stages]
    for i in range(len(stages) - 1):
        _arrow(ax, x0 + w / 2, ys[i], x0 + w / 2, ys[i + 1] + hs[i + 1], color=color)

    # Right: design notes + dual branch
    _box(
        ax,
        52,
        68,
        44,
        18,
        "双分支设计（FusionModel）\n"
        "• exp_encoder_tf / exp_encoder_target：结构相同、参数独立\n"
        "• 对调控对 (i,j) 分别编码 TF 与 Target 表达向量\n"
        "• 默认超参：d_model=128, out_dim=160, dropout=0.1",
        fc=fill,
        ec=color,
        fontsize=8.8,
        weight="bold",
    )
    _box(
        ax,
        52,
        42,
        44,
        22,
        "设计要点\n"
        "• 用 1D 卷积在变长细胞轴上提取局部表达模式\n"
        "• 全局 avg∥max 池化得到与 T 无关的固定向量\n"
        "• 不依赖固定 pooled_len / 学习位置编码，利于泛化\n"
        "• 可选 gradient checkpointing 降低长序列显存\n"
        "• 训练侧可用 max_expression_cols=10000 子采样控显存",
        fc=C["white"],
        ec=color,
        fontsize=8.5,
    )
    _box(
        ax,
        52,
        14,
        44,
        24,
        "张量形状流\n"
        "[B, T] → [B, 1, T]\n"
        "→ Stem/Res： [B, 128, T]\n"
        "→ AvgPool / MaxPool：各 [B, 128]\n"
        "→ concat： [B, 256]\n"
        "→ Linear： [B, 160]\n\n"
        "实现：grn/models/expression_set_encoder.py",
        fc=C["note"],
        ec=C["shared"],
        fontsize=8.5,
    )
    _footer(ax, "角色：刻画基因在表达谱上的动态模式，为调控方向预测提供细胞状态语境")


def draw_cds(ax):
    _title(
        ax,
        "CDS 模态特征提取器 — 预计算序列特征 + ResNetBranch",
        "离线从编码序列得到 20 维基因特征，再经残差分支投影为 160 维嵌入",
    )
    color = C["cds"]
    fill = C["cds_fill"]

    # Offline precompute panel
    ax.add_patch(Rectangle((3, 52), 46, 36, facecolor=fill, edgecolor=color, lw=1.4, alpha=0.45, zorder=0))
    ax.text(26, 85.5, "离线预计算（不进入端到端训练）", ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    pre = [
        (76, 8, "Ensembl GRCh38 CDS\n基因编码序列", C["white"]),
        (65, 8, "k-mer 切分 (k=4)\n滑窗提取 4-mer", C["white"]),
        (54, 8, "独热编码\none-hot(4-mer)", C["white"]),
        (
            40,
            11,
            "BiGRU\nGRU(input=|vocab|, hidden=10,\nbidirectional=True)\n末层输出均值 → R^20",
            C["white"],
        ),
    ]
    for y, h, text, fc in pre:
        _box(ax, 6, y, 40, h, text, fc=fc, ec=color, fontsize=8.2)
    for i in range(len(pre) - 1):
        _arrow(ax, 26, pre[i][0], 26, pre[i + 1][0] + pre[i + 1][1], color=color)

    _box(ax, 54, 76, 42, 12, "预计算表\ncds_gene_feature_k4.csv\n每基因一行：s_g ∈ R^20", fc=C["white"], ec=color, fontsize=9, weight="bold")
    _arrow(ax, 46, 45.5, 54, 80, color=color)

    # Online ResNetBranch
    ax.add_patch(Rectangle((54, 8), 42, 62, facecolor=C["note"], edgecolor=C["shared"], lw=1.4, alpha=0.55, zorder=0))
    ax.text(75, 66.5, "在线编码：ResNetBranch (CDS)", ha="center", va="center", fontsize=10, fontweight="bold", color=C["shared"])

    online = [
        (56, 7, "输入 s ∈ R^{B×20}", C["white"]),
        (47, 7, "Linear(20→256)\n投影到定长轴", fill),
        (38, 7, "unsqueeze → [B,1,256]", C["white"]),
        (29, 7, "Stem: Conv1d(1→32,k=1)\n+ BN + ReLU", fill),
        (20, 7, "ResBlock1D(32)\n两层 1×1 Conv + 残差", C["white"]),
        (11.5, 7, "Transition: 32→64\n+ ResBlock1D(64)", fill),
        (3.5, 6.5, "Flatten → Linear(64×256→160)\ne_CDS ∈ R^{B×160}", C["out_fill"]),
    ]
    for y, h, text, fc in online:
        ec = C["out"] if "e_CDS" in text else color
        _box(ax, 57, y, 36, h, text, fc=fc, ec=ec, fontsize=7.8)
    for i in range(len(online) - 1):
        _arrow(ax, 75, online[i][0], 75, online[i + 1][0] + online[i + 1][1], color=color)

    _box(
        ax,
        3,
        8,
        46,
        40,
        "ResBlock1D 细节\n"
        "• 无下采样：两层 Conv1d(channels→channels, k=1)\n"
        "• 每层后 BN；块末 residual 相加再 ReLU\n"
        "• 用 flatten + 线性投影替代全局池化，保留定长轴信息\n\n"
        "双分支\n"
        "• seq_branch_tf / seq_branch_target 参数独立\n"
        "• Fusion 中贡献 2×160 维\n\n"
        "实现：extract_cds_features.py（预计算）\n"
        "      grn/models/resnet.py::ResNetBranch（在线）",
        fc=C["white"],
        ec=color,
        fontsize=8.3,
    )
    _footer(ax, "角色：注入基因编码序列组成信息，补充表达谱未覆盖的序列先验")


def draw_ppi(ax):
    _title(
        ax,
        "PPI 模态特征提取器 — STRING 图卷积嵌入 + ResNetBranch",
        "先用轻量 GCN 从蛋白互作拓扑得到节点嵌入，再经残差分支映射到融合空间",
    )
    color = C["ppi"]
    fill = C["ppi_fill"]

    # Graph build
    ax.add_patch(Rectangle((3, 58), 94, 30, facecolor=fill, edgecolor=color, lw=1.4, alpha=0.4, zorder=0))
    ax.text(50, 85.5, "图构建与节点初始化（PPIFeatureExtractor）", ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)

    _box(ax, 5, 68, 22, 12, "STRING v12\n人类 PPI 边表\nscore ≥ 900\n(from, to)", fc=C["white"], ec=color, fontsize=8.2)
    _box(ax, 30, 68, 22, 12, "映射到当前 gene_list\n去自环/无效边\n无向化邻接 A", fc=C["white"], ec=color, fontsize=8.2)
    _box(ax, 55, 68, 20, 12, "对称归一化\nÂ = D^{-1/2} A D^{-1/2}", fc=C["white"], ec=color, fontsize=8.2)
    _box(ax, 78, 68, 17, 12, "初始特征\nX = I_N\n（单位矩阵）", fc=C["white"], ec=color, fontsize=8.2)
    _arrow(ax, 27, 74, 30, 74, color=color)
    _arrow(ax, 52, 74, 55, 74, color=color)
    _arrow(ax, 75, 74, 78, 74, color=color)

    # GCN
    ax.add_patch(Rectangle((3, 28), 52, 27, facecolor=C["note"], edgecolor=C["shared"], lw=1.4, alpha=0.5, zorder=0))
    ax.text(29, 52.5, "PPINet：3 层图卷积", ha="center", va="center", fontsize=10, fontweight="bold", color=C["shared"])

    _box(ax, 5, 40, 15, 9, "GCN-1\nLinear(N→160)\nÂX → LN → ReLU\nDropout 0.2", fc=C["white"], ec=color, fontsize=7.5)
    _box(ax, 22, 40, 15, 9, "GCN-2\nLinear(160→160)\nÂH → LN → ReLU\nDropout 0.2", fc=C["white"], ec=color, fontsize=7.5)
    _box(ax, 39, 40, 14, 9, "GCN-3\nLinear(160→160)\nÂH → LN\n(无激活/无 Dropout)", fc=C["white"], ec=color, fontsize=7.5)
    _arrow(ax, 20, 44.5, 22, 44.5, color=color)
    _arrow(ax, 37, 44.5, 39, 44.5, color=color)
    _box(ax, 5, 30, 48, 7, "节点嵌入 H ∈ R^{N×160}（可缓存；支持 inductive 训练掩码防泄露）", fc=C["out_fill"], ec=C["out"], fontsize=8.2, weight="bold")

    # ResNet branch
    ax.add_patch(Rectangle((58, 8), 39, 47, facecolor=fill, edgecolor=color, lw=1.4, alpha=0.35, zorder=0))
    ax.text(77.5, 52.5, "在线编码：ResNetBranch (PPI)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)

    ppi_online = [
        (43, 6.5, "取节点 h_g ∈ R^160", C["white"]),
        (35, 6.5, "Linear(160→256)", C["white"]),
        (27, 6.5, "Stem Conv1d 1→32", C["white"]),
        (19, 6.5, "ResBlock(32) → Trans→64\n→ ResBlock(64)", C["white"]),
        (10, 7, "Flatten + Linear\n→ e_PPI ∈ R^160", C["out_fill"]),
    ]
    for y, h, text, fc in ppi_online:
        ec = C["out"] if "e_PPI" in text else color
        _box(ax, 61, y, 33, h, text, fc=fc, ec=ec, fontsize=7.8)
    for i in range(len(ppi_online) - 1):
        _arrow(ax, 77.5, ppi_online[i][0], 77.5, ppi_online[i + 1][0] + ppi_online[i + 1][1], color=color)

    _box(
        ax,
        3,
        8,
        52,
        17,
        "Inductive 防泄露（训练可选）\n"
        "• 训练：仅保留 train–train 子图消息传递，测试节点嵌入置零\n"
        "• 推理：允许 train→test 聚合，禁止 test→train / test→test\n"
        "• 配对时 get_pair_features(i,j) 取两端节点嵌入，再送入双分支\n"
        "实现：grn/features/ppi_features.py + ResNetBranch",
        fc=C["white"],
        ec=color,
        fontsize=8.0,
    )
    _footer(ax, "角色：注入蛋白互作拓扑先验，约束调控推断落在生物学互作邻域")


def draw_resnet_detail(ax):
    _title(
        ax,
        "共享残差分支结构 — ResNetBranch（CDS / PPI 共用）",
        "CDS：input_dim=20；PPI：input_dim=160；二者均映射到 output_dim=160，fixed_dim=256",
    )
    color = C["shared"]
    fill = C["shared_fill"]

    stages = [
        (5, 86, 18, 8, "输入向量\nR^{d_in}", C["white"]),
        (28, 86, 18, 8, "Linear\nd_in → 256", fill),
        (51, 86, 18, 8, "unsqueeze\n[B,1,256]", C["white"]),
        (74, 86, 20, 8, "Stem\nConv1d 1→32,k=1\nBN + ReLU", fill),
        (10, 58, 24, 18, "ResBlock1D(32)\n\nconv1(k=1)→BN→ReLU\nconv2(k=1)→BN\n+ residual → ReLU", C["white"]),
        (40, 58, 20, 18, "Transition\nConv1d 32→64\nBN + ReLU", fill),
        (66, 58, 24, 18, "ResBlock1D(64)\n\n同左结构\n通道数=64", C["white"]),
        (28, 28, 44, 16, "Flatten → [B, 64×256]\nLinear → [B, 160]\n模态嵌入 e ∈ R^160", C["out_fill"]),
    ]
    for x, y, w, h, text, fc in stages:
        ec = C["out"] if "模态嵌入" in text else color
        _box(ax, x, y, w, h, text, fc=fc, ec=ec, fontsize=8.2, weight="bold" if "模态嵌入" in text else "normal")

    _arrow(ax, 23, 90, 28, 90, color=color)
    _arrow(ax, 46, 90, 51, 90, color=color)
    _arrow(ax, 69, 90, 74, 90, color=color)
    _arrow(ax, 84, 86, 78, 76.5, color=color)
    _arrow(ax, 34, 67, 40, 67, color=color)
    _arrow(ax, 60, 67, 66, 67, color=color)
    _arrow(ax, 78, 58, 50, 44.2, color=color)

    _box(
        ax,
        5,
        6,
        90,
        16,
        "在 FusionModel 中的用法\n"
        "• CDS：seq_branch_tf / seq_branch_target，输入预计算 20 维向量\n"
        "• PPI：ppi_branch_tf / ppi_branch_target，输入 GCN 节点 160 维向量\n"
        "• 表达模态不走 ResNetBranch，而使用 ExpressionSampleSetEncoder（见专页）\n"
        "• Ablation（FusionModelAblation）可关闭任一模态，融合维数按启用模态数 ×2×160 收缩",
        fc=C["note"],
        ec=color,
        fontsize=8.5,
    )
    _footer(ax, "实现：grn/models/resnet.py")


def draw_pair_flow(ax):
    _title(
        ax,
        "配对级前向数据流（特征提取 → 融合）",
        "DataLoader 为每个 (TF, Target) 组装六路张量，模型内分别过对应提取器",
    )

    _box(ax, 4, 70, 20, 16, "PairDataset\n表达矩阵列子集\n+ CDS 查表\n+ PPI 节点嵌入", fc=C["note"], ec=C["shared"], fontsize=8.5, weight="bold")

    items = [
        (30, 82, "exp_tf [B,T]", C["expr"], C["expr_fill"]),
        (30, 72, "exp_target [B,T]", C["expr"], C["expr_fill"]),
        (30, 58, "seq_tf [B,20]", C["cds"], C["cds_fill"]),
        (30, 48, "seq_target [B,20]", C["cds"], C["cds_fill"]),
        (30, 34, "ppi_tf [B,160]", C["ppi"], C["ppi_fill"]),
        (30, 24, "ppi_target [B,160]", C["ppi"], C["ppi_fill"]),
    ]
    for x, y, text, ec, fc in items:
        _box(ax, x, y, 18, 7.5, text, fc=fc, ec=ec, fontsize=8)
        _arrow(ax, 24, 78, x, y + 3.5, color=C["shared"], lw=0.9)

    encs = [
        (54, 77, "Expr Enc TF", C["expr"], C["expr_fill"]),
        (54, 67, "Expr Enc Tar", C["expr"], C["expr_fill"]),
        (54, 53, "CDS Branch TF", C["cds"], C["cds_fill"]),
        (54, 43, "CDS Branch Tar", C["cds"], C["cds_fill"]),
        (54, 29, "PPI Branch TF", C["ppi"], C["ppi_fill"]),
        (54, 19, "PPI Branch Tar", C["ppi"], C["ppi_fill"]),
    ]
    for (x1, y1, *_), (x2, y2, t, ec, fc) in zip(items, encs):
        _box(ax, x2, y2, 18, 7.5, t, fc=fc, ec=ec, fontsize=8, weight="bold")
        _arrow(ax, x1 + 18, y1 + 3.5, x2, y2 + 3.5, color=ec)

    _box(ax, 78, 40, 18, 28, "Concat\n6 × 160\n= 960\n\nFusion MLP\n960→256→160\n\nConv1d Head\n→ 3 类 logits", fc=C["shared_fill"], ec=C["shared"], fontsize=8.5, weight="bold")
    for _, y, *_ in encs:
        _arrow(ax, 72, y + 3.5, 78, 54, color=C["shared"], lw=0.9)

    _box(
        ax,
        4,
        6,
        70,
        10,
        "类别：0=NoReg（无调控），1=Repress（抑制），2=Activate（激活）\n监督：KEGG PPrel act/inhib + 负采样；CDS + PPI 与表达三联融合（TripGRN）",
        fc=C["white"],
        ec=C["shared"],
        fontsize=8.5,
    )
    _footer(ax, "对应：grn/training/pair_dataloader.py · FusionModel.forward")


def main():
    _setup_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pages = [
        ("00_TripGRN_多模态特征提取器总览.pdf", draw_overview),
        ("01_表达模态_ExpressionSampleSetEncoder.pdf", draw_expression),
        ("02_CDS模态_预计算与ResNetBranch.pdf", draw_cds),
        ("03_PPI模态_GCN与ResNetBranch.pdf", draw_ppi),
        ("04_共享ResNetBranch结构详图.pdf", draw_resnet_detail),
        ("05_配对级前向数据流.pdf", draw_pair_flow),
    ]

    # Individual PDFs
    for name, drawer in pages:
        fig, ax = _new_fig()
        drawer(ax)
        out = OUT_DIR / name
        fig.savefig(out, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[ok] {out}")

    # Combined multipage PDF
    combined = OUT_DIR / "TripGRN_各模态特征提取器示意图_合集.pdf"
    with PdfPages(combined) as pdf:
        for name, drawer in pages:
            fig, ax = _new_fig()
            drawer(ax)
            pdf.savefig(fig, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
    print(f"[ok] {combined}")


if __name__ == "__main__":
    main()
