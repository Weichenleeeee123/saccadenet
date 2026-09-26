"""Figure 0: SaccadeNet architecture — one glimpse of the closed loop (labels match the code)."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED = "#1f2a30", "#4f5f67"
SENSOR, PERCEIVE, DECIDE, LOOP = "#e8f1f2", "#e6f4ef", "#fdf1e3", "#006d77"


def box(ax, x, y, w, h, title, body, face, edge="#8aa0a8"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=face, ec=edge, lw=1.2))
    ax.text(x + 1.2, y + h - 1.3, title, fontsize=13, fontweight="bold", color=INK, va="top")
    ax.text(x + 1.2, y + h - 4.9, body, fontsize=10, color=MUTED, va="top", linespacing=1.5)


def arrow(ax, start, end, label="", color=INK, rad=0.0, label_xy=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, lw=1.7, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.2)
        ax.text(lx, ly, label, fontsize=10, color=color, ha="center", va="bottom", fontweight="bold")


def retina_icon(ax, cx, cy, r):
    """Fovea square plus log-polar rings/sectors (schematic, not to scale)."""
    for k in range(5):
        ax.add_patch(Circle((cx, cy), r * (0.28 * 1.38 ** k), fill=False, lw=0.6, ec=LOOP, alpha=0.55))
    for s in range(16):
        a = 2 * math.pi * s / 16
        ax.plot([cx + 0.28 * r * math.cos(a), cx + r * 1.02 * math.cos(a)],
                [cy + 0.28 * r * math.sin(a), cy + r * 1.02 * math.sin(a)], lw=0.5, color=LOOP, alpha=0.5)
    ax.add_patch(Wedge((cx, cy), r * 1.02, 20, 42, width=r * 0.35, fc="#ffb703", alpha=0.55, lw=0))
    half = 0.2 * r
    ax.add_patch(FancyBboxPatch((cx - half, cy - half), 2 * half, 2 * half, boxstyle="square,pad=0", fc=LOOP, ec=LOOP))


def render(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(16, 9.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    fig.suptitle("SaccadeNet：一次注视的闭环（每一眼重复一次，直到停机）", fontsize=17, fontweight="bold", color=INK, y=0.985)

    # Sensor band
    box(ax, 27, 50.5, 21, 7.5, "原图 W×H", "只交给传感器；推理回路\n从不直接读原图或真值", SENSOR)
    box(ax, 2, 50.5, 21, 7.5, "抗混叠金字塔", "2×2 平均逐级缩小，只建一次\n感知成本随像素数增长", SENSOR)
    arrow(ax, (26.4, 54.2), (23.6, 54.2))
    arrow(ax, (12.5, 50.0), (12.5, 47.6))

    # Row A: perception, left to right
    box(ax, 2, 30.5, 21, 16.5, "① 视网膜采样", "注视点 (x, y)：\n中央凹 96×96 高清\n外周对数极坐标\nS=128 扇区 × 79–122 环\n每眼 1.9–2.5 万个样本", PERCEIVE)
    retina_icon(ax, 18.6, 35.0, 3.4)
    box(ax, 27, 33.5, 21, 13.5, "② 候选检测与定位", "在样本上找亮卡连通域\n反投影回画布坐标\n用边界范围取中心\n跨眼合并成稳定候选 ID", PERCEIVE)
    box(ax, 52, 33.5, 21, 13.5, "③ 候选视图重建", "只用本眼的中央凹/外周\n样本插值出 96×96 视图\n不回读金字塔或原图", PERCEIVE)
    box(ax, 77, 33.5, 21, 13.5, "④ 共享 CNN", "FoveaNet：4 级卷积 + 池化\n11 类（数字 0–9 + 空白）\n5,764 万 FLOPs / 次\n输出查询数字的 log-odds", PERCEIVE)
    arrow(ax, (23.6, 40.2), (26.4, 40.2))
    arrow(ax, (48.6, 40.2), (51.4, 40.2))
    arrow(ax, (73.6, 40.2), (76.4, 40.2))
    arrow(ax, (87.5, 33.0), (87.5, 27.6))

    # Row B: decision, right to left
    box(ax, 77, 13.5, 21, 13.5, "⑤ 偏心率标定", "按离心率 e 分箱（独立标定集）\nx = (s − μ0(e)) / (μ1(e) − μ0(e))\nd′(e) = (μ1 − μ0) / σ\nd′ < 0.1 的观测不更新", DECIDE)
    box(ax, 52, 13.5, 21, 13.5, "⑥ 融合 B → 后验", "每个候选只留质量最高的一眼\nD = max d′²，LLR = D·(x − 1/2)\n在已发现候选上 softmax\n同质量重看不累加证据", DECIDE)
    box(ax, 27, 13.5, 21, 13.5, "⑦ 停机判断", "已发现 ≥ K 个候选\n且最大后验 ≥ τ = 0.95 → 停\n否则继续；到 T_max = 40\n或探索锚点用尽也停", DECIDE)
    box(ax, 2, 13.5, 21, 13.5, "⑧ 选下一眼", "候选 ≥ K：在候选中心与中点里\n最大化增益\nG = Σ p·max(0, d′²(e) − D)\n候选 < K：走 g×g 探索锚点", DECIDE)
    arrow(ax, (76.4, 20.2), (73.6, 20.2))
    arrow(ax, (51.4, 20.2), (48.6, 20.2))
    arrow(ax, (26.4, 20.2), (23.6, 20.2), label="否", color="#b5179e", label_xy=(25.0, 21.2))
    arrow(ax, (12.5, 27.6), (12.5, 30.0), label="下一注视点", color=LOOP, label_xy=(18.0, 27.9))

    # Output and notes
    box(ax, 27, 1.5, 21, 8.5, "输出", "后验最高候选的画布坐标\n评测层才与真值比对", "#eef2f5")
    arrow(ax, (37.5, 13.0), (37.5, 10.6), label="是", color="#2a9d8f", label_xy=(39.5, 11.0))
    ax.text(52, 9.3, "两个视野半径（由采样几何推出，见报告）", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(52, 1.4, "r_what ≈ 150 px：数字可读（d′ > 1）\n"
                     "r_where = 2^6.5 · S / (2π) ≈ 1,844 px：亮卡必定可见\n"
                     "探索网格 g = ceil(hypot(W, H) / (2·r_where))；冻结配置 g = 5，覆盖到约 16.1K 宽\n"
                     "成本分账：感知 = 金字塔 + 采样；语义 = 检测 + 重建 + CNN + 融合 + 路由",
            fontsize=10, color=MUTED, linespacing=1.55)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"architecture-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    fig.savefig(target / "figure0-architecture.png", dpi=200)
    fig.savefig(target / "figure0-architecture.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("reports/figures"))
    print(render(parser.parse_args().out))


if __name__ == "__main__":
    main()
