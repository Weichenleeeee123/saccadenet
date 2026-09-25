"""Figure 8: glimpse and cost scaling from 1080p to the 24K extrapolation (E1 + E5), from saved runs only."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saccadenet.exp.e1_resolution import _wilson
from saccadenet.retina.horizon import coverage_radius, derived_grid, detection_horizon

NAMES = {1920: "1080p", 3840: "4K", 7680: "8K", 15360: "16K", 24576: "24K"}


def _episodes(run: Path) -> list[dict]:
    with (run / "episodes.csv").open(newline="", encoding="utf-8") as file:
        latest = {}
        for row in csv.DictReader(file):
            latest[row["method"], int(row["width"]), int(row["seed"])] = row
    return list(latest.values())


def _decomposition(analysis: Path) -> dict[int, dict]:
    with (analysis / "episodes.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    result = {}
    for width in sorted({int(r["width"]) for r in rows}):
        group = [r for r in rows if int(r["width"]) == width]
        result[width] = {key: float(np.mean([float(r[key]) for r in group])) for key in ("explore_glimpses", "verify_glimpses", "steps")}
    return result


def build(e1_run: Path, e5_run: Path, e1_analysis: Path, e5_derived_analysis: Path, e5_frozen_analysis: Path) -> list[dict]:
    r1, _ = detection_horizon()
    rows = []
    sources = [(e1_run, "saccadenet_lite", _decomposition(e1_analysis), "E1 (5x5 grid)"),
               (e5_run, "saccadenet_derived_grid", _decomposition(e5_derived_analysis), "E5 derived grid"),
               (e5_run, "saccadenet_lite", _decomposition(e5_frozen_analysis), "E5 frozen 5x5")]
    for run, method, decomposition, label in sources:
        episodes = [row for row in _episodes(run) if row["method"] == method]
        for width in sorted({int(r["width"]) for r in episodes}):
            group = [r for r in episodes if int(r["width"]) == width]
            height = int(group[0]["height"])
            grid = derived_grid(width, height, r1) if method == "saccadenet_derived_grid" else 5
            hits = sum(r["target_hit"] == "1" for r in group)
            recall = sum(r["target_recall"] == "1" for r in group)
            low, high = _wilson(hits, len(group))
            rows.append({
                "source": label, "method": method, "width": width, "height": height, "n": len(group), "grid": grid,
                "coverage_radius": coverage_radius(width, height, grid), "r_where": r1,
                "hits": hits, "accuracy": hits / len(group), "wilson_low": low, "wilson_high": high, "target_recall": recall / len(group),
                "mean_glimpses": float(np.mean([float(r["steps"]) for r in group])),
                "explore_glimpses": decomposition[width]["explore_glimpses"], "verify_glimpses": decomposition[width]["verify_glimpses"],
                "semantic_gflops": float(np.mean([float(r["semantic_flops"]) for r in group])) / 1e9,
                "sensing_mbytes": float(np.mean([float(r["sensing_bytes"]) for r in group])) / 1e6,
                "seconds": float(np.mean([float(r["episode_seconds"]) for r in group])),
            })
    return rows


def render(rows: list[dict], out: Path, provenance: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"scaling-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "plot-data.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    main = [r for r in rows if r["source"] != "E5 frozen 5x5"]
    frozen = [r for r in rows if r["source"] == "E5 frozen 5x5"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.86, wspace=0.28)
    x = np.arange(len(main))
    verify = [r["verify_glimpses"] for r in main]
    explore = [r["explore_glimpses"] for r in main]
    left.bar(x, verify, width=0.6, color="#d1495b", label="verification glimpses")
    left.bar(x, explore, width=0.6, bottom=verify, color="#006d77", label="exploration glimpses")
    left.axhline(6.5, color="#d1495b", linestyle=":", linewidth=1, label="(K+1)/2 = 6.5: random-order serial search")
    for xi, r in zip(x, main):
        left.text(xi, r["mean_glimpses"] + 1.5, f"grid {r['grid']}x{r['grid']}\nhit {r['hits']}/{r['n']}", ha="center", fontsize=7.5, color="#333333")
    left.set_xticks(x, [NAMES[r["width"]] + ("*" if r["width"] == 24576 else "") for r in main])
    left.set_ylim(0, max(r["mean_glimpses"] for r in main) * 1.25)
    left.set_ylabel("Mean glimpses per episode")
    left.set_title("Glimpses: verification flat, exploration ~ area beyond r_where", fontsize=10)
    left.legend(fontsize=8, frameon=False, loc="upper left")

    widths = np.array([r["width"] for r in main], dtype=float)
    right.plot(widths, [r["semantic_gflops"] for r in main], marker="o", color="#006d77", label="semantic GFLOPs (mean)")
    right.plot(widths, [r["mean_glimpses"] / 100 for r in main], marker="s", color="#8a8a8a", linestyle="--", label="glimpses / 100")
    right.plot(widths, [r["sensing_mbytes"] / 1000 for r in main], marker="^", color="#e29578", label="bytes read (GB)")
    for r in frozen:
        right.scatter([r["width"]], [r["semantic_gflops"]], marker="x", color="#b5179e", zorder=5, label=f"24K frozen 5x5: recall {r['target_recall']:.0%}, hit {r['hits']}/{r['n']}")
    right.set_xscale("log", base=2)
    right.set_xticks([1920, 3840, 7680, 15360, 24576], ["1080p", "4K", "8K", "16K", "24K*"])
    right.set_ylabel("value (see legend)")
    right.set_title("Cost per episode (SaccadeNet)", fontsize=10)
    right.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.text(0.06, 0.06, "* 24K is the preregistered out-of-range test E5 (n=50, derived grid = ceil(hypot(W,H)/(2 r_where)), r_where = 2^6.5 x 128/(2 pi) = 1844 px).", fontsize=7)
    fig.text(0.06, 0.03, "E1: n=100 per width, frozen 5x5 grid (covers only up to 16.1K). Semantic = analytic CNN+postprocessing FLOPs; bytes include the full-image pyramid.", fontsize=7)
    fig.savefig(target / "figure8-scaling.png", dpi=180)
    fig.savefig(target / "figure8-scaling.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1-run", type=Path, required=True)
    parser.add_argument("--e5-run", type=Path, required=True)
    parser.add_argument("--e1-analysis", type=Path, required=True)
    parser.add_argument("--e5-derived-analysis", type=Path, required=True)
    parser.add_argument("--e5-frozen-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/analysis"))
    args = parser.parse_args()
    rows = build(args.e1_run, args.e5_run, args.e1_analysis, args.e5_derived_analysis, args.e5_frozen_analysis)
    provenance = {key: str(value) for key, value in vars(args).items()}
    print(render(rows, args.out, provenance))


if __name__ == "__main__":
    main()
