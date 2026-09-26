"""Detection-horizon probe across retina sector counts (development split only, no CNN, no test data).

For each sector count S the analytic model predicts r1(S) = 2^6.5 * S / (2 pi) (certain detection)
and r2(S) = 2^7.5 * S / (2 pi) (no detection beyond). We sample retinas at a fixed fixation grid on
16K development canvases and measure P(card detected | eccentricity).
"""

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

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.retina.detect import detect_on_retina
from saccadenet.retina.horizon import detection_horizon
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina

EDGES = np.array([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 10000, 13000, 18000], dtype=float)
MATCH = 144.0


def probe(seeds: list[int], sectors: list[int], grid: tuple[int, int]) -> list[tuple[int, float, int]]:
    bank = MnistBank.from_split("development")
    events = []
    for seed in seeds:
        config = EpisodeConfig(width=15360, height=8640, query=seed % 10)
        image, truth = make_canvas(config, seed, bank)
        pyramid = build_pyramid(image)
        centers = truth.centers_xy.astype(float)
        fixations = [((c + 0.5) * 15360 / grid[0], (r + 0.5) * 8640 / grid[1]) for c in range(grid[0]) for r in range(grid[1])]
        for s in sectors:
            for fx, fy in fixations:
                retina = sample_retina(pyramid, (fx, fy), fovea_size=96, sectors=s)
                detections = detect_on_retina(retina, brightness_threshold=195.0)
                for cx, cy in centers:
                    hit = any(math.hypot(d.xy[0] - cx, d.xy[1] - cy) <= MATCH for d in detections)
                    events.append((s, math.hypot(cx - fx, cy - fy), int(hit)))
    return events


def crossing(rows: list[dict]) -> float:
    """Eccentricity where the binned P(detect) first falls below 0.5 (linear interpolation on log r)."""
    prev = None
    for row in rows:
        if row["exposures"] == 0:
            continue
        mid = math.sqrt(max(row["ecc_low"], 50) * row["ecc_high"])
        if row["p_detect"] < 0.5 and prev is not None:
            (m0, p0), (m1, p1) = prev, (mid, row["p_detect"])
            t = (p0 - 0.5) / (p0 - p1)
            return math.exp(math.log(m0) + t * (math.log(m1) - math.log(m0)))
        prev = (mid, row["p_detect"])
    return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs=2, default=(21000, 21010))
    parser.add_argument("--sectors", type=int, nargs="+", default=(64, 128, 256))
    parser.add_argument("--out", type=Path, default=Path("reports/analysis"))
    args = parser.parse_args()
    events = probe(list(range(*args.seeds)), list(args.sectors), (7, 5))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = args.out / f"horizon-probe-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    table, summary = [], {"seeds": list(args.seeds), "split": "development", "fixation_grid": [7, 5], "sectors": {}}
    for s in args.sectors:
        ecc = np.array([e for ss, e, _ in events if ss == s])
        hit = np.array([h for ss, _, h in events if ss == s])
        rows = []
        for low, high in zip(EDGES[:-1], EDGES[1:]):
            mask = (ecc >= low) & (ecc < high)
            rows.append({"sectors": s, "ecc_low": low, "ecc_high": high, "exposures": int(mask.sum()), "detected": int(hit[mask].sum()), "p_detect": float(hit[mask].mean()) if mask.any() else float("nan")})
        table += rows
        r1, r2 = detection_horizon(192, s)
        summary["sectors"][str(s)] = {"r1_analytic": r1, "r2_analytic": r2, "p50_measured": crossing(rows)}
    with (target / "horizon-probe.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (target / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.2, top=0.88)
    colors = {64: "#8a7fe0", 128: "#006d77", 256: "#e29578"}
    for s in args.sectors:
        rows = [r for r in table if r["sectors"] == s and r["exposures"]]
        mids = [math.sqrt(max(r["ecc_low"], 50) * r["ecc_high"]) for r in rows]
        ax.plot(mids, [r["p_detect"] for r in rows], marker="o", color=colors.get(s, "#333333"), label=f"S={s} sectors (measured)")
        r1, r2 = detection_horizon(192, s)
        ax.axvspan(r1, r2, color=colors.get(s, "#333333"), alpha=0.10, lw=0)
        ax.axvline(r1, color=colors.get(s, "#333333"), linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(100, 20000)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Eccentricity of card from fixation (px)")
    ax.set_ylabel("P(card detected in this glimpse)")
    ax.set_title("Figure 10. Detection horizon scales with sector count: r1 = 2^6.5 S / (2 pi)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.text(0.09, 0.05, f"Development split seeds {args.seeds[0]}-{args.seeds[1] - 1}, 16K, 35 fixed fixations per canvas; detection only (no CNN). Dashed: analytic r1; band: r1-r2.", fontsize=7)
    fig.savefig(target / "figure10-horizon-sectors.png", dpi=180)
    fig.savefig(target / "figure10-horizon-sectors.svg")
    plt.close(fig)
    print(target)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
