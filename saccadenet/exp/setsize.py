"""E2 set-size analysis (D29): glimpses and cost vs K at 16K, from saved runs only."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saccadenet.exp.e1_resolution import _wilson
from saccadenet.exp.trace_analysis import glimpse_decomposition, load_traces

CNN_FLOPS = 57_642_496  # one 96x96 FoveaNet forward pass (count_model_flops)


def _latest(run: Path) -> list[dict]:
    with (run / "episodes.csv").open(newline="", encoding="utf-8") as file:
        latest = {}
        for row in csv.DictReader(file):
            latest[row["method"], int(row["seed"])] = row
    return list(latest.values())


def _slope(k: np.ndarray, y: np.ndarray, *, replicates: int = 2000) -> tuple[float, float, float]:
    """Least-squares slope with an episode-level bootstrap interval (seed 40000)."""
    estimate = float(np.polyfit(k, y, 1)[0])
    rng = np.random.default_rng(40000)
    draws = []
    for _ in range(replicates):
        index = rng.integers(0, len(k), len(k))
        if len(set(k[index])) > 1:
            draws.append(np.polyfit(k[index], y[index], 1)[0])
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def summarize(runs: list[Path]) -> tuple[list[dict], dict]:
    rows, points = [], {"saccade_verify": [], "two_stage_crops": []}
    for run in runs:
        k = int(json.loads((run / "config.json").read_text(encoding="utf-8"))["k"])
        episodes = _latest(run)
        traces = {tuple(t["key"])[3]: t for t in load_traces(run, "saccadenet_lite")}
        for method in ("saccadenet_lite", "two_stage"):
            group = [r for r in episodes if r["method"] == method]
            hits = sum(r["target_hit"] == "1" for r in group)
            low, high = _wilson(hits, len(group))
            row = {"k": k, "method": method, "n": len(group), "hits": hits, "accuracy": hits / len(group), "wilson_low": low, "wilson_high": high,
                   "mean_steps": float(np.mean([float(r["steps"]) for r in group])),
                   "semantic_gflops": float(np.mean([float(r["semantic_flops"]) for r in group])) / 1e9,
                   "seconds": float(np.mean([float(r["episode_seconds"]) for r in group])),
                   "reasons": json.dumps({reason: sum(r["reason"] == reason for r in group) for reason in sorted({r["reason"] for r in group})}),
                   "verify_glimpses": "", "explore_glimpses": "", "prediction_half_k_plus_1": (k + 1) / 2}
            if method == "saccadenet_lite":
                parts = []
                for r in group:
                    trace = traces[int(r["seed"])]
                    trace["_k"] = k
                    parts.append(glimpse_decomposition(trace))
                row["verify_glimpses"] = float(np.mean([p["verify_glimpses"] for p in parts]))
                row["explore_glimpses"] = float(np.mean([p["explore_glimpses"] for p in parts]))
                row["semantic_prediction_gflops"] = CNN_FLOPS * (1 + (k + 1) / 2) / 1e9
                points["saccade_verify"] += [(k, p["verify_glimpses"]) for p in parts]
            else:
                points["two_stage_crops"] += [(k, float(r["steps"])) for r in group]
            rows.append(row)
    fits = {}
    for name, kmax in (("saccade_verify", 16), ("two_stage_crops", 32)):
        data = np.array([(k, y) for k, y in points[name] if k <= kmax], dtype=float)
        fits[name] = dict(zip(("slope", "low_95", "high_95"), _slope(data[:, 0], data[:, 1])), k_max=kmax, n=len(data))
    return rows, fits


def render(rows: list[dict], fits: dict, runs: list[Path], out: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"setsize-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    fields = sorted({key for row in rows for key in row}, key=lambda key: list(rows[0]).index(key) if key in rows[0] else 99)
    with (target / "setsize.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (target / "fits.json").write_text(json.dumps({"fits": fits, "runs": [str(r) for r in runs]}, indent=2), encoding="utf-8")

    saccade = sorted((r for r in rows if r["method"] == "saccadenet_lite"), key=lambda r: r["k"])
    two = sorted((r for r in rows if r["method"] == "two_stage"), key=lambda r: r["k"])
    ks = np.array([r["k"] for r in saccade])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.87, wspace=0.25)
    grid = np.linspace(2, 66, 50)
    left.plot(grid, (grid + 1) / 2, color="#999999", linestyle=":", label="(K+1)/2 (serial self-terminating)")
    left.plot(ks, [r["verify_glimpses"] for r in saccade], marker="o", color="#d1495b", label="SaccadeNet verification glimpses")
    left.plot(ks, [r["explore_glimpses"] for r in saccade], marker="^", color="#006d77", label="SaccadeNet exploration glimpses")
    left.plot([r["k"] for r in two], [r["mean_steps"] for r in two], marker="s", color="#e05d78", linestyle="--", label="Two-stage high-res crops")
    left.plot(ks, [r["mean_steps"] for r in saccade], marker="D", color="#333333", label="SaccadeNet total glimpses")
    left.axhline(40, color="#555555", linewidth=0.8, linestyle="-.")
    left.text(64, 41, "frozen T_max = 40", fontsize=8, color="#555555", ha="right")
    left.set_xscale("log", base=2)
    left.set_xticks(ks, [str(k) for k in ks])
    left.set_xlabel("K (cards on a 16K canvas)")
    left.set_ylabel("Mean per episode")
    left.set_title("Glimpses vs set size", fontsize=10)
    left.set_ylim(0, 46)
    left.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.85))
    for rows_, color, marker, label in ((saccade, "#006d77", "o", "SaccadeNet (frozen)"), (two, "#e05d78", "s", "Two-stage")):
        y = np.array([r["accuracy"] for r in rows_])
        err = np.stack((y - [r["wilson_low"] for r in rows_], np.array([r["wilson_high"] for r in rows_]) - y))
        right.errorbar([r["k"] for r in rows_], y, yerr=np.maximum(err, 0), marker=marker, capsize=3, color=color, label=label)
    right.set_xscale("log", base=2)
    right.set_xticks(ks, [str(k) for k in ks])
    right.set_ylim(-0.04, 1.06)
    right.set_xlabel("K")
    right.set_ylabel("Target-hit accuracy (95% Wilson)")
    right.set_title("Accuracy vs set size", fontsize=10)
    right.legend(fontsize=8, frameon=False, loc="lower left")
    fit = fits["saccade_verify"]
    fig.text(0.07, 0.05, f"16K, n=50 per K and method, final_test seeds 30000-30049 (D29). Verification slope K<=16: {fit['slope']:.2f} (95% {fit['low_95']:.2f}-{fit['high_95']:.2f}); "
             f"two-stage crop slope K<=32: {fits['two_stage_crops']['slope']:.2f} ({fits['two_stage_crops']['low_95']:.2f}-{fits['two_stage_crops']['high_95']:.2f}).", fontsize=7)
    fig.savefig(target / "figure9-setsize.png", dpi=180)
    fig.savefig(target / "figure9-setsize.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/analysis"))
    args = parser.parse_args()
    rows, fits = summarize(args.runs)
    target = render(rows, fits, args.runs, args.out)
    print(target)
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
