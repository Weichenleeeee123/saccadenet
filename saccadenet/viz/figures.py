"""Render Figure 4/5 only from a completed, hash-identified E1 run."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saccadenet.exp.e1_resolution import _wilson, planned_jobs
from saccadenet.exp.store import episode_key


METHOD_LABELS = {
    "saccadenet_lite": "SaccadeNet-lite",
    "full_res_sliding": "Full-resolution sliding",
    "downsample_1stage": "1024px one-stage",
    "two_stage": "1024px two-stage",
}


def build_plot_data(run_dir: Path, *, bootstrap: int = 2000) -> list[dict]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    jobs = planned_jobs(cfg, smoke=manifest["split"] == "development")
    latest: dict[tuple[str, int, int, int], dict] = {}
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = episode_key(row["method"], row["width"], row["height"], row["seed"])
            latest[key] = row
    groups: dict[tuple[str, int, int], list[dict]] = {}
    for key in jobs:
        if key not in latest:
            raise ValueError(f"E1 is incomplete: missing {key}")
        groups.setdefault(key[:3], []).append(latest[key])
    result = []
    for method, width, height in sorted(groups, key=lambda key: (key[1], list(METHOD_LABELS).index(key[0]))):
        rows = groups[method, width, height]
        n = len(rows)
        hits = sum(row["status"] == "ok" and row["target_hit"] == "1" for row in rows)
        completed = sum(row["status"] == "ok" for row in rows)
        semantic = np.array([float(row["semantic_flops"] or 0) for row in rows])
        sensing = np.array([float(row["sensing_flops"] or 0) for row in rows])
        bytes_read = np.array([float(row.get("sensing_bytes") or 0) for row in rows])
        elapsed = np.array([float(row["episode_seconds"] or 0) for row in rows])
        if bootstrap > 0:
            rng = np.random.default_rng(40000)
            indices = rng.integers(0, n, (bootstrap, n))
            means = semantic[indices].mean(axis=1)
            cost_low, cost_high = np.quantile(means, [0.025, 0.975])
        else:
            cost_low = cost_high = float(semantic.mean())
        wilson_low, wilson_high = _wilson(hits, n)
        result.append({
            "method": method, "width": width, "height": height, "n": n,
            "completed_n": completed, "hits": hits, "accuracy": hits / n,
            "wilson_low": wilson_low, "wilson_high": wilson_high,
            "mean_semantic_flops": float(semantic.mean()), "semantic_low": float(cost_low), "semantic_high": float(cost_high),
            "mean_sensing_flops": float(sensing.mean()), "mean_sensing_bytes": float(bytes_read.mean()),
            "mean_total_estimated_flops": float((semantic + sensing).mean()), "mean_episode_seconds": float(elapsed.mean()),
            "estimate_type": "measured_analytic_operation_count", "unit": "FLOPs",
        })
    return result


def render(run_dir: Path, out: Path, *, bootstrap: int = 2000) -> Path:
    rows = build_plot_data(run_dir, bootstrap=bootstrap)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"{run_dir.name}-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with (target / "plot-data.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#006d77", "#e29578", "#5951a5", "#d1495b")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.13, right=0.74, bottom=0.20, top=0.88)
    for color, (method, label) in zip(colors, METHOD_LABELS.items()):
        series = [row for row in rows if row["method"] == method]
        x = np.array([row["width"] for row in series])
        y = np.array([row["mean_semantic_flops"] for row in series])
        lo = y - np.array([row["semantic_low"] for row in series])
        hi = np.array([row["semantic_high"] for row in series]) - y
        ax.errorbar(x, y, yerr=np.stack((np.maximum(0, lo), np.maximum(0, hi))), marker="o", capsize=3, label=label, color=color)
        if method == "full_res_sliding":
            ax.annotate(f"n={series[-1]['n']}", (x[-1], y[-1]), xytext=(-3, -18), textcoords="offset points", ha="right", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1920, 3840, 7680, 15360], ["1080p", "4K", "8K", "16K"])
    ax.set_xlabel("Image width")
    ax.set_ylabel("Mean semantic computation (estimated FLOPs)")
    ax.set_title("Figure 4. Semantic computation across image resolution")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.text(0.13, 0.05, f"Run {run_dir.name}; Git {manifest['git_head'][:8]}; 95% bootstrap CI. Sensing, bytes and latency: plot-data.csv.", fontsize=7)
    fig.savefig(target / "figure4-semantic.png", dpi=180)
    fig.savefig(target / "figure4-semantic.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.13, right=0.74, bottom=0.20, top=0.88)
    for color, (method, label) in zip(colors, METHOD_LABELS.items()):
        series = [row for row in rows if row["method"] == method]
        x = np.array([row["width"] for row in series])
        y = np.array([row["accuracy"] for row in series])
        lo = y - np.array([row["wilson_low"] for row in series])
        hi = np.array([row["wilson_high"] for row in series]) - y
        ax.errorbar(x, y, yerr=np.stack((np.maximum(0, lo), np.maximum(0, hi))), marker="o", capsize=3, label=label, color=color)
        if method == "full_res_sliding":
            ax.annotate(f"n={series[-1]['n']}", (x[-1], y[-1]), xytext=(-3, -18), textcoords="offset points", ha="right", fontsize=8)
    ax.axhline(1 / 12, color="#777777", linestyle=":", linewidth=1, label="Random (1/12)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1920, 3840, 7680, 15360], ["1080p", "4K", "8K", "16K"])
    ax.set_ylim(-0.04, 1.07)
    ax.set_xlabel("Image width")
    ax.set_ylabel("Target-hit accuracy")
    ax.set_title("Figure 5. Search accuracy across image resolution")
    ax.grid(alpha=0.2)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.text(0.13, 0.05, f"Run {run_dir.name}; Git {manifest['git_head'][:8]}; 95% Wilson CI; failures in denominator.", fontsize=7)
    fig.savefig(target / "figure5-accuracy.png", dpi=180)
    fig.savefig(target / "figure5-accuracy.svg")
    plt.close(fig)
    with (target / "provenance.json").open("x", encoding="utf-8") as file:
        json.dump({"source_run": str(run_dir), "git_head": manifest["git_head"], "config_sha256": manifest["config_sha256"], "checkpoint_sha256": manifest["checkpoint_sha256"], "bootstrap_seed": 40000, "bootstrap_replicates": bootstrap}, file, indent=2)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/e1"))
    args = parser.parse_args()
    print(render(args.run, args.out))


if __name__ == "__main__":
    main()
