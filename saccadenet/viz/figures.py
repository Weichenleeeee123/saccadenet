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


def render_g2_sensitivity(main_run: Path, supplement_run: Path, out: Path, *, s_min: dict[str, int]) -> Path:
    """Figure 5b: one-stage downsample accuracy with the frozen v1 vs the scale-augmented v2 classifier."""
    main_rows = [row for row in build_plot_data(main_run, bootstrap=0) if row["method"] == "downsample_1stage"]
    supp_rows = [row for row in build_plot_data(supplement_run, bootstrap=0) if row["method"] == "downsample_1stage"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"g2-sensitivity-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "plot-data.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("classifier", "width", "n", "hits", "accuracy", "wilson_low", "wilson_high"))
        writer.writeheader()
        for label, rows in (("v1", main_rows), ("v2", supp_rows)):
            for row in rows:
                writer.writerow({"classifier": label, **{key: row[key] for key in ("width", "n", "hits", "accuracy", "wilson_low", "wilson_high")}})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.9)
    for (label, rows), color, style in zip((("v1", main_rows), ("v2", supp_rows)), ("#5951a5", "#b5179e"), ("-", "--")):
        x = np.array([row["width"] for row in rows])
        y = np.array([row["accuracy"] for row in rows])
        err = np.stack((np.maximum(0, y - [row["wilson_low"] for row in rows]), np.maximum(0, np.array([row["wilson_high"] for row in rows]) - y)))
        name = f"{label}: {'frozen E1, 48px-only' if label == 'v1' else 'scale-aug, post-hoc D24'}, s_min={s_min[label]}px"
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, color=color, linestyle=style, label=name)
        critical = 48 * 1024 / s_min[label]
        ax.axvline(critical, color=color, linestyle=":", linewidth=1.2)
        ax.text(critical, 0.93, f" W_c={critical:.0f}px", color=color, fontsize=8, rotation=90, va="top")
    ax.axhline(1 / 12, color="#777777", linestyle=":", linewidth=1, label="Random (1/12)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1920, 3840, 7680, 15360], ["1080p", "4K", "8K", "16K"])
    ax.set_ylim(-0.04, 1.0)
    ax.set_xlabel("Image width")
    ax.set_ylabel("Target-hit accuracy (1024px one-stage)")
    ax.set_title("Figure 5b. Downsample collapse vs classifier scale range")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.text(0.1, 0.07, "Same 400 final_test canvases; 95% Wilson CI; dotted W_c = 48*1024/s_min (predicted critical width).", fontsize=7)
    fig.text(0.1, 0.035, f"v1: {main_run.name}   v2: {supplement_run.name}", fontsize=7)
    fig.savefig(target / "figure5b-g2-sensitivity.png", dpi=180)
    fig.savefig(target / "figure5b-g2-sensitivity.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/e1"))
    parser.add_argument("--g2-supplement", type=Path, help="D24 run; renders Figure 5b instead of Figure 4/5")
    args = parser.parse_args()
    if args.g2_supplement:
        print(render_g2_sensitivity(args.run, args.g2_supplement, args.out, s_min={"v1": 48, "v2": 24}))
    else:
        print(render(args.run, args.out))


if __name__ == "__main__":
    main()
