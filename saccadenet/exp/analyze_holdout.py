"""Audit and summarize the frozen B04 holdout (D47): full denominators, seed-paired contrasts, Figure 11."""

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from saccadenet.exp.e1_resolution import _wilson, planned_jobs
from saccadenet.exp.store import episode_key

NAMES = {1920: "1080p", 3840: "4K", 7680: "8K", 15360: "16K"}
LABELS = {"saccadenet_lite": "SaccadeNet (B03 localizer)", "two_stage_no_coarse": "Two-stage 1024 (no coarse CNN)",
          "two_stage_optimized": "Two-stage, width by resolution"}
BOOT, BOOT_SEED = 2000, 40000


def latest_rows(run: Path) -> dict:
    latest = {}
    with (run / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            latest[episode_key(row["method"], row["width"], row["height"], row["seed"])] = row
    return latest


def audit(run: Path, latest: dict) -> dict:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    planned = set(planned_jobs(cfg))
    with (run / "seeds.csv").open(newline="", encoding="utf-8") as file:
        seeds = list(csv.DictReader(file))
    return {
        "planned": len(planned), "recorded": len(latest), "missing": sorted(planned - latest.keys()),
        "unexpected": sorted(latest.keys() - planned), "not_ok": sorted(k for k, r in latest.items() if r["status"] != "ok"),
        "seed_rows": len(seeds), "errors_bytes": (run / "errors.jsonl").stat().st_size,
        "episodes_sha256": hashlib.sha256((run / "episodes.csv").read_bytes()).hexdigest(),
    }


def _boot_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    index = rng.integers(0, len(values), (BOOT, len(values)))
    return tuple(float(v) for v in np.quantile(values[index].mean(axis=1), [0.025, 0.975]))


def summarize(latest: dict, methods: list[str], widths: list[int]) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(BOOT_SEED)
    rows, pairs = [], []
    for width in widths:
        for method in methods:
            group = [r for k, r in latest.items() if k[0] == method and k[1] == width]
            hits = sum(r["status"] == "ok" and r["target_hit"] == "1" for r in group)
            low, high = _wilson(hits, len(group))
            seconds = np.array([float(r["episode_seconds"] or 0) for r in group])
            s_low, s_high = _boot_mean(seconds, rng)
            recall = [r["target_recall"] for r in group if r["target_recall"] not in ("", None)]
            rows.append({
                "width": width, "method": method, "n": len(group), "hits": hits, "accuracy": hits / len(group),
                "wilson_low": low, "wilson_high": high,
                "target_recall": (sum(x == "1" for x in recall) / len(recall)) if recall else "",
                "mean_steps": float(np.mean([float(r["steps"] or 0) for r in group])),
                "semantic_gflops": float(np.mean([float(r["semantic_flops"] or 0) for r in group])) / 1e9,
                "sensing_gflops": float(np.mean([float(r["sensing_flops"] or 0) for r in group])) / 1e9,
                "logical_read_mb": float(np.mean([float(r["sensing_bytes"] or 0) for r in group])) / 1e6,
                "seconds": float(seconds.mean()), "seconds_low": s_low, "seconds_high": s_high,
                "reasons": json.dumps(collections.Counter(r["reason"] for r in group), sort_keys=True),
            })
        for other in methods[1:]:
            seeds = sorted({k[3] for k in latest if k[1] == width})
            a = {s: latest[k] for s in seeds for k in [(methods[0], width, width * 9 // 16, s)] if k in latest}
            b = {s: latest[k] for s in seeds for k in [(other, width, width * 9 // 16, s)] if k in latest}
            common = sorted(a.keys() & b.keys())
            ha = np.array([a[s]["target_hit"] == "1" for s in common], dtype=float)
            hb = np.array([b[s]["target_hit"] == "1" for s in common], dtype=float)
            ta = np.array([float(a[s]["episode_seconds"]) for s in common])
            tb = np.array([float(b[s]["episode_seconds"]) for s in common])
            index = rng.integers(0, len(common), (BOOT, len(common)))
            hit_ci = np.quantile((ha[index] - hb[index]).mean(axis=1), [0.025, 0.975])
            time_ci = np.quantile((ta[index] - tb[index]).mean(axis=1), [0.025, 0.975])
            pairs.append({
                "width": width, "a": methods[0], "b": other, "n_pairs": len(common),
                "both_hit": int(((ha == 1) & (hb == 1)).sum()), "only_a": int(((ha == 1) & (hb == 0)).sum()),
                "only_b": int(((ha == 0) & (hb == 1)).sum()), "both_miss": int(((ha == 0) & (hb == 0)).sum()),
                "hit_diff": float((ha - hb).mean()), "hit_diff_low": float(hit_ci[0]), "hit_diff_high": float(hit_ci[1]),
                "seconds_diff": float((ta - tb).mean()), "seconds_diff_low": float(time_ci[0]), "seconds_diff_high": float(time_ci[1]),
                "seconds_ratio": float(ta.mean() / tb.mean()),
            })
    return rows, pairs


def pooled(latest: dict, methods: list[str]) -> list[dict]:
    """Across widths, resampling canvas seeds as clusters (a seed index is shared by the four widths)."""
    rng = np.random.default_rng(BOOT_SEED)
    seeds = sorted({k[3] for k in latest})
    result = []
    for other in methods[1:]:
        diff = np.zeros(len(seeds))
        counts = collections.Counter()
        for i, seed in enumerate(seeds):
            for key, row in latest.items():
                if key[0] == methods[0] and key[3] == seed:
                    other_row = latest.get((other, key[1], key[2], seed))
                    if other_row is None:
                        continue
                    a, b = row["target_hit"] == "1", other_row["target_hit"] == "1"
                    diff[i] += a - b
                    counts["pairs"] += 1
                    counts["only_a"] += a and not b
                    counts["only_b"] += b and not a
        index = rng.integers(0, len(seeds), (BOOT, len(seeds)))
        draws = diff[index].sum(axis=1) / counts["pairs"]
        low, high = np.quantile(draws, [0.025, 0.975])
        result.append({"a": methods[0], "b": other, "pairs": counts["pairs"], "only_a": counts["only_a"], "only_b": counts["only_b"],
                       "hit_diff": float(diff.sum() / counts["pairs"]), "low": float(low), "high": float(high)})
    return result


def render(rows: list[dict], methods: list[str], widths: list[int], target: Path, run: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.87, wspace=0.25)
    colors = {"saccadenet_lite": "#006d77", "two_stage_no_coarse": "#e05d78", "two_stage_optimized": "#8a7fe0"}
    offsets = dict(zip(methods, (-0.06, 0.0, 0.06)))
    for method in methods:
        series = [r for r in rows if r["method"] == method]
        x = np.array([r["width"] for r in series], dtype=float) * (2 ** offsets[method])
        y = np.array([r["accuracy"] for r in series])
        err = np.stack((y - [r["wilson_low"] for r in series], np.array([r["wilson_high"] for r in series]) - y))
        left.errorbar(x, y, yerr=np.maximum(err, 0), marker="o", capsize=3, color=colors[method], label=LABELS[method])
        t = np.array([r["seconds"] for r in series])
        terr = np.stack((t - [r["seconds_low"] for r in series], np.array([r["seconds_high"] for r in series]) - t))
        right.errorbar(x, t, yerr=np.maximum(terr, 0), marker="s", capsize=3, color=colors[method], label=LABELS[method])
    for ax in (left, right):
        ax.set_xscale("log", base=2)
        ax.set_xticks(widths, [NAMES[w] for w in widths])
        ax.set_xlabel("Image width")
    left.set_ylim(0.8, 1.02)
    left.set_ylabel("Target-hit accuracy (95% Wilson)")
    left.set_title("Accuracy on the new holdout (seeds 32000-32099)", fontsize=10)
    left.legend(fontsize=8, frameon=False, loc="lower left")
    right.set_yscale("log")
    right.set_ylabel("Seconds per episode (mean, 95% bootstrap)")
    right.set_title("Wall-clock per episode (profiling forwards removed)", fontsize=10)
    fig.text(0.07, 0.05, f"Run {run.name}; n=100 canvases per width, all three methods on every canvas; frozen in D47 before running.", fontsize=7)
    fig.savefig(target / "figure11-holdout.png", dpi=180)
    fig.savefig(target / "figure11-holdout.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/recovery"))
    args = parser.parse_args()
    cfg = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    methods, widths = cfg["methods"], [w for w, _ in cfg["resolutions"]]
    latest = latest_rows(args.run)
    checks = audit(args.run, latest)
    if checks["missing"] or checks["unexpected"] or checks["not_ok"]:
        raise SystemExit(f"holdout incomplete or has failures: { {k: v for k, v in checks.items() if isinstance(v, list) and v} }")
    rows, pairs = summarize(latest, methods, widths)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = args.out / f"holdout-b04-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    for name, data in (("summary.csv", rows), ("paired.csv", pairs)):
        with (target / name).open("x", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=tuple(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {"run": str(args.run), "audit": {k: v for k, v in checks.items() if not isinstance(v, list)}, "pooled_hit_diff": pooled(latest, methods),
              "bootstrap": {"replicates": BOOT, "seed": BOOT_SEED}}
    (target / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    render(rows, methods, widths, target, args.run)
    print(target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
