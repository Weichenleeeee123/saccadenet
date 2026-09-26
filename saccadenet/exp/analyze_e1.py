"""Audit the completed E1 ledger and compute preregistered paired contrasts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from saccadenet.exp.e1_resolution import planned_jobs
from saccadenet.exp.store import episode_key


def paired_bootstrap(low: np.ndarray, high: np.ndarray, *, metric: str, replicates: int = 2000) -> dict:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.shape != high.shape or low.ndim != 1 or low.size == 0 or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("paired arrays must have the same finite nonempty shape")
    rng = np.random.default_rng(40000)
    indices = rng.integers(0, len(low), (replicates, len(low)))
    low_means, high_means = low[indices].mean(axis=1), high[indices].mean(axis=1)
    if metric == "ratio":
        if np.any(low_means <= 0):
            raise ValueError("ratio denominator must be positive")
        draws = high_means / low_means
        estimate = float(high.mean() / low.mean())
    elif metric == "difference":
        draws = high_means - low_means
        estimate = float(high.mean() - low.mean())
    else:
        raise ValueError("unknown metric")
    return {"n_pairs": len(low), "estimate": estimate, "low_95": float(np.quantile(draws, 0.025)), "high_95": float(np.quantile(draws, 0.975)), "metric": metric, "bootstrap_replicates": replicates, "bootstrap_seed": 40000}


def audit_and_analyze(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    jobs = set(planned_jobs(cfg, smoke=manifest["split"] == "development"))
    latest: dict[tuple[str, int, int, int], dict] = {}
    successes: set[tuple[str, int, int, int]] = set()
    duplicate_successes = []
    all_rows = 0
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            all_rows += 1
            key = episode_key(row["method"], row["width"], row["height"], row["seed"])
            if row["status"] == "ok":
                if key in successes:
                    duplicate_successes.append(key)
                successes.add(key)
            latest[key] = row
    missing = sorted(jobs - latest.keys())
    unexpected = sorted(latest.keys() - jobs)
    failed = [key for key in jobs if key in latest and latest[key]["status"] != "ok"]
    invalid_numeric = []
    for key in jobs & latest.keys():
        row = latest[key]
        if row["status"] != "ok":
            continue
        try:
            numbers = [float(row[field]) for field in ("sensing_flops", "sensing_bytes", "semantic_flops", "episode_seconds")]
            if not all(math.isfinite(value) and value >= 0 for value in numbers) or row["target_hit"] not in ("0", "1"):
                invalid_numeric.append(key)
        except (ValueError, KeyError):
            invalid_numeric.append(key)
    seed_rows = {}
    with (run_dir / "seeds.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = int(row["width"]), int(row["height"]), int(row["seed"])
            if key in seed_rows:
                raise ValueError(f"duplicate seed manifest row {key}")
            if len(json.loads(row["digit_ids"])) != 12 or row["split"] != manifest["split"]:
                raise ValueError(f"invalid seed manifest row {key}")
            seed_rows[key] = row
    expected_seeds = {key[1:] for key in jobs}
    trace_keys = set()
    trace_errors = []
    trace_rows = 0
    for path in sorted(run_dir.glob("trace*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if "key" not in item:
                    continue
                trace_rows += 1
                key = tuple(item["key"])
                if key in trace_keys:
                    trace_errors.append((key, "duplicate"))
                trace_keys.add(key)
                if key not in latest:
                    trace_errors.append((key, "unexpected"))
                    continue
                row = latest[key]
                if int(item["attempt_id"]) != int(row["attempt_id"]):
                    trace_errors.append((key, "attempt_mismatch"))
                for field in ("semantic_flops", "sensing_flops", "sensing_bytes"):
                    if int(item["costs"].get(field, 0)) != int(float(row.get(field) or 0)):
                        trace_errors.append((key, f"cost_mismatch_{field}"))
                if key[0] in ("saccadenet_lite", "two_stage") and len(item["steps"]) != int(row["steps"]):
                    trace_errors.append((key, "step_count"))
                for field in ("semantic_flops", "sensing_flops"):
                    values = [step["costs"][field] for step in item["steps"]]
                    if any(current < previous for previous, current in zip(values, values[1:])):
                        trace_errors.append((key, f"nonmonotonic_{field}"))
    audit = {"planned_keys": len(jobs), "episode_rows": all_rows, "latest_keys": len(latest), "seed_rows": len(seed_rows), "trace_rows": trace_rows, "missing": missing, "unexpected": unexpected, "failed_latest": failed, "duplicate_successes": duplicate_successes, "invalid_numeric": invalid_numeric, "missing_seeds": sorted(expected_seeds - seed_rows.keys()), "extra_seeds": sorted(seed_rows.keys() - expected_seeds), "trace_errors": trace_errors, "missing_traces": sorted(jobs - trace_keys)}
    if any(audit[field] for field in ("missing", "unexpected", "duplicate_successes", "invalid_numeric", "missing_seeds", "extra_seeds", "trace_errors", "missing_traces")):
        raise ValueError(f"E1 audit failed: {audit}")
    if manifest["split"] != "final_test":
        return {"manifest": manifest, "audit": audit, "paired_contrasts": {}}
    def pair(method_a: str, width_a: int, method_b: str, width_b: int, field: str, count: int):
        low, high = [], []
        for seed in range(cfg["test_seed_start"], cfg["test_seed_start"] + count):
            a = latest[method_a, width_a, width_a * 9 // 16, seed]
            b = latest[method_b, width_b, width_b * 9 // 16, seed]
            low.append(float(a[field] or 0) if a["status"] == "ok" else 0)
            high.append(float(b[field] or 0) if b["status"] == "ok" else 0)
        return np.array(low), np.array(high)
    cost_1080, cost_16k = pair("saccadenet_lite", 1920, "saccadenet_lite", 15360, "semantic_flops", cfg["test_seed_count"])
    hit_1080, hit_16k = pair("saccadenet_lite", 1920, "saccadenet_lite", 15360, "target_hit", cfg["test_seed_count"])
    full_cost, saccade_cost = pair("full_res_sliding", 15360, "saccadenet_lite", 15360, "semantic_flops", cfg["full_16k_count"])
    full_hit, saccade_hit = pair("full_res_sliding", 15360, "saccadenet_lite", 15360, "target_hit", cfg["full_16k_count"])
    contrasts = {
        "g1_semantic_16k_over_1080": paired_bootstrap(cost_1080, cost_16k, metric="ratio"),
        "g1_accuracy_16k_minus_1080": paired_bootstrap(hit_1080, hit_16k, metric="difference"),
        "saccade_over_full_16k_semantic_common20": paired_bootstrap(full_cost, saccade_cost, metric="ratio"),
        "saccade_minus_full_16k_accuracy_common20": paired_bootstrap(full_hit, saccade_hit, metric="difference"),
    }
    return {"manifest": {key: manifest[key] for key in ("git_head", "git_dirty", "config_sha256", "checkpoint_sha256", "split")}, "audit": audit, "paired_contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/e1"))
    args = parser.parse_args()
    result = audit_and_analyze(args.run)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.out / f"analysis-{args.run.name}-{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps({"output": str(destination), "audit": result["audit"], "paired_contrasts": result["paired_contrasts"]}))


if __name__ == "__main__":
    main()
