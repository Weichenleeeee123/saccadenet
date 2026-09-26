"""Describe Level 2 candidate and evidence failures from saved development traces."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from saccadenet.exp.trace_analysis import checked_centers, load_traces


def _nearest(xy, centers: np.ndarray) -> tuple[int, float]:
    distances = np.linalg.norm(centers - np.asarray(xy, dtype=float), axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def best_observation(steps: list[dict], candidate_id: int) -> dict | None:
    """Match FusionB's highest-quality rule, including candidate resets."""
    best = None
    for step in steps:
        if candidate_id in step.get("reset_ids", []):
            best = None
        for observation in step.get("observations", []):
            if observation["candidate_id"] != candidate_id or observation["normalized_score"] is None:
                continue
            if best is None or float(observation["dprime"]) > float(best["dprime"]):
                best = observation | {"step": step.get("step")}
    return best


def summarize_trace(trace: dict, centers: np.ndarray | None = None) -> dict:
    """Classify observations descriptively; truth is used only after inference."""
    if centers is None:
        centers = checked_centers(trace)
    target_index = int(trace["truth_target_index"])
    final = trace["steps"][-1] if trace["steps"] else {"candidates": [], "posterior": []}
    candidates = final["candidates"]
    posterior = final["posterior"]
    if len(candidates) != len(posterior):
        raise ValueError("candidate and posterior counts differ")
    winner = candidates[int(np.argmax(posterior))] if candidates else None
    winner_card, winner_offset = _nearest(winner["xy"], centers) if winner else (None, None)
    target_candidates = [c for c in candidates if (match := _nearest(c["xy"], centers))[0] == target_index and match[1] <= 48]
    target = min(target_candidates, key=lambda c: _nearest(c["xy"], centers)[1], default=None)
    target_obs = best_observation(trace["steps"], int(target["id"])) if target else None
    winner_obs = best_observation(trace["steps"], int(winner["id"])) if winner else None
    hit = winner_card == target_index and winner_offset is not None and winner_offset <= 48
    if hit:
        category = "hit"
    elif target is None:
        category = "target_not_recalled"
    elif target_obs is None:
        category = "target_not_scored"
    elif trace["reason"] == "threshold" and max(posterior, default=0) >= 0.95:
        category = "confident_wrong"
    else:
        category = "wrong_other_stop"
    return {
        "width": int(trace["key"][1]), "height": int(trace["key"][2]), "seed": int(trace["key"][3]),
        "category": category, "reason": trace["reason"], "steps": len(trace["steps"]),
        "target_recalled": target is not None, "target_scored": target_obs is not None,
        "target_offset": _nearest(target["xy"], centers)[1] if target else None,
        "winner_offset": winner_offset, "winner_card_index": winner_card,
        "winner_probability": max(posterior, default=None),
        "target_best_raw_score": target_obs["raw_score"] if target_obs else None,
        "winner_best_raw_score": winner_obs["raw_score"] if winner_obs else None,
        "target_best_valid_fraction": target_obs["valid_fraction"] if target_obs else None,
        "winner_best_valid_fraction": winner_obs["valid_fraction"] if winner_obs else None,
        "target_best_eccentricity": target_obs["eccentricity"] if target_obs else None,
        "winner_best_eccentricity": winner_obs["eccentricity"] if winner_obs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/recovery"))
    args = parser.parse_args()
    traces = load_traces(args.run)
    if not traces:
        raise ValueError("no SaccadeNet traces")
    rows = [summarize_trace(trace) for trace in traces]
    destination = args.out / f"diagnostics-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "episodes.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"source_run": str(args.run), "n": len(rows), "categories": dict(Counter(row["category"] for row in rows))}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(destination)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
