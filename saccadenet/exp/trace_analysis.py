"""Post-hoc descriptive analyses of saved Level 2 traces (no inference, no tuning).

1. Detection horizon: probability that a not-yet-discovered card is discovered at a glimpse,
   as a function of its eccentricity from that fixation.
2. Failure anatomy: localization offset of the answered candidate in hits vs misses.
Card centers are recomputed from the recorded seed via the independent layout stream.
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
from saccadenet.data.canvas import _place_centers
from saccadenet.retina.horizon import derived_grid, detection_horizon

MATCH = 144.0  # tracker merge radius; a candidate within it is attributed to that card
EDGES = np.array([0, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 12000, 18000], dtype=float)


def card_centers(width: int, height: int, seed: int, k: int = 12) -> np.ndarray:
    config = EpisodeConfig(width=width, height=height, k=k)
    layout_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(4)[1])
    return _place_centers(config, layout_rng).astype(float)


def checked_centers(trace: dict) -> np.ndarray:
    _, width, height, seed = trace["key"]
    centers = card_centers(width, height, seed, trace.get("_k", 12))
    if not np.allclose(centers[int(trace["truth_target_index"])], trace["truth_target_xy"]):
        raise ValueError(f"layout regeneration does not match saved truth for {trace['key']}")
    return centers


def _attribute(xy, centers: np.ndarray) -> int | None:
    distances = np.hypot(centers[:, 0] - xy[0], centers[:, 1] - xy[1])
    index = int(np.argmin(distances))
    return index if distances[index] <= MATCH else None


def load_traces(run_dir: Path, method: str = "saccadenet_lite") -> list[dict]:
    """Latest attempt per key for one method."""
    items = {}
    for path in sorted(run_dir.glob("trace*.jsonl")):
        with path.open(encoding="utf-8") as file:
            for line in file:
                item = json.loads(line)
                if item.get("key", [None])[0] == method:
                    key = tuple(item["key"])
                    if key not in items or item["attempt_id"] > items[key]["attempt_id"]:
                        items[key] = item
    return [items[key] for key in sorted(items)]


def horizon_events(trace: dict) -> list[tuple[int, float, int, int]]:
    """Return (width, eccentricity, discovered, step) for every undiscovered card at every glimpse."""
    _, width, height, seed = trace["key"]
    centers = checked_centers(trace)
    known: set[int] = set()
    events = []
    for step in trace["steps"]:
        fx, fy = step["fixation_xy"]
        now = {card for card in (_attribute(c["xy"], centers) for c in step["candidates"]) if card is not None}
        for card in range(len(centers)):
            if card in known:
                continue
            eccentricity = math.hypot(centers[card, 0] - fx, centers[card, 1] - fy)
            events.append((width, eccentricity, int(card in now), int(step["step"]) if "step" in step else len(events)))
        known |= now
    return events


def answer_offsets(trace: dict) -> dict:
    _, width, height, seed = trace["key"]
    centers = checked_centers(trace)
    final = trace["steps"][-1]
    answer = trace["answer_xy"]
    chosen = min(final["candidates"], key=lambda c: math.dist(c["xy"], answer)) if final["candidates"] and answer else None
    card = _attribute(chosen["xy"], centers) if chosen else None
    target = int(trace["truth_target_index"])
    return {
        "width": width, "seed": seed, "steps": len(trace["steps"]), "reason": trace["reason"],
        "hit": int(answer is not None and math.dist(answer, trace["truth_target_xy"]) <= 48),
        "answered_card_is_target": int(card == target),
        "candidate_offset_px": math.dist(chosen["xy"], centers[card]) if card is not None else float("nan"),
        "max_posterior": max(final["posterior"], default=0.0),
        "first_glimpse_visible": 0,
    }


def trace_grid(trace: dict) -> int:
    method, width, height, _ = trace["key"]
    return derived_grid(width, height, detection_horizon()[0]) if method == "saccadenet_derived_grid" else 5


def glimpse_decomposition(trace: dict) -> dict:
    """Split glimpses into exploration (at a grid anchor or the start) and verification (elsewhere)."""
    _, width, height, _ = trace["key"]
    grid = trace_grid(trace)
    anchors = {((c + 0.5) * width / grid, (r + 0.5) * height / grid) for c in range(grid) for r in range(grid)} | {(width / 2, height / 2)}
    fixations = [tuple(step["fixation_xy"]) for step in trace["steps"]]
    explore = sum(1 for f in fixations if any(abs(f[0] - x) < 1e-6 and abs(f[1] - y) < 1e-6 for x, y in anchors))
    all_found = next((index + 1 for index, step in enumerate(trace["steps"]) if len(step["candidates"]) >= trace.get("_k", 12)), None)
    return {"grid": grid, "explore_glimpses": explore, "verify_glimpses": len(fixations) - explore, "step_all_candidates": all_found if all_found else ""}


def first_glimpse_visible(trace: dict) -> int:
    _, width, height, seed = trace["key"]
    centers = checked_centers(trace)
    first = trace["steps"][0]
    return len({card for card in (_attribute(c["xy"], centers) for c in first["candidates"]) if card is not None})


def analyze(run_dir: Path, out: Path, method: str = "saccadenet_lite") -> Path:
    traces = load_traces(run_dir, method)
    if not traces:
        raise ValueError("no SaccadeNet traces found")
    k = int(json.loads((run_dir / "config.json").read_text(encoding="utf-8")).get("k", 12))
    for trace in traces:
        trace["_k"] = k
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = out / f"trace-analysis-{run_dir.name}-{method}-{stamp}"
    target.mkdir(parents=True, exist_ok=False)

    events = [event for trace in traces for event in horizon_events(trace)]
    ecc = np.array([e[1] for e in events])
    hit = np.array([e[2] for e in events])
    rows = []
    for low, high in zip(EDGES[:-1], EDGES[1:]):
        mask = (ecc >= low) & (ecc < high)
        n = int(mask.sum())
        rows.append({"ecc_low": low, "ecc_high": high, "exposures": n, "discoveries": int(hit[mask].sum()), "p_discover": float(hit[mask].mean()) if n else float("nan")})
    with (target / "horizon.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_episode = []
    for trace in traces:
        item = answer_offsets(trace)
        item["first_glimpse_visible"] = first_glimpse_visible(trace)
        item.update(glimpse_decomposition(trace))
        item["method"] = trace["key"][0]
        per_episode.append(item)
    with (target / "episodes.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(per_episode[0]))
        writer.writeheader()
        writer.writerows(per_episode)

    widths = sorted({item["width"] for item in per_episode})
    summary = {"source_run": run_dir.name, "episodes": len(per_episode), "events": len(events), "by_width": {}}
    for width in widths:
        group = [item for item in per_episode if item["width"] == width]
        misses = [item for item in group if not item["hit"]]
        hits = [item for item in group if item["hit"]]
        summary["by_width"][str(width)] = {
            "n": len(group), "mean_first_glimpse_visible": float(np.mean([i["first_glimpse_visible"] for i in group])),
            "mean_steps": float(np.mean([i["steps"] for i in group])),
            "mean_explore_glimpses": float(np.mean([i["explore_glimpses"] for i in group])),
            "mean_verify_glimpses": float(np.mean([i["verify_glimpses"] for i in group])),
            "hit_offset_median": float(np.nanmedian([i["candidate_offset_px"] for i in hits])) if hits else None,
            "miss_offsets": [round(i["candidate_offset_px"], 1) for i in misses],
            "miss_answered_wrong_card": sum(1 - i["answered_card_is_target"] for i in misses),
        }
    all_hits = np.array([i["candidate_offset_px"] for i in per_episode if i["hit"]])
    all_miss = np.array([i["candidate_offset_px"] for i in per_episode if not i["hit"]])
    summary["offset_hits_quantiles"] = np.nanquantile(all_hits, [0.5, 0.9, 0.99]).tolist()
    summary["offset_miss_values"] = sorted(np.round(all_miss, 1).tolist())
    summary["miss_offset_above_hit_p90"] = int(np.sum(all_miss > np.nanquantile(all_hits, 0.9)))
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.86, wspace=0.28)
    centers = [(r["ecc_low"] + r["ecc_high"]) / 2 for r in rows if r["exposures"]]
    probability = [r["p_discover"] for r in rows if r["exposures"]]
    r_certain, r_possible = detection_horizon()
    left.axvspan(10, r_certain, color="#83c5be", alpha=0.18, lw=0)
    left.axvspan(r_certain, r_possible, color="#ffb703", alpha=0.15, lw=0)
    left.text(r_certain * 0.97, 0.5, f"r1={r_certain:.0f}", ha="right", fontsize=8, color="#006d77", rotation=90, va="center")
    left.text(r_possible * 0.97, 0.5, f"r2={r_possible:.0f}", ha="right", fontsize=8, color="#b07d00", rotation=90, va="center")
    left.plot(centers, probability, marker="o", color="#006d77", label="measured (binned)")
    for width, label in ((1920, "1080p"), (3840, "4K"), (7680, "8K"), (15360, "16K")):
        half_diag = math.hypot(width, width * 9 / 16) / 2
        left.axvline(half_diag, color="#999999", linestyle=":", linewidth=1)
        left.text(half_diag, -0.13, label, ha="center", fontsize=7, color="#666666", transform=left.get_xaxis_transform())
    left.set_xlim(100, 20000)
    left.set_xscale("log")
    left.set_ylim(-0.03, 1.05)
    left.set_xlabel("Eccentricity from fixation (px); dotted = canvas half-diagonal", labelpad=14)
    left.set_ylabel("P(discovered at this glimpse)")
    left.set_title("Detection horizon: analytic bands vs measured", fontsize=10)
    names = {1920: "1080p", 3840: "4K", 7680: "8K", 15360: "16K", 24576: "24K"}
    explore = [summary["by_width"][str(w)]["mean_explore_glimpses"] for w in widths]
    verify = [summary["by_width"][str(w)]["mean_verify_glimpses"] for w in widths]
    first = [summary["by_width"][str(w)]["mean_first_glimpse_visible"] for w in widths]
    x = np.arange(len(widths))
    right.bar(x, verify, width=0.55, color="#d1495b", label="Verification glimpses (off-grid)")
    right.bar(x, explore, width=0.55, bottom=verify, color="#006d77", label="Exploration glimpses (start + grid anchors)")
    right.axhline(6.5, color="#d1495b", linestyle=":", linewidth=1)
    right.text(-0.45, 7.0, "(K+1)/2 = 6.5", color="#d1495b", fontsize=8, ha="left", va="bottom")
    for xi, value in zip(x, first):
        right.text(xi, verify[xi] + explore[xi] + 0.6, f"{value:.1f}/12 seen\nat glimpse 1", ha="center", fontsize=7, color="#555555")
    right.set_xticks(x, [names.get(w, str(w)) for w in widths])
    right.set_ylim(0, max(v + e for v, e in zip(verify, explore)) * 1.25)
    right.set_ylabel("Mean glimpses per episode")
    right.legend(fontsize=8, frameon=False, loc="upper left")
    right.set_title("Glimpse budget: verify is flat, explore grows", fontsize=10)
    fig.text(0.07, 0.03, f"Post-hoc descriptive analysis of saved traces; run {run_dir.name}; {len(events)} card-glimpse exposures.", fontsize=7)
    fig.savefig(target / "figure6-horizon.png", dpi=180)
    fig.savefig(target / "figure6-horizon.svg")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/analysis"))
    parser.add_argument("--method", default="saccadenet_lite")
    args = parser.parse_args()
    print(analyze(args.run, args.out, args.method))


if __name__ == "__main__":
    main()
