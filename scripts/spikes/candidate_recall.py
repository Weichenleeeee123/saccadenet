"""Measure Level 2 candidate recall on development MNIST canvases."""

import csv
from pathlib import Path
import statistics

import numpy as np

from saccadenet.config import EpisodeConfig, RESOLUTIONS
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.retina.detect import CandidateTracker, detect_on_retina
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina


SEEDS = range(20_000, 20_020)
GRID = 5
THRESHOLD = 195.0
MERGE_RADIUS = 144.0


def measure(width: int, height: int, seed: int, bank: MnistBank) -> dict:
    config = EpisodeConfig(width=width, height=height)
    canvas, truth = make_canvas(config, seed, bank)
    pyramid = build_pyramid(canvas)
    tracker = CandidateTracker(merge_radius=MERGE_RADIUS)
    centers = [(width / 2, height / 2)]
    centers.extend(
        ((column + 0.5) * width / GRID, (row + 0.5) * height / GRID)
        for row in range(GRID)
        for column in range(GRID)
    )
    for step, fixation in enumerate(centers):
        retina = sample_retina(pyramid, fixation)
        tracker.update(detect_on_retina(retina, brightness_threshold=THRESHOLD), step)
    candidates = np.asarray([candidate.xy for candidate in tracker.candidates], dtype=np.float32).reshape(-1, 2)
    if len(candidates):
        distance = np.linalg.norm(truth.centers_xy[:, None, :] - candidates[None, :, :], axis=-1)
        closest = distance.min(axis=1)
    else:
        closest = np.full(config.k, np.inf)
    return {
        "width": width,
        "height": height,
        "seed": seed,
        "truth_k": config.k,
        "candidate_count": len(candidates),
        "matched_48px": int(np.sum(closest <= 48)),
        "matched_20px": int(np.sum(closest <= 20)),
        "target_found_48px": int(closest[truth.target_index] <= 48),
        "target_found_20px": int(closest[truth.target_index] <= 20),
        "nonmatched_candidates_48px": int(sum(np.min(np.linalg.norm(truth.centers_xy - point, axis=1)) > 48 for point in candidates)),
        "fixations": len(centers),
    }


def main() -> None:
    bank = MnistBank.from_split("development")
    output = Path("reports/spikes/candidate_recall.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = None
        for width, height in RESOLUTIONS:
            rows = []
            for seed in SEEDS:
                row = measure(width, height, seed, bank)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                rows.append(row)
            print(
                f"{width}x{height}: targets={sum(row['target_found_48px'] for row in rows)}/{len(rows)}, "
                f"cards={sum(row['matched_48px'] for row in rows)}/{sum(row['truth_k'] for row in rows)}, "
                f"mean_candidates={statistics.mean(row['candidate_count'] for row in rows):.2f}, "
                f"mean_false={statistics.mean(row['nonmatched_candidates_48px'] for row in rows):.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
