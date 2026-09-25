"""Day-0 probe: candidate discovery without any truth passed to the detector."""

import argparse
import json
import time

import numpy as np

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.retina.detect import CandidateTracker, detect_on_retina
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import sample_retina


class SyntheticDigitBank:
    def sample(self, label, rng):
        digit = np.zeros((28, 28), dtype=np.uint8)
        digit[6:22, 8:20] = 200 + 5 * label
        return digit, int(rng.integers(0, 1_000_000))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=15360)
    parser.add_argument("--height", type=int, default=8640)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid", action="store_true", help="also inspect a fixed exploration grid")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=195.0)
    parser.add_argument("--merge-radius", type=float, default=144.0)
    args = parser.parse_args()
    config = EpisodeConfig(width=args.width, height=args.height)
    start = time.perf_counter()
    image, truth = make_canvas(config, args.seed, SyntheticDigitBank())
    generated = time.perf_counter()
    pyramid = build_pyramid(image)
    pooled = time.perf_counter()
    retina = sample_retina(pyramid, (config.width / 2, config.height / 2))
    sampled = time.perf_counter()
    detections = detect_on_retina(retina, brightness_threshold=args.threshold)
    detected = time.perf_counter()
    positions = np.asarray([detection.xy for detection in detections]).reshape(-1, 2)
    distances = np.linalg.norm(truth.centers_xy[:, None, :] - positions[None, :, :], axis=-1)
    hits = [bool(np.any(row <= 48)) for row in distances]
    grid_result = None
    if args.grid:
        tracker = CandidateTracker(merge_radius=args.merge_radius)
        all_detections = list(detections)
        for row in range(args.grid_size):
            for column in range(args.grid_size):
                fixation = ((column + 0.5) * config.width / args.grid_size, (row + 0.5) * config.height / args.grid_size)
                observed = sample_retina(pyramid, fixation)
                visible = detect_on_retina(observed, brightness_threshold=args.threshold)
                all_detections.extend(visible)
                tracker.update(visible, step=1 + args.grid_size * row + column)
        positions_grid = np.asarray([detection.xy for detection in all_detections]).reshape(-1, 2)
        distances_grid = np.linalg.norm(truth.centers_xy[:, None, :] - positions_grid[None, :, :], axis=-1)
        tracker_positions = np.asarray([candidate.xy for candidate in tracker.candidates]).reshape(-1, 2)
        tracker_distances = np.linalg.norm(truth.centers_xy[:, None, :] - tracker_positions[None, :, :], axis=-1)
        grid_result = {
            "raw_detection_count": len(all_detections),
            "raw_matched_cards_48px": int(np.sum(np.any(distances_grid <= 48, axis=1))),
            "tracker_candidate_count": len(tracker.candidates),
            "tracker_matched_cards_48px": int(np.sum(np.any(tracker_distances <= 48, axis=1))),
            "grid_size": args.grid_size,
            "target_nearest_raw_px": round(float(distances_grid[truth.target_index].min()), 1),
            "target_nearest_tracker_px": round(float(tracker_distances[truth.target_index].min()), 1),
            "worst_nearest_tracker_px": round(float(np.min(tracker_distances, axis=1).max()), 1),
            "failed_matches": [
                {
                    "truth_xy": truth.centers_xy[index].tolist(),
                    "raw": [
                        {"xy": [round(v, 1) for v in detection.xy], "score": round(detection.score, 2)}
                        for detection in all_detections
                        if np.linalg.norm(np.asarray(detection.xy) - truth.centers_xy[index]) < 144
                    ],
                    "nearest_tracker_xy": [round(float(v), 1) for v in tracker_positions[int(np.argmin(tracker_distances[index]))]],
                }
                for index in range(len(truth.digits))
                if np.min(tracker_distances[index]) > 48
            ],
        }
    print(
        json.dumps(
            {
                "resolution": [args.width, args.height],
                "seed": args.seed,
                "brightness_threshold": args.threshold,
                "merge_radius": args.merge_radius,
                "truth_count": len(truth.digits),
                "detection_count": len(detections),
                "matched_cards_48px": sum(hits),
                "target_matched_48px": hits[truth.target_index],
                "exploration_grid": grid_result,
                "times_seconds": {
                    "canvas": round(generated - start, 3),
                    "pyramid": round(pooled - generated, 3),
                    "retina": round(sampled - pooled, 3),
                    "detector": round(detected - sampled, 3),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
