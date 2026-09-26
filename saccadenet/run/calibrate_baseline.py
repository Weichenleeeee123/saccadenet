"""Fit the two-stage stopping probability on calibration-only true card crops."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet
from saccadenet.run.baselines import _crop_scores, fit_platt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=10000)
    args = parser.parse_args()
    if args.scenes <= 0 or not 10000 <= args.seed_start or args.seed_start + args.scenes > 11000:
        raise ValueError("Platt scenes must use the registered calibration seeds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    bank = MnistBank.from_split("calibration")
    scores, labels = [], []
    for seed in range(args.seed_start, args.seed_start + args.scenes):
        config = EpisodeConfig(query=seed % 10)
        image, truth = make_canvas(config, seed, bank)
        scene_scores, _ = _crop_scores(image, [tuple(xy) for xy in truth.centers_xy], network, config.query, device)
        scores.extend(scene_scores)
        labels.extend(int(index == truth.target_index) for index in range(config.k))
    calibration = fit_platt(np.array(scores), np.array(labels))
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("reports/calibration") / f"two-stage-platt-{stamp}-{checkpoint_hash[:8]}.json"
    probabilities = np.array([calibration.probability(score) for score in scores])
    payload = {
        "checkpoint_sha256": checkpoint_hash,
        "split": "calibration",
        "seed_start": args.seed_start,
        "scenes": args.scenes,
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
        "slope": calibration.slope,
        "intercept": calibration.intercept,
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "stop_threshold": 0.95,
        "note": "Oracle card centers used only for calibration labels/crops; inference uses coarse detector.",
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps({"output": str(output), **payload}))


if __name__ == "__main__":
    main()
