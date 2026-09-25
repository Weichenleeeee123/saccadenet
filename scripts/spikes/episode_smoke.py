"""Run a small, separately labeled development smoke on the full Level 2 loop."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from saccadenet.bayes.calibrate import GaussianCalibrator
from saccadenet.config import EpisodeConfig
from saccadenet.contracts import EpisodeInput
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.run.episode import SceneSensor, run_episode
from saccadenet.run.evaluate import evaluate_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=20050)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    fit = json.loads(args.fit.read_text(encoding="utf-8"))
    calibrator = GaussianCalibrator(**fit["model"])
    bank = MnistBank.from_split("development")
    ckpt_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    fit_hash = hashlib.sha256(args.fit.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("reports/spikes") / f"episode-smoke-{stamp}-{ckpt_hash[:8]}.csv"
    rows = []
    with output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("seed", "resolution", "target_hit", "reason", "steps", "candidate_count", "sensing_flops", "semantic_flops", "seconds", "checkpoint_sha256", "fit_sha256"),
        )
        writer.writeheader()
        for index in range(args.scenes):
            seed = args.seed_start + index
            config = EpisodeConfig(width=args.width, height=args.height, query=seed % 10)
            image, truth = make_canvas(config, seed, bank)
            sensor = SceneSensor(build_pyramid(image), fovea_size=config.fovea_size, sectors=config.sectors)
            episode = run_episode(
                sensor,
                EpisodeInput(config.query, config.width, config.height, config.k),
                config,
                calibrator,
                network,
                device=device,
            )
            evaluation = evaluate_answer(episode.answer_xy, truth)
            row = {
                "seed": seed,
                "resolution": f"{config.width}x{config.height}",
                "target_hit": int(evaluation.target_hit),
                "reason": episode.reason,
                "steps": episode.steps,
                "candidate_count": len(episode.trace[-1]["candidates"]) if episode.trace else 0,
                "sensing_flops": episode.costs["sensing_flops"],
                "semantic_flops": episode.costs["semantic_flops"],
                "seconds": episode.elapsed_seconds,
                "checkpoint_sha256": ckpt_hash,
                "fit_sha256": fit_hash,
            }
            rows.append(row)
            writer.writerow(row)
            file.flush()
            print(json.dumps(row), flush=True)
    print(f"hit={sum(row['target_hit'] for row in rows)}/{len(rows)} output={output}")


if __name__ == "__main__":
    main()
