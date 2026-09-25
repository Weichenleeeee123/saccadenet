"""Development-only baseline comparison with persistent per-scene rows."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.models.fovea import FoveaNet
from saccadenet.run.baselines import run_downsample_1stage, run_full_res_sliding, run_two_stage
from saccadenet.run.evaluate import evaluate_answer


METHODS = {
    "full_res_sliding": run_full_res_sliding,
    "downsample_1stage": run_downsample_1stage,
    "two_stage": run_two_stage,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20050)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--tile-outputs", type=int, default=48)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    network = FoveaNet().to(device).eval()
    network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    bank = MnistBank.from_split("development")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = Path("reports/spikes") / f"baseline-smoke-{stamp}-{checkpoint_hash[:8]}.csv"
    with output.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("seed", "resolution", "method", "target_hit", "reason", "steps", "sensing_flops", "semantic_flops", "episode_seconds", "canvas_seconds", "checkpoint_sha256"))
        writer.writeheader()
        for offset in range(args.scenes):
            seed = args.seed_start + offset
            config = EpisodeConfig(width=args.width, height=args.height, query=seed % 10)
            start = time.perf_counter()
            image, truth = make_canvas(config, seed, bank)
            canvas_seconds = time.perf_counter() - start
            for name in args.methods:
                kwargs = {"tile_outputs": args.tile_outputs} if name != "two_stage" else {}
                result = METHODS[name](image, config.query, network, device=device, **kwargs)
                evaluation = evaluate_answer(result.answer_xy, truth)
                row = {"seed": seed, "resolution": f"{args.width}x{args.height}", "method": name, "target_hit": int(evaluation.target_hit), "reason": result.reason, "steps": result.steps, "sensing_flops": result.costs["sensing_flops"], "semantic_flops": result.costs["semantic_flops"], "episode_seconds": result.elapsed_seconds, "canvas_seconds": canvas_seconds, "checkpoint_sha256": checkpoint_hash}
                writer.writerow(row)
                file.flush()
                print(json.dumps(row), flush=True)
    print(f"output={output}")


if __name__ == "__main__":
    main()
