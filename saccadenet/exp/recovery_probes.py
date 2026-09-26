"""Read-only counterfactual crops for development failures; truth never enters inference."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from saccadenet.bayes.calibrate import score_query_logits
from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import make_canvas
from saccadenet.data.mnist_bank import MnistBank
from saccadenet.exp.recovery_diagnostics import best_observation, summarize_trace
from saccadenet.exp.trace_analysis import checked_centers, load_traces
from saccadenet.models.fovea import FoveaNet
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.retina.sampler import sample_retina
from saccadenet.run.baselines import _crop


def _score_patch(patch: np.ndarray, query: int, model: torch.nn.Module, device: torch.device) -> dict:
    # Keep the batch dimension and strides identical to run_episode's np.stack(views).
    tensor = torch.from_numpy(np.stack([patch])).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
    with torch.inference_mode():
        logits = model(tensor)
        score = score_query_logits(logits, query)
    return {"raw_score": float(score[0].cpu()), "predicted_digit": int(logits[0].argmax().cpu())}


def replay_observation(
    image: np.ndarray,
    fixation_xy: tuple[float, float],
    candidate_xy: tuple[float, float],
    query: int,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    retina = sample_retina(build_pyramid(image), fixation_xy)
    patch, valid = candidate_view(retina, candidate_xy)
    return _score_patch(patch, query, model, device) | {"valid_fraction": float(valid.mean())}


def probe_failure(trace: dict, bank: MnistBank, model: FoveaNet, device: torch.device) -> dict:
    centers = checked_centers(trace)
    category = summarize_trace(trace, centers)["category"]
    if category == "hit":
        raise ValueError("probe_failure requires an unsuccessful episode")
    _, width, height, seed = trace["key"]
    config = EpisodeConfig(width=width, height=height, query=seed % 10)
    image, truth = make_canvas(config, seed, bank)
    if truth.target_index != trace["truth_target_index"] or not np.array_equal(truth.centers_xy, centers):
        raise ValueError("reconstructed development canvas differs from saved trace")
    target_xy = tuple(map(float, centers[truth.target_index]))
    candidates = trace["steps"][-1]["candidates"]
    nearby = sorted(
        ((math.dist(c["xy"], target_xy), c) for c in candidates),
        key=lambda pair: pair[0],
    )
    candidate = nearby[0][1] if nearby and nearby[0][0] <= config.candidate_merge_radius else None
    observation = best_observation(trace["steps"], int(candidate["id"])) if candidate else None
    if observation is not None:
        fixation = tuple(trace["steps"][int(observation["step"])]["fixation_xy"])
    else:
        fixation = tuple(min(trace["steps"], key=lambda step: math.dist(step["fixation_xy"], target_xy))["fixation_xy"])
    true_retinal = replay_observation(image, fixation, target_xy, config.query, model, device)
    true_original = _score_patch(_crop(image, target_xy), config.query, model, device)
    detected_retinal = None
    if observation is not None:
        detected_retinal = replay_observation(image, fixation, tuple(observation["candidate_xy"]), config.query, model, device)
        if abs(detected_retinal["raw_score"] - float(observation["raw_score"])) > 1e-4:
            raise ValueError("saved candidate score failed deterministic replay")
    return {
        "width": width, "height": height, "seed": seed, "category": category,
        "true_label": config.query, "target_xy": target_xy,
        "nearest_candidate_offset": nearby[0][0] if nearby else None,
        "selected_candidate_id": candidate["id"] if candidate else None,
        "selected_fixation_xy": fixation,
        "observed_candidate_retinal": detected_retinal,
        "oracle_center_retinal": true_retinal,
        "oracle_center_original": true_original,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/recovery"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoveaNet().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"])
    bank = MnistBank.from_split("development")
    traces = [t for t in load_traces(args.run) if summarize_trace(t)["category"] != "hit"]
    rows = [probe_failure(trace, bank, model, device) for trace in traces]
    destination = args.out / f"probes-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "failures.json").write_text(json.dumps({"source_run": str(args.run), "checkpoint": str(args.checkpoint), "episodes": rows}, indent=2), encoding="utf-8")
    print(destination)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
