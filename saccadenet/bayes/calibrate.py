"""Eccentricity-conditional Gaussian evidence calibration."""

import argparse
import bisect
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml


def score_query_logits(logits: torch.Tensor, query: int) -> torch.Tensor:
    if logits.ndim != 2 or not 0 <= query < logits.shape[1]:
        raise ValueError("query must select one logit column")
    others = torch.cat((logits[:, :query], logits[:, query + 1 :]), dim=1)
    return logits[:, query] - torch.logsumexp(others, dim=1)


def choose_inbounds_fixation(
    center: tuple[float, float] | np.ndarray,
    *,
    eccentricity: float,
    canvas_shape: tuple[int, int],
    phase: float = 0.0,
) -> tuple[float, float] | None:
    height, width = canvas_shape
    for offset in range(128):
        angle = phase + offset * 2 * math.pi / 128
        xy = (float(center[0] + eccentricity * math.cos(angle)), float(center[1] + eccentricity * math.sin(angle)))
        if 0 <= xy[0] < width and 0 <= xy[1] < height:
            return xy
    return None


@dataclass(frozen=True)
class GaussianCalibrator:
    edges: tuple[float, ...]
    mu0: tuple[float, ...]
    mu1: tuple[float, ...]
    sigma: tuple[float, ...]
    count0: tuple[int, ...]
    count1: tuple[int, ...]

    def _index(self, eccentricity: float) -> int:
        # Same bin as clip(searchsorted(edges, e, side="right") - 1, 0, bins - 1), without numpy overhead.
        return min(max(bisect.bisect_right(self.edges, eccentricity) - 1, 0), len(self.mu0) - 1)

    def dprime(self, eccentricity: float) -> float:
        index = self._index(eccentricity)
        return max(0.0, (self.mu1[index] - self.mu0[index]) / self.sigma[index])

    def normalize(self, score: float, eccentricity: float, minimum_dprime: float = 0.1) -> float | None:
        index = self._index(eccentricity)
        if self.dprime(eccentricity) < minimum_dprime:
            return None
        evidence = (score - self.mu0[index]) / (self.mu1[index] - self.mu0[index])
        if not np.isfinite(evidence):
            raise ValueError("nonfinite calibrated evidence")
        return float(evidence)


def fit_gaussian(
    scores: np.ndarray,
    labels: np.ndarray,
    eccentricities: np.ndarray,
    *,
    edges: tuple[float, ...],
) -> GaussianCalibrator:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    eccentricities = np.asarray(eccentricities, dtype=np.float64)
    if not (len(scores) == len(labels) == len(eccentricities)) or not np.all(np.isfinite(scores)):
        raise ValueError("scores, labels, eccentricities must be finite aligned arrays")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must strictly increase")
    means0, means1, sigmas, counts0, counts1 = [], [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        within = (eccentricities >= low) & (eccentricities < high)
        negative = scores[within & (labels == 0)]
        positive = scores[within & (labels == 1)]
        if len(negative) < 2 or len(positive) < 2:
            raise ValueError(f"class samples missing in eccentricity bin [{low}, {high})")
        means0.append(float(negative.mean()))
        means1.append(float(positive.mean()))
        pooled = np.sqrt((negative.var(ddof=1) + positive.var(ddof=1)) / 2)
        sigmas.append(float(max(pooled, 1e-6)))
        counts0.append(len(negative))
        counts1.append(len(positive))
    return GaussianCalibrator(tuple(edges), tuple(means0), tuple(means1), tuple(sigmas), tuple(counts0), tuple(counts1))


def main() -> None:
    from saccadenet.config import EpisodeConfig
    from saccadenet.data.canvas import make_canvas
    from saccadenet.data.mnist_bank import MnistBank
    from saccadenet.models.fovea import FoveaNet
    from saccadenet.retina.pyramid import build_pyramid
    from saccadenet.retina.reconstruct import candidate_view
    from saccadenet.retina.sampler import sample_retina

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/calibration.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scene_count = args.scenes if args.scenes is not None else int(cfg["scenes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = FoveaNet().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(state["model"])
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    bank = MnistBank.from_split("calibration")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"calibration-{stamp}-{checkpoint_hash[:8]}"
    out_dir = Path("reports/calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    csv_path = out_dir / f"{run_name}-scores.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("seed", "candidate_index", "digit_id", "is_target", "eccentricity", "score", "valid_fraction"))
        writer.writeheader()
        for scene_index in range(scene_count):
            seed = int(cfg["seed_start"]) + scene_index
            config = EpisodeConfig(query=scene_index % 10)
            canvas, truth = make_canvas(config, seed, bank)
            pyramid = build_pyramid(canvas)
            rng = np.random.default_rng(seed)
            negative_id = next(index for index in rng.permutation(config.k) if index != truth.target_index)
            observations = []
            for candidate_id in (truth.target_index, negative_id):
                center = truth.centers_xy[candidate_id]
                for eccentricity in cfg["eccentricities"]:
                    fixation = choose_inbounds_fixation(
                        center,
                        eccentricity=float(eccentricity),
                        canvas_shape=(config.height, config.width),
                        phase=float(rng.uniform(0, 2 * math.pi)),
                    )
                    if fixation is None:
                        continue
                    retina = sample_retina(pyramid, fixation)
                    view, valid = candidate_view(retina, tuple(center))
                    if valid.mean() < 0.5:
                        continue
                    observations.append((candidate_id, eccentricity, view, float(valid.mean())))
            if not observations:
                continue
            pixels = np.stack([item[2] for item in observations])
            patches = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
            with torch.inference_mode():
                scores = score_query_logits(network(patches), config.query).cpu().numpy()
            for (candidate_id, eccentricity, _, valid_fraction), score in zip(observations, scores):
                row = {
                    "seed": seed,
                    "candidate_index": candidate_id,
                    "digit_id": truth.digit_ids[candidate_id],
                    "is_target": int(candidate_id == truth.target_index),
                    "eccentricity": eccentricity,
                    "score": float(score),
                    "valid_fraction": valid_fraction,
                }
                writer.writerow(row)
                rows.append(row)
            output.flush()
            if (scene_index + 1) % 50 == 0:
                print(f"calibration scenes={scene_index + 1}/{scene_count} rows={len(rows)}", flush=True)
    if scene_count < 2:
        print(f"collected {len(rows)} observations in {csv_path}; fitting requires at least two scenes")
        return
    calibration = fit_gaussian(
        np.asarray([row["score"] for row in rows]),
        np.asarray([row["is_target"] for row in rows]),
        np.asarray([row["eccentricity"] for row in rows]),
        edges=tuple(float(edge) for edge in cfg["edges"]),
    )
    fit_path = out_dir / f"{run_name}-fit.json"
    with fit_path.open("x", encoding="utf-8") as output:
        json.dump(
            {"model": asdict(calibration), "checkpoint_sha256": checkpoint_hash, "source": "MNIST/train/calibration", "scenes": scene_count},
            output,
            indent=2,
        )
    print(json.dumps({"fit": str(fit_path), "dprime": [calibration.dprime(e) for e in cfg["eccentricities"]]}), flush=True)


if __name__ == "__main__":
    main()
