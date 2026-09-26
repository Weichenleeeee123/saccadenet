"""Fair image-access baselines using the same foveal CNN weights."""

from dataclasses import dataclass
import time

import cv2
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from saccadenet.bayes.calibrate import score_query_logits
from saccadenet.contracts import EpisodeLog
from saccadenet.cost.accounting import CostMeter, count_fovea_flops


class _DenseAdapter(torch.nn.Module):
    def __init__(self, network: torch.nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network.dense(image)


@dataclass(frozen=True)
class PlattCalibration:
    slope: float
    intercept: float

    def probability(self, score: float) -> float:
        return float(expit(self.slope * score + self.intercept))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> PlattCalibration:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if scores.shape != labels.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("invalid Platt calibration samples")
    if not np.any(labels == 0) or not np.any(labels == 1) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Platt calibration needs positive and negative labels")

    def loss(weights: np.ndarray) -> float:
        z = weights[0] * scores + weights[1]
        return float(np.logaddexp(0, z).sum() - np.dot(labels, z) + 0.0001 * np.dot(weights, weights))

    result = minimize(loss, np.array([1.0, 0.0]), method="BFGS")
    if not np.isfinite(result.x).all():
        raise ValueError("Platt fit failed")
    return PlattCalibration(float(result.x[0]), float(result.x[1]))


def _window_starts(length: int) -> list[int]:
    if length < 96:
        raise ValueError("full-resolution side shorter than receptive field")
    positions = list(range(0, length - 96 + 1, 16))
    if positions[-1] != length - 96:
        positions.append(length - 96)
    return positions


def ideal_dense_flops(network: torch.nn.Module, width: int, height: int) -> int:
    """Analytic no-overlap dense CNN FLOPs; excludes added edge windows."""
    return count_fovea_flops(network, width, height)


def _groups(starts: list[int], tile_outputs: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for start in range(0, len(starts), tile_outputs):
        group = starts[start : start + tile_outputs]
        if len(group) >= 2 and any(b - a != 16 for a, b in zip(group, group[1:])):
            groups.append(group[:-1])
            groups.append(group[-1:])
        else:
            groups.append(group)
    return groups


def _dense_query_map_tiled(
    image: np.ndarray,
    network: torch.nn.Module,
    query: int,
    *,
    device: torch.device,
    tile_outputs: int = 48,
    collect: bool = True,
) -> tuple[dict[tuple[int, int], float], dict[str, int | float | tuple[float, float] | None]]:
    """Scan every stride-16 window and the final image edges; profile computed tiles."""
    if tile_outputs <= 0:
        raise ValueError("tile_outputs must be positive")
    height, width = image.shape[:2]
    x_groups = _groups(_window_starts(width), tile_outputs)
    y_groups = _groups(_window_starts(height), tile_outputs)
    adapter = _DenseAdapter(network).to(device).eval()
    flops_cache: dict[tuple[int, int], int] = {}
    scores: dict[tuple[int, int], float] = {}
    best_score, best_xy = -float("inf"), None
    actual_flops = 0
    input_bytes = 0
    tile_count = 0
    for ys in y_groups:
        for xs in x_groups:
            left, top = xs[0], ys[0]
            patch = np.ascontiguousarray(image[top : ys[-1] + 96, left : xs[-1] + 96])
            input_bytes += patch.nbytes
            tensor = torch.from_numpy(patch).permute(2, 0, 1)[None].to(device=device, dtype=torch.float32).div_(255)
            shape = (tensor.shape[2], tensor.shape[3])
            if shape not in flops_cache:
                flops_cache[shape] = count_fovea_flops(network, shape[1], shape[0])
            with torch.inference_mode():
                logits = adapter(tensor)[0]
                raw = score_query_logits(logits.permute(1, 2, 0).reshape(-1, 11), query)
                values = raw.reshape(len(ys), len(xs)).cpu().numpy()
            actual_flops += flops_cache[shape]
            tile_count += 1
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    score = float(values[yi, xi])
                    if collect:
                        scores[x, y] = score
                    if score > best_score:
                        best_score = score
                        best_xy = (float(x + 48), float(y + 48))
    return scores, {"semantic_flops": actual_flops, "input_bytes": input_bytes, "tiles": tile_count, "best_score": best_score, "best_xy": best_xy}


def _coarse_card_centers(image: np.ndarray, *, threshold: float = 195) -> list[tuple[float, float]]:
    gray = image.mean(axis=2)
    mask = (gray >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= 16:
            x, y = centroids[index]
            result.append((float(x), float(y)))
    return result


def _crop(image: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    h, w = image.shape[:2]
    x = int(np.clip(round(center[0] - 48), 0, w - 96))
    y = int(np.clip(round(center[1] - 48), 0, h - 96))
    return np.ascontiguousarray(image[y : y + 96, x : x + 96])


def _crop_scores(
    image: np.ndarray,
    centers: list[tuple[float, float]],
    network: torch.nn.Module,
    query: int,
    device: torch.device,
) -> tuple[list[float], int]:
    if not centers:
        return [], 0
    one_flops = count_fovea_flops(network, 96, 96)
    scores: list[float] = []
    for offset in range(0, len(centers), 64):
        patches = np.stack([_crop(image, xy) for xy in centers[offset : offset + 64]])
        tensor = torch.from_numpy(patches).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
        with torch.inference_mode():
            scores.extend(score_query_logits(network(tensor), query).cpu().tolist())
    return scores, one_flops * len(centers)


def run_full_res_sliding(
    image: np.ndarray, query: int, network: torch.nn.Module, *, device: torch.device, tile_outputs: int = 48
) -> EpisodeLog:
    started = time.perf_counter()
    _, result = _dense_query_map_tiled(image, network, query, device=device, tile_outputs=tile_outputs, collect=False)
    costs = {"sensing_flops": 0, "sensing_bytes": int(result["input_bytes"]), "semantic_flops": int(result["semantic_flops"]), "semantic_cnn_flops": int(result["semantic_flops"])}
    return EpisodeLog(result["best_xy"], "exhaustive", int(result["tiles"]), [], costs, time.perf_counter() - started)


def _downsample(image: np.ndarray, width: int = 1024) -> tuple[np.ndarray, float, float, int]:
    height, original_width = image.shape[:2]
    reduced_height = max(96, round(height * width / original_width))
    small = cv2.resize(image, (width, reduced_height), interpolation=cv2.INTER_AREA)
    # Analytic lower-bound arithmetic estimate; OpenCV's exact SIMD operations are not introspectable.
    cost = int(height * original_width * 3 * 2)
    return small, original_width / width, height / reduced_height, cost


def run_downsample_1stage(
    image: np.ndarray, query: int, network: torch.nn.Module, *, device: torch.device, tile_outputs: int = 48
) -> EpisodeLog:
    started = time.perf_counter()
    small, scale_x, scale_y, resize_flops = _downsample(image)
    _, result = _dense_query_map_tiled(small, network, query, device=device, tile_outputs=tile_outputs, collect=False)
    xy = result["best_xy"]
    answer = (xy[0] * scale_x, xy[1] * scale_y) if xy is not None else None
    costs = {"sensing_flops": resize_flops, "sensing_bytes": int(image.nbytes + result["input_bytes"]), "semantic_flops": int(result["semantic_flops"]), "semantic_cnn_flops": int(result["semantic_flops"])}
    return EpisodeLog(answer, "exhaustive", int(result["tiles"]), [], costs, time.perf_counter() - started)


def run_two_stage(
    image: np.ndarray,
    query: int,
    network: torch.nn.Module,
    *,
    device: torch.device,
    threshold: float = 0.95,
    probability_calibration: PlattCalibration | None = None,
    coarse_ranking: bool = True,
    overview_width: int = 1024,
) -> EpisodeLog:
    started = time.perf_counter()
    small, scale_x, scale_y, resize_flops = _downsample(image, width=overview_width)
    coarse = _coarse_card_centers(small)
    meter = CostMeter()
    meter.add_sensing(resize_flops)
    meter.add_sensing_bytes(image.nbytes + small.nbytes)
    meter.add_semantic("detector", small.shape[0] * small.shape[1] * 4)
    if not coarse:
        return EpisodeLog(None, "no_candidates", 0, [], meter.snapshot(), time.perf_counter() - started)
    if coarse_ranking:
        coarse_scores, flops = _crop_scores(small, coarse, network, query, device)
        meter.add_sensing_bytes(len(coarse) * 96 * 96 * 3)
        meter.add_semantic("cnn", flops)
        ordered = sorted(range(len(coarse)), key=lambda index: (-coarse_scores[index], index))
    else:
        ordered = range(len(coarse))
    best_score, best_xy = -float("inf"), None
    trace = []
    one_flops = count_fovea_flops(network, 96, 96)
    reason = "exhausted"
    for rank, index in enumerate(ordered):
        xy = (coarse[index][0] * scale_x, coarse[index][1] * scale_y)
        high_scores, _ = _crop_scores(image, [xy], network, query, device)
        meter.add_sensing_bytes(96 * 96 * 3)
        meter.add_semantic("cnn", one_flops)
        meter.add_semantic("routing", 4)
        score = high_scores[0]
        if score > best_score:
            best_score, best_xy = score, xy
        probability = (
            probability_calibration.probability(score)
            if probability_calibration is not None
            else float(expit(score))
        )
        trace.append({"step": rank, "fixation_xy": xy, "query_probability": probability, "costs": meter.snapshot()})
        if probability >= threshold:
            reason = "threshold"
            break
    return EpisodeLog(best_xy, reason, len(trace), trace, meter.snapshot(), time.perf_counter() - started)
