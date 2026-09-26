"""Level 2 inference loop; only sensor output reaches recognition code."""

import math
import time

import numpy as np
import torch

from saccadenet.bayes.calibrate import score_query_logits
from saccadenet.bayes.fusion import FusionB
from saccadenet.bayes.policy import choose_fixation, confirmation_fixation, exploration_anchor, fixation_candidates, should_stop
from saccadenet.config import EpisodeConfig
from saccadenet.contracts import EpisodeInput, EpisodeLog, Sensor
from saccadenet.cost.accounting import CostMeter, count_fovea_flops, count_model_flops
from saccadenet.models.fovea import FoveaNet
from saccadenet.retina.detect import CandidateTracker, detect_on_retina
from saccadenet.retina.pyramid import Pyramid
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.retina.sampler import sample_retina


class SceneSensor:
    """The only object allowed to retain the canvas pyramid during inference."""

    def __init__(self, pyramid: Pyramid, *, fovea_size: int = 96, sectors: int = 128) -> None:
        self._pyramid = pyramid
        self.pyramid_add_count = pyramid.add_count
        self.pyramid_bytes = sum(level.nbytes for level in pyramid.levels)
        self.fovea_size = fovea_size
        self.sectors = sectors

    def observe(self, fixation_xy: tuple[float, float]):
        return sample_retina(self._pyramid, fixation_xy, fovea_size=self.fovea_size, sectors=self.sectors)


def run_episode(
    sensor: Sensor,
    episode_input: EpisodeInput,
    config: EpisodeConfig,
    calibrator,
    network: torch.nn.Module,
    *,
    device: torch.device,
    require_all_scored: bool = False,
) -> EpisodeLog:
    if (episode_input.width, episode_input.height, episode_input.expected_k) != (config.width, config.height, config.k):
        raise ValueError("episode input and configuration disagree")
    started = time.perf_counter()
    meter = CostMeter()
    pyramid_charge = int(getattr(sensor, "pyramid_add_count", 0))
    meter.record_pyramid_once(pyramid_charge)
    meter.add_sensing_bytes(int(getattr(sensor, "pyramid_bytes", 0)))
    one_cnn_flops = (
        count_fovea_flops(network, 96, 96)
        if isinstance(network, FoveaNet)
        else count_model_flops(network, (1, 3, 96, 96))
    )
    tracker = CandidateTracker(merge_radius=config.candidate_merge_radius)
    fusion = FusionB()
    network.eval()
    current = (config.width / 2, config.height / 2)
    visited_anchors = {current}
    trace: list[dict] = []
    reason = "max_steps"
    posterior = np.empty(0)
    for step in range(config.t_max):
        retina = sensor.observe(current)
        meter.add_sensing(max(0, int(retina.sensing_flops) - pyramid_charge))
        meter.add_sensing_bytes(int(retina.fovea.nbytes + retina.logpolar.nbytes))
        detections = detect_on_retina(retina, brightness_threshold=config.detector_threshold)
        meter.add_semantic("detector", retina.fovea.size + retina.logpolar.size)
        candidates, reset_ids = tracker.update(detections, step)
        for candidate_id in reset_ids:
            fusion.reset(candidate_id)
        views, observation_meta = [], []
        observations: list[dict] = []
        for candidate in candidates:
            eccentricity = math.dist(current, candidate.xy)
            dprime = float(calibrator.dprime(eccentricity))
            if dprime < config.dprime_min:
                continue
            view, valid = candidate_view(retina, candidate.xy)
            meter.add_semantic("reconstruction", view.size * 8)
            valid_fraction = float(valid.mean())
            if valid_fraction < 0.5:
                continue
            views.append(view)
            observation_meta.append((candidate.stable_id, candidate.xy, eccentricity, dprime, valid_fraction))
        if views:
            pixels = np.stack(views)
            batch = torch.from_numpy(pixels).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
            with torch.inference_mode():
                scores = score_query_logits(network(batch), episode_input.query).cpu().numpy()
            meter.add_semantic("cnn", one_cnn_flops * len(views))
            for (candidate_id, candidate_xy, eccentricity, dprime, valid_fraction), score in zip(observation_meta, scores):
                normalized = calibrator.normalize(float(score), eccentricity)
                observations.append({
                    "candidate_id": candidate_id,
                    "candidate_xy": candidate_xy,
                    "eccentricity": eccentricity,
                    "dprime": dprime,
                    "valid_fraction": valid_fraction,
                    "raw_score": float(score),
                    "normalized_score": float(normalized) if normalized is not None else None,
                })
                if normalized is not None:
                    fusion.update(candidate_id, normalized, dprime)
        candidate_ids = [candidate.stable_id for candidate in candidates]
        unscored_count = sum(fusion.quality(candidate_id) <= 0 for candidate_id in candidate_ids)
        posterior = fusion.posterior(candidate_ids)
        meter.add_semantic("belief", max(1, len(candidates) * 8))
        trace.append(
            {
                "step": step,
                "fixation_xy": current,
                "candidates": [{"id": item.stable_id, "xy": item.xy} for item in candidates],
                "posterior": posterior.tolist(),
                "reset_ids": sorted(reset_ids),
                "observations": observations,
                "unscored_count": unscored_count,
                "costs": meter.snapshot(),
            }
        )
        if should_stop(
            posterior,
            discovered=len(candidates),
            expected_k=episode_input.expected_k,
            tau=config.tau,
            has_evidence=any(fusion.quality(candidate_id) > 0 for candidate_id in candidate_ids),
            require_all_scored=require_all_scored,
            all_scored=unscored_count == 0,
        ):
            reason = "threshold"
            break
        if len(candidates) >= episode_input.expected_k:
            next_fixation = confirmation_fixation(candidates, fusion, current=current) if require_all_scored else None
            if next_fixation is None:
                omega = fixation_candidates(candidates, e_half=96, canvas_shape=(config.height, config.width))
                next_fixation = choose_fixation(
                    posterior, candidates, fusion, omega, current=current, dprime=calibrator.dprime
                )
                meter.add_semantic("routing", len(candidates) * max(1, len(omega)) * 10)
            else:
                meter.add_semantic("routing", len(candidates))
        else:
            next_fixation = None
        if next_fixation is None:
            next_fixation = exploration_anchor(
                config.width, config.height, grid=config.exploration_grid,
                visited=visited_anchors, current=current,
            )
            if next_fixation is None:
                reason = "search_exhausted"
                break
            visited_anchors.add(next_fixation)
        current = next_fixation
    candidates = tracker.candidates
    answer = candidates[int(np.argmax(posterior))].xy if len(posterior) else None
    if not candidates and reason == "max_steps":
        reason = "no_candidates"
    return EpisodeLog(
        answer_xy=answer,
        reason=reason,
        steps=len(trace),
        trace=trace,
        costs=meter.snapshot(),
        elapsed_seconds=time.perf_counter() - started,
    )
