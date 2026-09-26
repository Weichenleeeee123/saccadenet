import numpy as np
import torch

from saccadenet.config import EpisodeConfig
from saccadenet.contracts import EpisodeInput, SceneTruth
from saccadenet.exp.e1_resolution import run_job
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.run.episode import SceneSensor, run_episode
from saccadenet.run.evaluate import evaluate_answer


class ClearCalibrator:
    def dprime(self, eccentricity):
        return 2.0 if eccentricity < 100 else 0.0

    def normalize(self, score, eccentricity, minimum_dprime=0.1):
        return 1.0


class NarrowCalibrator(ClearCalibrator):
    def dprime(self, eccentricity):
        return 4.0 if eccentricity < 100 else 0.0


class BrightCardNet(torch.nn.Module):
    def forward(self, patches):
        logits = torch.zeros((len(patches), 11), device=patches.device)
        logits[:, 0] = patches.mean(dim=(1, 2, 3)) * 10
        return logits


def test_one_found_candidate_is_scored_and_stops_with_recorded_cost():
    image = np.full((1080, 1920, 3), 126, dtype=np.uint8)
    image[444:636, 864:1056] = 225
    sensor = SceneSensor(build_pyramid(image))
    config = EpisodeConfig(k=1, query=0, t_max=3)
    episode = run_episode(
        sensor,
        EpisodeInput(query=0, width=1920, height=1080, expected_k=1),
        config,
        ClearCalibrator(),
        BrightCardNet(),
        device=torch.device("cpu"),
    )
    assert episode.reason == "threshold"
    assert episode.steps == 1
    assert np.linalg.norm(np.asarray(episode.answer_xy) - (960, 540)) <= 48
    assert episode.costs["sensing_flops"] > 0
    assert episode.costs["semantic_flops"] > 0
    assert len(episode.trace) == 1
    observation = episode.trace[0]["observations"][0]
    assert observation["candidate_id"] == episode.trace[0]["candidates"][0]["id"]
    assert 0 <= observation["valid_fraction"] <= 1
    assert np.isfinite(observation["raw_score"])
    assert np.isfinite(observation["normalized_score"])
    assert np.linalg.norm(np.asarray(observation["candidate_xy"]) - (960, 540)) <= 48


def test_evaluation_keeps_truth_outside_search_loop():
    truth = SceneTruth(np.asarray([[960, 540]]), (0,), (42,), 0)
    assert evaluate_answer((960, 540), truth).target_hit
    assert not evaluate_answer(None, truth).target_hit


def test_confirmation_prevents_stopping_before_a_discovered_card_is_scored():
    image = np.full((1080, 1920, 3), 126, dtype=np.uint8)
    image[444:636, 864:1056] = 225
    image[444:636, 1204:1396] = 225
    sensor = SceneSensor(build_pyramid(image))
    config = EpisodeConfig(k=2, query=0, t_max=4)
    episode = run_episode(
        sensor,
        EpisodeInput(query=0, width=1920, height=1080, expected_k=2),
        config, NarrowCalibrator(), BrightCardNet(),
        device=torch.device("cpu"), require_all_scored=True,
    )
    first = episode.trace[0]
    assert len(first["candidates"]) == 2
    assert len(first["observations"]) == 1
    assert episode.steps > 1
    assert np.linalg.norm(np.asarray(episode.trace[1]["fixation_xy"]) - (1300, 540)) < 48


def test_experiment_driver_dispatches_confirmation_method():
    image = np.full((1080, 1920, 3), 126, dtype=np.uint8)
    image[444:636, 864:1056] = 225
    result = run_job(
        "saccadenet_confirm_unscored", image, EpisodeConfig(k=1, query=0),
        BrightCardNet(), ClearCalibrator(), None,
        device=torch.device("cpu"), tile_outputs=48, two_stage_threshold=0.95,
    )
    assert result.reason == "threshold"
    assert result.trace[0]["unscored_count"] == 0
