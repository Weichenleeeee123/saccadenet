from dataclasses import fields
import inspect

from saccadenet.contracts import EpisodeInput
from saccadenet.retina.detect import detect_on_retina
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.run.episode import run_episode


def test_level_two_entry_points_have_no_truth_or_full_image_argument():
    assert set(field.name for field in fields(EpisodeInput)) == {"query", "width", "height", "expected_k"}
    assert tuple(inspect.signature(detect_on_retina).parameters) == ("retina", "brightness_threshold")
    assert tuple(inspect.signature(candidate_view).parameters) == ("retina", "xy", "size")
    assert tuple(inspect.signature(run_episode).parameters) == (
        "sensor", "episode_input", "config", "calibrator", "network", "device",
        "require_all_scored", "strategy", "policy_seed"
    )
