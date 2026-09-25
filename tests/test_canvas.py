import numpy as np
import pytest

from saccadenet.config import EpisodeConfig
from saccadenet.data.canvas import CANVAS_VERSION, make_canvas
from saccadenet.data.splits import split_digit_ids


class FakeDigits:
    def sample(self, label, rng):
        digit_id = int(rng.integers(0, 1_000_000))
        image = np.full((28, 28), 255, dtype=np.uint8)
        image[4:24, 4:24] = 12 * label
        return image, digit_id


def test_canvas_is_reproducible_and_has_one_target():
    assert CANVAS_VERSION == "synthetic-v1"
    cfg = EpisodeConfig(k=12, query=7)
    first, first_truth = make_canvas(cfg, seed=412, digits=FakeDigits())
    second, second_truth = make_canvas(cfg, seed=412, digits=FakeDigits())
    assert first.shape == (1080, 1920, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert np.array_equal(first_truth.centers_xy, second_truth.centers_xy)
    assert first_truth.digits == second_truth.digits
    assert first_truth.digits.count(cfg.query) == 1
    assert first_truth.digits[first_truth.target_index] == cfg.query


def test_canvas_centers_obey_geometry():
    cfg = EpisodeConfig(k=12)
    _, truth = make_canvas(cfg, seed=190, digits=FakeDigits())
    xy = truth.centers_xy
    half = cfg.card_size / 2
    assert np.all((xy[:, 0] >= half) & (xy[:, 0] <= cfg.width - half))
    assert np.all((xy[:, 1] >= half) & (xy[:, 1] <= cfg.height - half))
    distance = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    assert distance.min() >= cfg.card_min_distance


def test_mnist_source_ids_are_disjoint_between_splits():
    splits = split_digit_ids()
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 50_000,
        "calibration": 5_000,
        "development": 5_000,
        "final_test": 10_000,
    }
    keys = ("train", "calibration", "development")
    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            assert set(splits[left]).isdisjoint(splits[right])
    # MNIST test is a separate source; its numeric indices may overlap train.
    assert splits["final_test"] == range(0, 10_000)


def test_canvas_rejects_impossible_packing_instead_of_shrinking_k():
    with pytest.raises(ValueError, match="placement_failed"):
        make_canvas(EpisodeConfig(k=1000), seed=1, digits=FakeDigits())
