import numpy as np
import torch

from saccadenet.models.train_fovea import FoveaPatchDataset


class FakeBank:
    def sample(self, label, rng):
        digit = np.zeros((28, 28), dtype=np.uint8)
        digit[5:23, 8:20] = 255
        return digit, int(rng.integers(0, 10000))


def test_digit_patch_is_deterministic_and_has_requested_shape():
    dataset = FoveaPatchDataset(FakeBank(), samples=100, seed=1, blank_fraction=0)
    image_a, label_a = dataset[17]
    image_b, label_b = dataset[17]
    assert image_a.shape == (3, 96, 96)
    assert image_a.dtype == torch.float32
    assert 0 <= label_a <= 9
    assert torch.equal(image_a, image_b)
    assert label_a == label_b


def test_empty_patch_uses_eleventh_class():
    dataset = FoveaPatchDataset(FakeBank(), samples=5, seed=0, blank_fraction=1)
    for image, label in dataset:
        assert image.shape == (3, 96, 96)
        assert label == 10


def test_size_augmentation_changes_digit_extent_without_changing_label():
    ordinary = FoveaPatchDataset(FakeBank(), samples=1, seed=5, blank_fraction=0, max_shift=0, degrade_probability=0)
    scaled = FoveaPatchDataset(
        FakeBank(), samples=1, seed=5, blank_fraction=0, max_shift=0,
        degrade_probability=0, scale_augment_probability=1, digit_sizes=(8,)
    )
    large, original_label = ordinary[0]
    small, scaled_label = scaled[0]
    assert original_label == scaled_label
    assert int((large[0] < 150 / 255).sum()) > int((small[0] < 150 / 255).sum()) * 5
