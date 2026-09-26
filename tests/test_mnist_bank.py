import numpy as np
import pytest

from saccadenet.data.mnist_bank import MnistBank


class TinyDataset:
    def __init__(self):
        self.targets = np.asarray([1, 2, 1, 3, 2, 1])
        self.data = np.arange(6 * 28 * 28, dtype=np.int64).reshape(6, 28, 28).astype(np.uint8)


def test_sampling_only_uses_allowed_source_indices_and_requested_label():
    bank = MnistBank(TinyDataset(), allowed_indices=(1, 2, 3, 4, 5))
    rng = np.random.default_rng(42)
    for _ in range(20):
        image, source_id = bank.sample(1, rng)
        assert source_id in {2, 5}
        assert image.shape == (28, 28)
        assert np.array_equal(image, bank.dataset.data[source_id])


def test_missing_class_is_an_error_not_fallback_to_another_label():
    bank = MnistBank(TinyDataset(), allowed_indices=(0, 2, 5))
    with pytest.raises(ValueError, match="label 9"):
        bank.sample(9, np.random.default_rng(0))
