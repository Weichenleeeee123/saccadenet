"""MNIST digit sampling with explicit source-index partitions."""

import numpy as np
from torchvision.datasets import MNIST

from saccadenet.data.splits import split_digit_ids


class MnistBank:
    def __init__(self, dataset, allowed_indices):
        self.dataset = dataset
        labels = np.asarray(dataset.targets)
        indices = np.asarray(tuple(allowed_indices), dtype=np.int64)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(labels)):
            raise ValueError("allowed_indices must refer to existing source images")
        self._by_label = {label: indices[labels[indices] == label] for label in range(10)}

    @classmethod
    def from_split(cls, split: str, root: str = "data/mnist", *, download: bool = False) -> "MnistBank":
        splits = split_digit_ids()
        if split not in splits:
            raise ValueError(f"unknown split: {split}")
        source = MNIST(root=root, train=split != "final_test", download=download)
        return cls(source, splits[split])

    def sample(self, label: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        if label not in self._by_label or len(self._by_label[label]) == 0:
            raise ValueError(f"label {label} has no source images in this split")
        source_id = int(rng.choice(self._by_label[label]))
        image = np.asarray(self.dataset.data[source_id], dtype=np.uint8)
        return image, source_id
