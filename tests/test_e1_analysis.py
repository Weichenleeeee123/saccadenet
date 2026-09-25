import numpy as np
import pytest

from saccadenet.exp.analyze_e1 import paired_bootstrap


def test_paired_bootstrap_preserves_pairs_and_ratio_definition():
    low = np.array([1, 2, 3, 4], dtype=float)
    high = low * 2
    ratio = paired_bootstrap(low, high, metric="ratio", replicates=100)
    assert ratio["n_pairs"] == 4
    assert ratio["estimate"] == 2
    assert ratio["low_95"] == ratio["high_95"] == 2
    difference = paired_bootstrap(np.array([0, 1, 0, 1]), np.array([1, 1, 1, 1]), metric="difference", replicates=100)
    assert difference["estimate"] == 0.5
    with pytest.raises(ValueError, match="denominator"):
        paired_bootstrap(np.zeros(4), np.ones(4), metric="ratio", replicates=10)
