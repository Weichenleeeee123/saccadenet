import numpy as np

from saccadenet.bayes.s_min import render_digit_at_size


def test_smaller_digit_occupies_less_of_identical_card_patch():
    source = np.zeros((28, 28), dtype=np.uint8)
    source[6:22, 8:20] = 255
    large = render_digit_at_size(source, 48)
    small = render_digit_at_size(source, 8)
    assert large.shape == small.shape == (96, 96, 3)
    assert large.dtype == small.dtype == np.uint8
    assert (large < 150).sum() > (small < 150).sum() * 5
    assert np.all(large[0, 0] == 220)
