import numpy as np

from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.retina.sampler import sample_retina
from saccadenet.viz.demo_player import _grid, retina_view


def _scene() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)


def test_retina_view_matches_candidate_view_on_unit_grid():
    retina = sample_retina(build_pyramid(_scene()), (400.0, 260.0))
    center = (610.0, 300.0)
    expected, expected_valid = candidate_view(retina, center, size=96)
    row = np.arange(96, dtype=np.float32) - 48
    gx, gy = np.meshgrid(row + center[0], row + center[1])
    view, valid = retina_view(retina, gx, gy)
    assert np.array_equal(valid, expected_valid)
    assert np.array_equal(view, expected)


def test_retina_view_never_reads_the_source_image_again():
    image = _scene()
    retina = sample_retina(build_pyramid(image), (480.0, 270.0))
    before, _ = retina_view(retina, *_grid(960, 540, 320, 180))
    image[:] = 0
    after, _ = retina_view(retina, *_grid(960, 540, 320, 180))
    assert np.array_equal(before, after)


def test_zoom_grid_uses_square_pixels_centered_on_fixation():
    gx, gy = _grid(960, 540, 480, 270, center=(100.0, 50.0), span=960)
    assert gx.shape == (270, 480)
    assert np.isclose(gx[0, 1] - gx[0, 0], 2.0) and np.isclose(gy[1, 0] - gy[0, 0], 2.0)
    assert np.isclose(gx.mean(), 100.0) and np.isclose(gy.mean(), 50.0)
