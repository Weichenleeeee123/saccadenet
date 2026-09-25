import math

import numpy as np
import pytest

from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.sampler import ring_count, sample_retina
from saccadenet.retina.reconstruct import candidate_view


@pytest.mark.parametrize(
    ("shape", "rings", "samples"),
    [
        ((1080, 1920), 79, 19_328),
        ((2160, 3840), 94, 21_248),
        ((4320, 7680), 108, 23_040),
        ((8640, 15360), 122, 24_832),
    ],
)
def test_sampling_budget_matches_closed_form(shape, rings, samples):
    assert ring_count(shape, fovea_size=96, sectors=128) == rings
    assert 96 * 96 + rings * 128 == samples


def test_center_view_is_only_the_observed_fovea():
    rng = np.random.default_rng(21)
    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    retina = sample_retina(build_pyramid(image), (64, 64))
    view, mask = candidate_view(retina, (64, 64))
    assert view.shape == (96, 96, 3)
    assert np.array_equal(view, retina.fovea)
    assert mask.all()
    assert not hasattr(retina, "pyramid")
    assert not hasattr(retina, "image")


def test_outermost_ring_radius_covers_image_diagonal():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    retina = sample_retina(build_pyramid(image), (0, 0))
    outer_radius = np.linalg.norm(retina.sample_xy[-1], axis=-1).max()
    assert outer_radius >= math.hypot(*image.shape[:2])


def test_constant_image_survives_fovea_and_peripheral_sampling():
    image = np.full((256, 256, 3), 137, dtype=np.uint8)
    retina = sample_retina(build_pyramid(image), (128, 128))
    assert np.all(retina.fovea[retina.fovea_valid] == 137)
    assert np.allclose(retina.logpolar[retina.logpolar_valid], 137, atol=1)
    view, mask = candidate_view(retina, (200, 128))
    assert mask.any()
    assert np.allclose(view[mask], 137, atol=1)


def test_corner_fixation_masks_samples_outside_canvas():
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    retina = sample_retina(build_pyramid(image), (0, 0))
    assert retina.fovea_valid[48, 48]
    assert not retina.fovea_valid[0, 0]
    assert retina.logpolar_valid.any()
    assert not retina.logpolar_valid.all()


def test_far_checkerboard_is_averaged_before_sparse_sampling():
    checker = (np.indices((512, 512)).sum(axis=0) % 2 * 255).astype(np.uint8)
    image = np.repeat(checker[:, :, None], 3, axis=2)
    retina = sample_retina(build_pyramid(image), (256, 256))
    rings = np.linalg.norm(retina.sample_xy - np.asarray((256, 256)), axis=-1)
    values = retina.logpolar[(rings > 150) & retina.logpolar_valid]
    assert len(values) > 100
    assert abs(float(values.mean()) - 127.5) < 3.0
