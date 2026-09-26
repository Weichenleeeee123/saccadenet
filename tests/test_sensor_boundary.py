from dataclasses import fields
import inspect

import numpy as np

from saccadenet.contracts import RetinaOut
from saccadenet.retina.pyramid import build_pyramid
from saccadenet.retina.reconstruct import candidate_view
from saccadenet.retina.sampler import sample_retina


def test_retina_output_has_no_full_image_or_pyramid_reference():
    names = {field.name for field in fields(RetinaOut)}
    assert "image" not in names
    assert "pyramid" not in names
    assert tuple(inspect.signature(candidate_view).parameters) == ("retina", "xy", "size")


def test_reconstruction_does_not_re_read_mutated_original():
    image = np.full((256, 256, 3), 137, dtype=np.uint8)
    retina = sample_retina(build_pyramid(image), (128, 128))
    image.fill(0)
    view, valid = candidate_view(retina, (128, 128))
    assert valid.all()
    assert np.all(view == 137)
