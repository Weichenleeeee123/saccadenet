import math

from saccadenet.retina.horizon import coverage_radius, derived_grid, detection_horizon


def test_horizon_for_default_card_and_retina():
    certain, possible = detection_horizon(192, 128)
    delta = 2 * math.pi / 128
    assert math.isclose(certain, 2 ** 6.5 / delta)
    assert math.isclose(possible, 2 ** 7.5 / delta)
    assert 1840 < certain < 1850 and 3680 < possible < 3695


def test_frozen_five_by_five_grid_just_covers_16k():
    certain, _ = detection_horizon()
    assert coverage_radius(15360, 8640, 5) < certain
    assert coverage_radius(24576, 13824, 5) > certain
    assert derived_grid(15360, 8640, certain) == 5
    assert derived_grid(24576, 13824, certain) == 8
    assert derived_grid(1920, 1080, certain) == 1


def test_coverage_radius_is_corner_distance_to_first_anchor():
    width, height, grid = 1000, 600, 4
    corner = math.hypot(width / (2 * grid), height / (2 * grid))
    assert math.isclose(coverage_radius(width, height, grid), corner)
