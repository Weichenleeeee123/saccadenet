"""Analytic detection horizon of the log-polar retina and exploration-grid coverage.

At eccentricity r the ring spacing is r*Δ (Δ = 2π/sectors) and the sampler reads pyramid
level L = round(log2(r*Δ)), whose pixels average 2^L x 2^L source pixels. A bright card of
side s is detected when some bilinear sample averages mostly card pixels:

* guaranteed for any alignment while the bilinear footprint fits inside the card, 3*2^L <= s;
* possible only for favorable grid alignment while one level pixel fits, 2^L <= s;
* impossible once 2^L > s (every level pixel mixes in background).

The switch from level L to L+1 happens at r = 2^(L+0.5)/Δ, which gives two radii.
"""

import math


def detection_horizon(card_size: float = 192, sectors: int = 128) -> tuple[float, float]:
    """Return (r_certain, r_possible) in canvas pixels."""
    delta = 2 * math.pi / sectors
    level_certain = math.floor(math.log2(card_size / 3))
    level_possible = math.floor(math.log2(card_size))
    return 2 ** (level_certain + 0.5) / delta, 2 ** (level_possible + 0.5) / delta


def coverage_radius(width: float, height: float, grid: int) -> float:
    """Largest distance from any canvas point to the nearest cell-center anchor of a grid x grid layout."""
    return math.hypot(width, height) / (2 * grid)


def derived_grid(width: float, height: float, horizon: float) -> int:
    """Smallest grid whose coverage radius does not exceed the horizon."""
    return max(1, math.ceil(math.hypot(width, height) / (2 * horizon)))
