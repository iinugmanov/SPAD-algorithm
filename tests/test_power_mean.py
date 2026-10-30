"""Power-mean radius and the exponent and level grids."""

import numpy as np
import pytest

from spad_algorithm.algorithm import (
    exponent_grid,
    level_grid,
    modal_pairwise_distance,
    power_mean_radius,
)


def test_harmonic_mean_at_exponent_minus_one():
    distances = np.array([1.0, 4.0, 4.0])
    assert power_mean_radius(distances, -1) == pytest.approx(2.0)


def test_geometric_mean_at_exponent_zero():
    distances = np.array([1.0, 4.0])
    assert power_mean_radius(distances, 0) == pytest.approx(2.0)


def test_power_mean_matches_the_defining_sum():
    distances = np.array([0.5, 1.0, 2.0, 8.0, 20.0])
    exponents = list(exponent_grid()[::5]) + [0.0, 1.0]
    for exponent in exponents:
        got = power_mean_radius(distances, exponent)
        if exponent != 0:
            expected = (
                np.sum(distances ** exponent) / len(distances)
            ) ** (1 / exponent)
        else:
            expected = np.exp(np.sum(np.log(distances)) / len(distances))
        assert got == expected


def test_grids_are_the_arange_sequences():
    exponents = exponent_grid()
    levels = level_grid()
    assert len(exponents) == 28
    assert len(levels) == 20
    assert exponents[0] == np.float64(-2.9)
    assert exponents[-1] < -0.1
    assert levels[0] == np.float64(-1.0)
    assert levels[-1] < 1.0
    assert not np.any(levels == 0.0)


def test_modal_distance_is_the_centre_of_the_tallest_bin():
    generator = np.random.default_rng(1)
    distances = np.concatenate([
        generator.normal(5.0, 0.15, 500),
        generator.normal(30.0, 0.15, 40),
    ])
    mode, num_bins = modal_pairwise_distance(distances)
    hist_values, edges = np.histogram(distances, bins=num_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    assert mode == centers[np.argmax(hist_values)]
    assert mode == pytest.approx(5.0, abs=0.5)
