"""Fuzzy density, the level alpha, and the peak-similarity score."""

import numpy as np
import pytest

from spad_algorithm.algorithm import (
    analyze_peak_similarity,
    compute_density,
    find_dense_subsets,
    freedman_diaconis_bin_count,
    freedman_diaconis_bin_count_guarded,
    fuzzy_density_comparison,
)


def test_linear_kernel_excludes_the_point_itself():
    distance_matrix = np.array([
        [0.0, 10.0, 30.0],
        [10.0, 0.0, 40.0],
        [30.0, 40.0, 0.0],
    ])
    density = compute_density(distance_matrix, 20.0)
    assert density == pytest.approx([0.5, 0.5, 0.0])


def test_neighbour_on_the_radius_contributes_zero():
    distance_matrix = np.array([
        [0.0, 10.0],
        [10.0, 0.0],
    ])
    density = compute_density(distance_matrix, 10.0)
    assert density == pytest.approx([0.0, 0.0])


def test_alpha_is_one_minus_beta_times_the_maximum():
    densities = np.array([0.0, 2.0, 5.0])
    assert fuzzy_density_comparison(densities, 0.5) == pytest.approx(2.5)
    assert fuzzy_density_comparison(densities, 0.0) == pytest.approx(5.0)
    assert fuzzy_density_comparison(densities, -1.0) == pytest.approx(10.0)
    assert fuzzy_density_comparison(np.zeros(3), 0.2) == 0


def test_dense_subset_is_the_level_set():
    densities = np.array([1.0, 3.0, 2.0])
    selected = find_dense_subsets(densities, 2.0)
    assert np.array_equal(selected, np.array([1, 2]))


def test_identical_histograms_have_peak_score_zero():
    values = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 9.0, 9.0])
    score = analyze_peak_similarity(values, values.copy())
    assert score == pytest.approx(0.0)


def test_monotonic_sample_has_an_infinite_peak_score():
    values = np.arange(30, dtype=float)
    score = analyze_peak_similarity(values, values.copy())
    assert score == float("inf")


def test_guarded_bin_count_is_ten_when_the_width_is_zero():
    values = np.array([2.0, 2.0, 2.0, 2.0])
    assert freedman_diaconis_bin_count_guarded(values) == 10


def test_freedman_diaconis_count_matches_the_percentile_formula():
    values = np.arange(1, 101, dtype=float)
    quartile_1, quartile_3 = np.percentile(values, [25, 75])
    bin_width = 2 * (quartile_3 - quartile_1) / (len(values) ** (1 / 3))
    expected = int(np.ceil((max(values) - min(values)) / bin_width))
    assert freedman_diaconis_bin_count(values) == expected
