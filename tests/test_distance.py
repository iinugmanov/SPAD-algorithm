"""Haversine-plus-depth distances against values computed independently."""

import math

import numpy as np
import pytest

from spad_algorithm.algorithm import (
    compute_3d_distance,
    positive_pairwise_distances,
)


def _reference_distance(lat1, lon1, depth1, lat2, lon2, depth2):
    earth_radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    haversine_a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    haversine_a = min(max(haversine_a, 0.0), 1.0)
    horizontal = 2 * earth_radius_km * math.asin(math.sqrt(haversine_a))
    return math.sqrt(horizontal ** 2 + (depth1 - depth2) ** 2)


def test_depth_only_separation_is_the_depth_difference():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 12.5],
    ])
    distance = compute_3d_distance(coords)
    assert distance[0, 1] == pytest.approx(12.5)
    assert distance[1, 0] == pytest.approx(12.5)
    assert distance[0, 0] == pytest.approx(0.0)


def test_one_degree_of_longitude_at_the_equator():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    expected = _reference_distance(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    distance = compute_3d_distance(coords)
    assert distance[0, 1] == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(111.19492664455873, abs=1e-9)


def test_horizontal_and_depth_combine_in_quadrature():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 5.0],
    ])
    expected = _reference_distance(0.0, 0.0, 0.0, 0.0, 1.0, 5.0)
    horizontal = _reference_distance(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert expected == pytest.approx(math.sqrt(horizontal ** 2 + 25.0))
    distance = compute_3d_distance(coords)
    assert distance[0, 1] == pytest.approx(expected, abs=1e-9)
    assert distance[1, 0] == pytest.approx(distance[0, 1])


def test_asymmetric_matrix_is_averaged_before_the_upper_triangle(capsys):
    matrix = np.array([
        [0.0, 1.0],
        [3.0, 0.0],
    ])
    distances = positive_pairwise_distances(matrix)
    assert distances == pytest.approx([2.0])
    assert "not symmetric" in capsys.readouterr().out
    assert matrix[0, 1] == 1.0
