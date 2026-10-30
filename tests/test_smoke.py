"""End-to-end run on a small synthetic catalogue."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spad_algorithm.algorithm import main, run


def _write_catalog(path: Path, include_aftershock_column: bool) -> None:
    generator = np.random.default_rng(0)

    def blob(latitude, longitude, depth, count):
        return np.column_stack([
            latitude + generator.normal(0.0, 0.01, count),
            longitude + generator.normal(0.0, 0.01, count),
            depth + generator.normal(0.0, 0.2, count),
        ])

    coordinates = np.vstack([
        blob(45.0, 150.0, 30.0, 15),
        blob(52.0, 160.0, 80.0, 15),
    ])
    count = len(coordinates)
    frame = pd.DataFrame({
        "time": ["1990-08-16 11:57:07.880000+00:00"] * count,
        "lat": coordinates[:, 0],
        "lon": coordinates[:, 1],
        "depth": coordinates[:, 2],
        "magnitude": np.full(count, 4.5),
    })
    if include_aftershock_column:
        frame["Aftershock"] = False
        aftershock = frame.iloc[[0]].copy()
        aftershock["Aftershock"] = True
        aftershock["lat"] = 10.0
        frame = pd.concat([frame, aftershock], ignore_index=True)
    frame.to_csv(path, index=False)


def _output_paths(directory: Path):
    return (
        directory / "dense.csv",
        directory / "clustered.csv",
        directory / "histogram.csv",
        directory / "histogram.png",
        directory / "dendrogram.png",
    )


def test_synthetic_catalog_writes_groups_and_patches(tmp_path):
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, include_aftershock_column=True)
    dense, clustered, histogram_csv, histogram_png, dendrogram_png = (
        _output_paths(tmp_path)
    )
    solution = run(
        str(catalog),
        str(dense),
        str(clustered),
        str(histogram_csv),
        str(histogram_png),
        str(dendrogram_png),
    )
    assert solution.n_events == 30
    assert solution.n_dense_groups >= 1
    assert solution.n_dense_events >= 2
    assert solution.n_patches == len(solution.patch_counts)
    assert solution.n_patches >= 1
    assert solution.radius_km > 0.0
    assert solution.threshold_km > 0.0
    assert solution.n_bins >= 1
    assert sum(solution.patch_counts) == solution.n_dense_events

    dense_frame = pd.read_csv(dense)
    clustered_frame = pd.read_csv(clustered)
    assert "dense_group_id" in dense_frame.columns
    assert "cluster" in clustered_frame.columns
    assert 10.0 not in set(dense_frame["lat"])
    kept = dense_frame["dense_group_id"] != -1
    assert len(clustered_frame) == int(kept.sum())
    assert histogram_csv.stat().st_size > 0
    assert histogram_png.read_bytes().startswith(b"\x89PNG")
    assert dendrogram_png.read_bytes().startswith(b"\x89PNG")

    again = run(
        str(catalog),
        str(tmp_path / "dense_again.csv"),
        str(tmp_path / "clustered_again.csv"),
        str(tmp_path / "histogram_again.csv"),
        str(tmp_path / "histogram_again.png"),
        str(tmp_path / "dendrogram_again.png"),
    )
    assert again == solution


def test_catalog_without_an_aftershock_column(tmp_path):
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, include_aftershock_column=False)
    dense, clustered, histogram_csv, histogram_png, dendrogram_png = (
        _output_paths(tmp_path)
    )
    solution = run(
        str(catalog),
        str(dense),
        str(clustered),
        str(histogram_csv),
        str(histogram_png),
        str(dendrogram_png),
    )
    assert solution.n_events == 30


def test_missing_magnitude_is_rejected(tmp_path):
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, include_aftershock_column=False)
    frame = pd.read_csv(catalog)
    frame.drop(columns=["magnitude"]).to_csv(catalog, index=False)
    with pytest.raises(ValueError, match="magnitude"):
        run(
            str(catalog),
            str(tmp_path / "dense.csv"),
            str(tmp_path / "clustered.csv"),
            str(tmp_path / "histogram.csv"),
            str(tmp_path / "histogram.png"),
            str(tmp_path / "dendrogram.png"),
        )


def test_command_line_accepts_the_six_paths(tmp_path):
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog, include_aftershock_column=True)
    dense, clustered, histogram_csv, histogram_png, dendrogram_png = (
        _output_paths(tmp_path)
    )
    main([
        "--input", str(catalog),
        "--dense-events", str(dense),
        "--clustered-results", str(clustered),
        "--histogram-csv", str(histogram_csv),
        "--histogram-png", str(histogram_png),
        "--dendrogram-png", str(dendrogram_png),
    ])
    assert clustered.exists()
