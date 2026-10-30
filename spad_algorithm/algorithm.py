"""Seismogenic patch detection by fuzzy density analysis.

A declustered catalogue of earthquake hypocentres is scanned over a
power-mean radius and a density level. The level set is clustered by
centroid linkage. Seismogenic patches are a second centroid-linkage cut,
taken at the modal pairwise distance.

Magnitude and time are required columns of the catalogue and are not
used in the distance, the density, or either clustering.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from numba import jit, prange
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

_REQUIRED_COLUMNS = ("lat", "lon", "depth", "magnitude", "time")

_PROMPTS = (
    (
        "Enter the path to the input declustered earthquake "
        "catalog file (CSV format): "
    ),
    (
        "Enter the path for saving the output CSV file of "
        "dense grouping events: "
    ),
    (
        "Enter the path for saving the output CSV file of "
        "clustered results: "
    ),
    "Enter the path for saving the histogram data CSV file: ",
    (
        "Enter the path for saving the output PNG file of "
        "pairwise histogram: "
    ),
    "Enter the path for saving the output PNG file for dendrogram: ",
)


@dataclass(frozen=True)
class Solution:
    """Numeric outcome of one catalogue.

    Attributes
    ----------
    q, beta : float
        Selected grid values, as produced by ``numpy.arange``.
    radius_km : float
        Power-mean radius at ``q``, in kilometres.
    alpha : float
        Density level ``(1 - beta) * max(density)``.
    n_events : int
        Rows kept after the optional aftershock filter.
    n_dense_events : int
        Events placed in a dense group.
    n_dense_groups : int
        Centroid-linkage clusters of the dense set at ``radius_km``.
    n_bins : int
        Freedman-Diaconis bin count of the pairwise distances.
    threshold_km : float
        Centre of the modal distance bin, in kilometres.
    n_patches : int
        Seismogenic patches cut at ``threshold_km``.
    patch_counts : tuple of int
        Event counts of patches 1, 2, ... in patch-id order.
    """

    q: float
    beta: float
    radius_km: float
    alpha: float
    n_events: int
    n_dense_events: int
    n_dense_groups: int
    n_bins: int
    threshold_km: float
    n_patches: int
    patch_counts: tuple[int, ...]


@jit(nopython=True)
def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Haversine distance from one point to many points.

    Parameters
    ----------
    lon1, lat1 : float
        Longitude and latitude of the reference point, in degrees.
    lon2, lat2 : ndarray
        Longitude and latitude of the other points, in degrees.

    Returns
    -------
    ndarray
        Great-circle distances in kilometres. The haversine argument
        is clipped to the closed interval from 0 to 1.
    """
    earth_radius_km = 6371.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sin_dlat = np.sin(dlat / 2.0)
    sin_dlon = np.sin(dlon / 2.0)
    haversine_a = (
        sin_dlat ** 2
        + np.cos(lat1) * np.cos(lat2) * sin_dlon ** 2
    )
    haversine_a = np.minimum(np.maximum(haversine_a, 0.0), 1.0)
    central_angle = 2 * np.arcsin(np.sqrt(haversine_a))
    return earth_radius_km * central_angle


@jit(nopython=True)
def compute_depth_differences(depth_array, depth_scalar):
    """Absolute depth difference, in kilometres.

    Parameters
    ----------
    depth_array : ndarray
        Depths in kilometres.
    depth_scalar : float
        Reference depth in kilometres.

    Returns
    -------
    ndarray
        Absolute differences, in kilometres.
    """
    return np.abs(depth_array - depth_scalar)


@jit(nopython=True, parallel=True)
def compute_3d_distance(coords):
    """Combine the haversine separation with the depth difference.

    Parameters
    ----------
    coords : ndarray of shape (n, 3)
        Columns are latitude in degrees, longitude in degrees, and
        depth in kilometres.

    Returns
    -------
    ndarray of shape (n, n)
        Distances in kilometres. The diagonal is zero.
    """
    num_events = len(coords)
    distance_matrix = np.zeros((num_events, num_events))
    for event_index in prange(num_events):
        longitude = coords[event_index, 1]
        latitude = coords[event_index, 0]
        depth = coords[event_index, 2]
        horizontal = haversine_vectorized(
            longitude,
            latitude,
            coords[:, 1],
            coords[:, 0],
        )
        depth_difference = compute_depth_differences(
            coords[:, 2],
            depth,
        )
        distance_matrix[event_index, :] = np.sqrt(
            horizontal ** 2 + depth_difference ** 2
        )
    return distance_matrix


@jit(nopython=True, parallel=True)
def compute_density(distance_matrix, radius):
    """Sum the linear kernel inside a radius.

    The point itself is not included. The sum is not normalised.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Pairwise distances in kilometres.
    radius : float
        Neighbourhood radius in kilometres.

    Returns
    -------
    ndarray of shape (n,)
        Un-normalised fuzzy density of each event.
    """
    num_events = distance_matrix.shape[0]
    densities = np.zeros(num_events)
    for event_index in prange(num_events):
        density = 0.0
        for other_index in range(num_events):
            distance_ij = distance_matrix[event_index, other_index]
            if event_index != other_index and distance_ij <= radius:
                density += 1.0 - distance_ij / radius
        densities[event_index] = density
    return densities


def exponent_grid() -> np.ndarray:
    """Return the exponent grid ``q``.

    Returns
    -------
    ndarray of shape (28,)
        ``numpy.arange(-2.9, -0.1, 0.1)``. The entries are not exact
        decimal tenths.
    """
    return np.arange(-2.9, -0.1, 0.1)


def level_grid() -> np.ndarray:
    """Return the level grid ``beta``.

    Returns
    -------
    ndarray of shape (20,)
        ``numpy.arange(-1.0, 1.0, 0.1)``. The value nearest zero is a
        small negative number, not zero, and 1 is excluded.
    """
    return np.arange(-1.0, 1.0, 0.1)


def load_catalog(path: str) -> pd.DataFrame:
    """Read a catalogue and drop rows marked as aftershocks.

    Parameters
    ----------
    path : str
        CSV path. An ``Aftershock`` column is optional.

    Returns
    -------
    DataFrame
        Rows whose aftershock flag is not true. Missing flags are kept.
    """
    data = pd.read_csv(path)
    if "Aftershock" in data.columns:
        data = data.loc[data["Aftershock"].ne(True)]
    return data


def require_columns(data: pd.DataFrame) -> None:
    """Raise if a column used by the original script is absent.

    Parameters
    ----------
    data : DataFrame
        Catalogue after the aftershock filter.

    Raises
    ------
    ValueError
        A required column is missing. ``magnitude`` and ``time`` are
        required even though their values are not read.
    """
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Catalog is missing required columns: {joined}")


def finite_coordinates(data: pd.DataFrame) -> np.ndarray:
    """Return latitude, longitude and depth, dropping non-finite rows.

    Rows with a NaN or an Inf in any of the three columns are removed
    from the returned array only. The frame is left unchanged, so those
    rows are no longer aligned with the matrix.

    Parameters
    ----------
    data : DataFrame
        Must contain ``lat``, ``lon`` and ``depth``.

    Returns
    -------
    ndarray of shape (m, 3)
        Finite coordinates, with ``m <= len(data)``.
    """
    coordinates = data[["lat", "lon", "depth"]].to_numpy()
    invalid = np.isnan(coordinates).any(axis=1)
    invalid = invalid | np.isinf(coordinates).any(axis=1)
    return coordinates[~invalid]


def positive_pairwise_distances(distance_matrix: np.ndarray) -> np.ndarray:
    """Return the positive upper triangle of a distance matrix.

    A matrix that is not symmetric at absolute tolerance ``1e-8``
    (NumPy's default relative tolerance still applies) is replaced by
    its symmetric part before the triangle is read. The caller's matrix
    is not modified.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Distances in kilometres.

    Returns
    -------
    ndarray of shape (n_pairs,)
        Distances ``d_ij`` with ``i < j`` and ``d_ij > 0``.
    """
    matrix = distance_matrix
    symmetric = np.allclose(matrix, matrix.T, atol=1e-8)
    if not symmetric:
        print("Distance matrix is not symmetric. Symmetrizing it.")
        matrix = (matrix + matrix.T) / 2
    distances = matrix[np.triu_indices_from(matrix, k=1)]
    return distances[distances > 0]


def freedman_diaconis_bin_count(values: np.ndarray) -> int:
    """Bin count for the pairwise-distance mode.

    A zero bin width is not replaced. The division then fails.

    Parameters
    ----------
    values : ndarray
        Sample whose interquartile range sets the bin width.

    Returns
    -------
    int
        ``ceil((max - min) / bin_width)``.
    """
    quartile_1, quartile_3 = np.percentile(values, [25, 75])
    bin_width = 2 * (quartile_3 - quartile_1) / (len(values) ** (1 / 3))
    span = max(values) - min(values)
    return int(np.ceil(span / bin_width))


def freedman_diaconis_bin_count_guarded(values: np.ndarray) -> int:
    """Bin count for the peak comparison.

    Parameters
    ----------
    values : ndarray
        Sample whose interquartile range sets the bin width.

    Returns
    -------
    int
        The Freedman-Diaconis count, or 10 when the width is zero.
    """
    quartile_1, quartile_3 = np.percentile(values, [25, 75])
    bin_width = 2 * (quartile_3 - quartile_1) / (len(values) ** (1 / 3))
    if bin_width == 0:
        return 10
    span = max(values) - min(values)
    return int(np.ceil(span / bin_width))


def modal_pairwise_distance(
    distances: np.ndarray,
) -> tuple[np.floating, int]:
    """Centre of the fullest Freedman-Diaconis bin.

    Parameters
    ----------
    distances : ndarray
        Positive pairwise distances, in kilometres.

    Returns
    -------
    mode : floating
        Bin centre, in kilometres. The lowest centre is kept when
        several bins share the maximum count.
    n_bins : int
        Number of bins.
    """
    num_bins = freedman_diaconis_bin_count(distances)
    hist_values, bin_edges = np.histogram(distances, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[np.argmax(hist_values)], num_bins


def power_mean_radius(distances: np.ndarray, exponent: float) -> np.floating:
    """Power mean of the positive pairwise distances.

    Parameters
    ----------
    distances : ndarray
        Positive distances in kilometres.
    exponent : float
        Power ``q``. Zero selects the geometric mean.

    Returns
    -------
    floating
        Radius in kilometres.
    """
    count = len(distances)
    if exponent != 0:
        mean_power = np.sum(distances ** exponent) / count
        return mean_power ** (1 / exponent)
    log_mean = np.sum(np.log(distances)) / count
    return np.exp(log_mean)


def fuzzy_density_comparison(densities: np.ndarray, beta: float):
    """Density level ``(1 - beta)`` times the maximum density.

    Parameters
    ----------
    densities : ndarray
        Fuzzy density of each event.
    beta : float
        Level parameter. Values below zero put the level above the
        maximum whenever that maximum is positive.

    Returns
    -------
    float
        The level. The integer 0 when the maximum density is 0.
    """
    max_density = np.max(densities)
    if max_density == 0:
        return 0
    return max_density - beta * max_density


def find_dense_subsets(densities: np.ndarray, alpha: float) -> np.ndarray:
    """Indices whose density is at least ``alpha``.

    Parameters
    ----------
    densities : ndarray
        Fuzzy density of each event.
    alpha : float
        Density level.

    Returns
    -------
    ndarray of int
        Increasing indices into ``densities``.
    """
    return np.where(densities >= alpha)[0]


def analyze_peak_similarity(all_densities, dense_densities):
    """Mean bin-index gap between density-histogram peaks.

    Both histograms share the Freedman-Diaconis edges of the combined
    sample. Peaks are strict local maxima. The score is infinite when
    either histogram has no peak.

    Parameters
    ----------
    all_densities : ndarray
        Density of every event.
    dense_densities : ndarray
        Density of the events in the level set.

    Returns
    -------
    float
        Mean, over peaks of the full histogram, of the distance in bin
        index to the nearest peak of the dense histogram.
    """
    combined = np.concatenate((all_densities, dense_densities))
    num_bins = freedman_diaconis_bin_count_guarded(combined)
    bins = np.histogram_bin_edges(combined, bins=num_bins)
    hist_all, _ = np.histogram(all_densities, bins=bins)
    hist_dense, _ = np.histogram(dense_densities, bins=bins)
    peaks_all, _ = find_peaks(hist_all)
    peaks_dense, _ = find_peaks(hist_dense)
    if len(peaks_all) == 0 or len(peaks_dense) == 0:
        return float("inf")
    peak_differences = [
        np.min(np.abs(peaks_dense - peak_all)) for peak_all in peaks_all
    ]
    return np.mean(peak_differences)


def select_dense_parameters(distance_matrix, all_distances):
    """Choose ``q`` and ``beta`` by peak similarity, then cluster count.

    The density depends only on ``r(q)``, so it is evaluated once for
    each exponent. The comparison itself is unchanged: the smallest
    peak score wins, and an equal score keeps the larger number of
    centroid-linkage clusters at distance ``r(q)``.

    Parameters
    ----------
    distance_matrix : ndarray of shape (n, n)
        Distances in kilometres.
    all_distances : ndarray
        Positive pairwise distances used for the power mean.

    Returns
    -------
    best_params : dict or None
        ``q``, ``beta``, ``r``, ``alpha``, ``clusters`` and
        ``dense_indices`` of the selected pair. ``None`` when every
        pair has fewer than two dense events.
    best_clust_num : int
        Number of clusters in ``best_params``, or 0 when there is none.
    """
    best_params = None
    best_clust_num = 0
    min_peak_diff = float("inf")
    for exponent in exponent_grid():
        radius = power_mean_radius(all_distances, exponent)
        densities = compute_density(distance_matrix, radius)
        for beta in level_grid():
            alpha = fuzzy_density_comparison(densities, beta)
            dense_indices = find_dense_subsets(densities, alpha)
            if len(dense_indices) < 2:
                continue
            peak_diff = analyze_peak_similarity(
                densities,
                densities[dense_indices],
            )
            subset = distance_matrix[np.ix_(dense_indices, dense_indices)]
            subset = (subset + subset.T) / 2
            condensed = squareform(subset)
            linkage_matrix = linkage(condensed, method="centroid")
            clusters = fcluster(
                linkage_matrix,
                t=radius,
                criterion="distance",
            )
            num_clusters = len(np.unique(clusters))
            improved = peak_diff < min_peak_diff
            tied = (
                peak_diff == min_peak_diff
                and num_clusters > best_clust_num
            )
            if improved or tied:
                best_params = {
                    "q": exponent,
                    "beta": beta,
                    "r": radius,
                    "alpha": alpha,
                    "clusters": clusters,
                    "dense_indices": dense_indices,
                }
                min_peak_diff = peak_diff
                best_clust_num = num_clusters
    return best_params, best_clust_num


def assign_dense_groups(data: pd.DataFrame, best_params: dict) -> pd.DataFrame:
    """Write ``dense_group_id`` using matrix-row indices.

    ``dense_indices`` index the coordinate array. They are applied to
    ``data`` by position. The two agree when every coordinate row was
    finite.

    Parameters
    ----------
    data : DataFrame
        Catalogue after the aftershock filter.
    best_params : dict
        Selected clusters and their matrix-row indices.

    Returns
    -------
    DataFrame
        The same object, with ``dense_group_id``. Events outside the
        level set are ``-1``. Kept groups are numbered from 0.
    """
    renamed = LabelEncoder().fit_transform(best_params["clusters"])
    cluster_labels = np.full(len(data), -1)
    for index, cluster_label in zip(best_params["dense_indices"], renamed):
        cluster_labels[index] = cluster_label
    data["dense_group_id"] = cluster_labels
    return data


def cluster_patches(data: pd.DataFrame, threshold: float):
    """Cut seismogenic patches at the modal pairwise distance.

    Parameters
    ----------
    data : DataFrame
        Catalogue that already has ``dense_group_id``.
    threshold : float
        Cut distance in kilometres.

    Returns
    -------
    data_dense : DataFrame
        Dense events only, with a ``cluster`` column. Identifiers are
        the 1-based labels returned by ``fcluster``.
    linkage_matrix : ndarray
        Centroid linkage of those events.
    distance_matrix : ndarray
        Distances used for that linkage, in kilometres. The upper
        triangle is taken without symmetrising.
    """
    data_dense = data[data["dense_group_id"] != -1].copy()
    coordinates = finite_coordinates(data_dense)
    distance_matrix = compute_3d_distance(coordinates)
    condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed, method="centroid")
    data_dense["cluster"] = fcluster(
        linkage_matrix,
        t=threshold,
        criterion="distance",
    )
    return data_dense, linkage_matrix, distance_matrix


def plot_distance_histogram(
    distance_matrix,
    distance_matrix_dense,
    output_histogram_csv,
    output_histogram_png,
):
    """Save the 40-bin pairwise-distance histogram and its CSV.

    Parameters
    ----------
    distance_matrix, distance_matrix_dense : ndarray
        Full-catalogue and dense-event distances, in kilometres.
    output_histogram_csv, output_histogram_png : str
        Destination paths. The figure is a PNG.

    Returns
    -------
    distances, distances_dense : ndarray
        Positive upper-triangle samples used in the figure.
    """
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    dense = distance_matrix_dense[
        np.triu_indices_from(distance_matrix_dense, k=1)
    ]
    distances = distances[distances > 0]
    dense = dense[dense > 0]

    plt.figure(figsize=(9, 6), dpi=600)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 24,
    })

    hist_all, edges_all = np.histogram(distances, bins=40, density=True)
    hist_dense, edges_dense = np.histogram(dense, bins=40, density=True)
    centers_all = (edges_all[:-1] + edges_all[1:]) / 2
    centers_dense = (edges_dense[:-1] + edges_dense[1:]) / 2
    hist_data = pd.DataFrame({
        "Distance_all_km": centers_all,
        "Frequency_all": hist_all,
        "Distance_dense_km": centers_dense,
        "Frequency_dense": hist_dense,
    })
    hist_data.to_csv(output_histogram_csv, index=False)
    print(f"Histogram data saved to {output_histogram_csv}")

    plt.plot(centers_all, hist_all, marker="o", color="black")
    plt.plot(centers_dense, hist_dense, marker="*", color="red")
    plt.xlabel("Distances, km", fontsize=24)
    plt.ylabel("Relative Frequency", fontsize=24)
    axes = plt.gca()
    for tick in axes.get_yticklabels():
        tick.set_fontsize(20)
        tick.set_fontname("Times New Roman")
        tick.set_position((
            tick.get_position()[0] - 0.01,
            tick.get_position()[1],
        ))
    for tick in axes.get_xticklabels():
        tick.set_fontsize(20)
        tick.set_fontname("Times New Roman")
        tick.set_position((
            tick.get_position()[0] - 0.01,
            tick.get_position()[1],
        ))
    plt.savefig(
        output_histogram_png,
        format="png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    return distances, dense


def plot_dendrogram(linkage_matrix, threshold, output_path) -> None:
    """Save the patch dendrogram as a PNG.

    Parameters
    ----------
    linkage_matrix : ndarray
        Centroid linkage of the dense events.
    threshold : float
        Horizontal cut, in kilometres.
    output_path : str
        PNG path.
    """
    plt.figure(figsize=(9, 6), facecolor="#D0CFD4")
    dendrogram(
        linkage_matrix,
        color_threshold=threshold,
        p=5,
        truncate_mode="level",
    )
    plt.axhline(
        y=threshold,
        color="r",
        linestyle="--",
        label="Specific distance",
    )
    plt.xlabel("Set index", fontsize=24, fontname="Times New Roman")
    plt.ylabel("Distance, km", fontsize=24, fontname="Times New Roman")
    plt.xticks(fontsize=12, fontname="Times New Roman")
    plt.yticks(fontsize=20, fontname="Times New Roman")
    plt.legend(fontsize=22)
    plt.savefig(output_path, format="png", dpi=600, bbox_inches="tight")
    plt.close()


def run(
    input_path: str,
    dense_events_path: str,
    clustered_results_path: str,
    histogram_csv_path: str,
    histogram_png_path: str,
    dendrogram_png_path: str,
) -> Solution:
    """Run the density scan and write the six products of one catalogue.

    Parameters
    ----------
    input_path : str
        Declustered catalogue in CSV form.
    dense_events_path : str
        Catalogue with ``dense_group_id`` appended.
    clustered_results_path : str
        Dense events with a ``cluster`` column.
    histogram_csv_path, histogram_png_path : str
        Pairwise-distance histogram.
    dendrogram_png_path : str
        Dendrogram of the patch linkage.

    Returns
    -------
    Solution
        Selected parameters and the patch sizes.

    Raises
    ------
    ValueError
        The catalogue misses a required column, or no grid pair has a
        dense set of at least two events.
    """
    data = load_catalog(input_path)
    require_columns(data)
    coordinates = finite_coordinates(data)
    distance_matrix = compute_3d_distance(coordinates)
    all_distances = positive_pairwise_distances(distance_matrix)
    threshold, num_bins = modal_pairwise_distance(all_distances)
    print(f"Suggested number of bins: {num_bins}")

    best_params, best_clust_num = select_dense_parameters(
        distance_matrix,
        all_distances,
    )
    if best_params is None:
        print("No suitable parameters found for clustering.")
        raise ValueError("No suitable parameters found for clustering.")

    print(
        "Best parameters: "
        f"q = {best_params['q']}, beta = {best_params['beta']}, "
        f"r = {best_params['r']}, alpha = {best_params['alpha']}"
    )
    print(f"Number of dense group: {best_clust_num}")
    assign_dense_groups(data, best_params)
    data.to_csv(dense_events_path, index=False)

    data_dense, linkage_matrix, distance_matrix_dense = cluster_patches(
        data,
        threshold,
    )
    print(
        "Specific distance of nontrivial pairwise distance "
        f"distribution: {threshold} km"
    )
    data_dense.to_csv(clustered_results_path, index=False)

    plot_distance_histogram(
        distance_matrix,
        distance_matrix_dense,
        histogram_csv_path,
        histogram_png_path,
    )
    plot_dendrogram(linkage_matrix, threshold, dendrogram_png_path)

    patch_ids, patch_sizes = np.unique(
        data_dense["cluster"].to_numpy(),
        return_counts=True,
    )
    ordered = patch_sizes[np.argsort(patch_ids)]
    return Solution(
        q=float(best_params["q"]),
        beta=float(best_params["beta"]),
        radius_km=float(best_params["r"]),
        alpha=float(best_params["alpha"]),
        n_events=int(len(data)),
        n_dense_events=int(len(data_dense)),
        n_dense_groups=int(best_clust_num),
        n_bins=int(num_bins),
        threshold_km=float(threshold),
        n_patches=int(len(patch_ids)),
        patch_counts=tuple(int(size) for size in ordered),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the six-path command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Detect seismogenic patches by fuzzy density analysis "
            "of earthquake hypocentres."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Declustered earthquake catalogue (CSV).",
    )
    parser.add_argument(
        "--dense-events",
        required=True,
        help="Output CSV of events with dense_group_id.",
    )
    parser.add_argument(
        "--clustered-results",
        required=True,
        help="Output CSV of dense events with patch ids.",
    )
    parser.add_argument(
        "--histogram-csv",
        required=True,
        help="Output CSV of the pairwise-distance histogram.",
    )
    parser.add_argument(
        "--histogram-png",
        required=True,
        help="Output PNG of the pairwise-distance histogram.",
    )
    parser.add_argument(
        "--dendrogram-png",
        required=True,
        help="Output PNG of the patch dendrogram.",
    )
    return parser


def _paths_from_prompts() -> tuple[str, str, str, str, str, str]:
    """Read the six paths from the terminal, matching the script prompts."""
    answers = []
    for prompt in _PROMPTS:
        answers.append(input(prompt).strip())
    return tuple(answers)


def main(argv: list[str] | None = None) -> None:
    """Run from the command line, or prompt when no arguments are given.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name. The default is ``sys.argv``.
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        paths = _paths_from_prompts()
    else:
        args = _parser().parse_args(argv)
        paths = (
            args.input,
            args.dense_events,
            args.clustered_results,
            args.histogram_csv,
            args.histogram_png,
            args.dendrogram_png,
        )
    run(*paths)
