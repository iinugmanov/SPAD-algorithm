# SPAD-algorithm

SPAD (Seismogenic Patches Detection) selects a radius and a density
level for a declustered earthquake catalogue, clusters the events above
that level, and cuts the dense events into seismogenic patches.

The radius is a power mean of the pairwise hypocentral distances. The
density is a linear kernel. Dense groups and patches are both
centroid-linkage cuts. Declustering, and the alpha-filtration used in
the companion implementation, are described in
[iinugmanov/SPAD](https://github.com/iinugmanov/SPAD). The formulas
used here are in [`docs/algorithm.md`](docs/algorithm.md).

## Installation

The program is written for Python 3.11.7. The pins in
[`requirements.txt`](requirements.txt) install on that version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from the repository root. The first call compiles
the Numba distance kernels and is slower than later calls.

## Input

CSV, one event per row.

| Column | Type | Role |
| --- | --- | --- |
| `lat` | float, degrees | Latitude |
| `lon` | float, degrees | Longitude |
| `depth` | float, km | Depth, positive downward |
| `magnitude` | float | Required. Not read by the calculation |
| `time` | text | Required. Not parsed and not read by the calculation |
| `Aftershock` | boolean | Optional. Rows equal to true are removed |

```csv
time,lat,lon,depth,magnitude,Aftershock
1990-08-16 11:57:07.880000+00:00,45.727,149.860,94.7,4.9,False
```

Other columns are copied through to the output tables.

## Usage

```bash
python -m spad_algorithm \
    --input catalog.csv \
    --dense-events results/dense_events.csv \
    --clustered-results results/clustered_results.csv \
    --histogram-csv results/pairwise_histogram.csv \
    --histogram-png results/pairwise_histogram.png \
    --dendrogram-png results/dendrogram.png
```

With no arguments the program asks for the same six paths on the
terminal.

## Outputs

`dense_events.csv` is the catalogue after the aftershock filter, with
`dense_group_id` appended. Events outside the selected level set are
`-1`. Groups that were kept are numbered from 0.

`clustered_results.csv` is the rows with a non-negative
`dense_group_id`, with `cluster` appended. Patch identifiers are the
1-based labels of the second linkage cut.

`pairwise_histogram.csv` holds the 40-bin density histograms of the
positive pairwise distances:

| Column | Meaning |
| --- | --- |
| `Distance_all_km` | Bin centre, full catalogue, km |
| `Frequency_all` | Density-normalised count |
| `Distance_dense_km` | Bin centre, dense events, km |
| `Frequency_dense` | Density-normalised count |

The two PNG files are the histogram and the dendrogram of the patch
linkage. Both are written at 600 dpi.

## Example

On `catalog_bacground_proj.csv` (Kuril–Kamchatka, 4131 background
events, 16 August 1990 to 17 August 2026) the run reported below
selected:

| Quantity | Value |
| --- | --- |
| `q` | -2.6999999999999997 |
| `beta` | 0.8999999999999995 |
| `r` | 68.50470798967017 km |
| `alpha` | 14.291079899024027 |
| Dense groups | 48 |
| Modal distance | 181.4697367713092 km |
| Patches | 8 |

The catalogue file is not part of this repository. The full table, the
figures, and the command that reproduces them are in
[`docs/results.md`](docs/results.md).

## Tests

```bash
python -m pytest
python -m flake8 spad_algorithm tests
python -m pycodestyle spad_algorithm tests
```

The pytest suite covers the distance, the density, the power mean, and
a synthetic catalogue. It does not read `catalog_bacground_proj.csv`.
