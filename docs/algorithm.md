# SPAD algorithm

This program takes a declustered earthquake catalogue and returns
seismogenic patches. A power mean of the pairwise distances sets a
radius \(r(q)\). A linear kernel gives a fuzzy density \(\rho\). One
pair \((q, \beta)\) is kept by comparing density histograms, and the
events with \(\rho \ge \alpha(\beta)\) are clustered by centroid
linkage at distance \(r(q)\). Those dense events are clustered again
at the modal pairwise distance. That second cut is the set of patches.

Declustering is an input, not a step of this program. The companion
repository [SPAD](https://github.com/iinugmanov/SPAD) builds an
alpha-filtration of a normalised discrete-perfect-set density. Section
[Relation to that filtration](#relation-to-the-spad-filtration) records
the differences. Locations of strong earthquakes are not an input.

## Symbols

| Symbol | Meaning | Unit |
| --- | --- | --- |
| \(i, j\) | Event indices | — |
| \(n\) | Event count of the matrix | — |
| \(\varphi\) | Latitude | degrees |
| \(\lambda\) | Longitude | degrees |
| \(z\) | Catalogue depth, positive downward | km |
| \(R\) | Earth radius, \(6371\) | km |
| \(h_{ij}\) | Haversine distance | km |
| \(d_{ij}\) | Hypocentral distance | km |
| \(\mathcal{D}\) | Positive pairwise distances | km |
| \(q\) | Power-mean exponent | dimensionless |
| \(r(q)\) | Radius | km |
| \(\rho_i\) | Fuzzy density | dimensionless |
| \(\beta\) | Level parameter | dimensionless |
| \(\alpha\) | Density threshold | dimensionless |
| \(S\) | Index set of the level set | — |
| \(s\) | Peak score | bin index |
| \(\tau\) | Modal distance, the patch threshold | km |

## 1. Catalogue

The table is a CSV file. The calculation reads `lat`, `lon` and
`depth`. `magnitude` and `time` must be present and their values are
ignored. `time` is not parsed. An `Aftershock` column, when present,
drops every row whose value equals true. A missing flag is kept.

Rows with a NaN or an Inf in `lat`, `lon` or `depth` are removed from
the coordinate array that enters the distance matrix. They stay in the
table. On a catalogue whose three coordinates are finite, matrix row
\(i\) is table row \(i\) after the aftershock filter.

## 2. Distance

Event \(i\) has latitude \(\varphi_i\), longitude \(\lambda_i\) and
depth \(z_i\). Degree differences are converted to radians,

\[
\Delta\varphi_{ij}
=
(\varphi_j - \varphi_i)\frac{\pi}{180},
\qquad
\Delta\lambda_{ij}
=
(\lambda_j - \lambda_i)\frac{\pi}{180}.
\]

The haversine argument, with latitudes in radians, is

\[
a_{ij}
=
\sin^2\!\left(\frac{\Delta\varphi_{ij}}{2}\right)
+
\cos\varphi_i\cos\varphi_j
\sin^2\!\left(\frac{\Delta\lambda_{ij}}{2}\right).
\]

\(a_{ij}\) is clipped to \([0, 1]\). The great-circle distance and the
hypocentral distance are

\[
h_{ij} = 2 R \arcsin\sqrt{a_{ij}},
\qquad
d_{ij} = \sqrt{h_{ij}^{2} + (z_i - z_j)^{2}},
\]

with \(R = 6371\,\mathrm{km}\). Both \(h_{ij}\) and \(d_{ij}\) are in
kilometres. The diagonal \(d_{ii}\) is 0.

The code stores coordinates as \((\varphi, \lambda, z)\) and passes
longitude before latitude to the haversine.

If \(d_{ij}\) and \(d_{ji}\) disagree by more than the tolerance of
`numpy.allclose` at absolute tolerance \(10^{-8}\) (the default
relative tolerance is left in place), the sample below uses
\((d_{ij} + d_{ji}) / 2\). The matrix passed to the density is the
matrix that was computed.

The positive upper triangle is

\[
\mathcal{D}
=
\{\, d_{ij} : i < j,\ d_{ij} > 0 \,\}.
\]

A repeated hypocentre contributes a zero to the matrix and contributes
nothing to \(\mathcal{D}\).

## 3. Radius

For \(q \ne 0\),

\[
r(q)
=
\left(
\frac{1}{|\mathcal{D}|}
\sum_{d \in \mathcal{D}} d^{\,q}
\right)^{1/q}.
\]

For \(q = 0\), which the scan never requests,

\[
r(0)
=
\exp\!\left(
\frac{1}{|\mathcal{D}|}
\sum_{d \in \mathcal{D}} \ln d
\right).
\]

\(r(q)\) is in kilometres. A negative \(q\) weights the short distances.

## 4. Fuzzy density and the level

Inside the radius the kernel is linear. The density of event \(i\) is

\[
\rho_i(r)
=
\sum_{\substack{j \ne i \\ d_{ij} \le r}}
\left(1 - \frac{d_{ij}}{r}\right).
\]

Each term lies in \([0, 1]\). The event itself is omitted, so a point
with no neighbour has \(\rho_i = 0\). The sum is not divided by a
constant. \(\rho_i\) is dimensionless.

The level at a parameter \(\beta\) is

\[
\alpha(\beta) = (1 - \beta)\,\max_i \rho_i(r).
\]

When the maximum density is 0, the program sets \(\alpha = 0\). The
dense index set is

\[
S(\beta) = \{\, i : \rho_i(r) \ge \alpha(\beta) \,\}.
\]

A pair with \(|S| < 2\) is skipped. For \(\beta < 0\) and a positive
maximum density, \(\alpha\) lies strictly above every \(\rho_i\), so
\(S\) is empty.

## 5. Parameter grid

The scan is the NumPy sequences

```text
q    = numpy.arange(-2.9, -0.1, 0.1)    # 28 values
beta = numpy.arange(-1.0,  1.0, 0.1)    # 20 values
```

The nominal steps are tenths of a unit. The value 0.1 has no finite
binary expansion, so the stored \(q\) and \(\beta\) are not the
decimal tenths, and the \(\beta\) entry nearest zero is a small
negative number rather than 0. Those stored values are the ones raised
to the power in \(r(q)\) and multiplied into \(\alpha\). The upper
endpoints \(-0.1\) and \(1\) are excluded.

## 6. Peak similarity

For a candidate pair, let \(\rho\) be the vector of all densities at
that \(r(q)\), and let \(\rho_S\) be the same vector restricted to
\(S\). The combined sample \(\rho \cup \rho_S\) (a concatenation, so a
dense event is counted twice) sets a Freedman–Diaconis bin width. With
\(n\) the sample size, \(Q_1\) and \(Q_3\) the 25% and 75% percentiles
from `numpy.percentile` (linear interpolation), and
\(\mathrm{IQR} = Q_3 - Q_1\),

\[
w = 2\,\mathrm{IQR}\, n^{-1/3},
\qquad
n_{\mathrm{bins}}
=
\left\lceil \frac{\max x - \min x}{w} \right\rceil.
\]

A zero width is replaced by 10 bins in this comparison. Both histograms
use the edges from `numpy.histogram_bin_edges` on the combined sample.
A peak is a strict local maximum, the default of
`scipy.signal.find_peaks`. Bins at the ends of the histogram are not
peaks.

Let \(U\) and \(V\) be the sets of peak bin indices. When either set is
empty the score is \(+\infty\). Otherwise

\[
s
=
\frac{1}{|U|}
\sum_{u \in U}
\min_{v \in V} |v - u|.
\]

\(s\) counts bins, not kilometres and not density.

## 7. Dense groups

The submatrix of \(S\) is replaced by its symmetric part, condensed,
and linked with centroid linkage (UPGMC, `method="centroid"`).
`fcluster` cuts that tree at distance \(r(q)\), in kilometres. The
number of clusters is the number of distinct labels.

The search keeps the pair with the smallest \(s\). When \(s\) is equal
for two pairs, including the case \(+\infty = +\infty\), it keeps the
pair with more clusters. The first pair is kept when the cluster counts
are equal as well.

`sklearn.preprocessing.LabelEncoder` then renumbers the kept labels to
\(0, 1, \ldots\) in sorted order. Those integers are written to
`dense_group_id` at the matrix-row positions in \(S\). Every other row
of the filtered table receives \(-1\).

The count reported as the number of dense groups is the number of
clusters of this cut. It is not the number of events in \(S\).

## 8. Seismogenic patches

The threshold \(\tau\) is computed once, from \(\mathcal{D}\), before
the scan. The bin count is the Freedman–Diaconis expression above with
no replacement when the width is zero. \(\tau\) is the centre of the
bin with the largest count,

\[
\tau = \frac{e_{k} + e_{k+1}}{2},
\qquad
k = \operatorname{argmax}_m H_m,
\]

where \(e\) are the bin edges and \(H\) is the histogram of
\(\mathcal{D}\). A tie keeps the lowest such centre. \(\tau\) is in
kilometres.

Rows with `dense_group_id` equal to \(-1\) are dropped. The 3D
distances of the remaining rows are computed again, after the same
removal of non-finite coordinate rows from the array only.
`squareform(..., checks=False)` reads the upper triangle and does not
average \(d_{ij}\) with \(d_{ji}\). Centroid linkage of that condensed
vector is cut by `fcluster` at distance \(\tau\). The labels, which
start at 1, are stored in `cluster` without renumbering.

## 9. Figures

The histogram drawn for the catalogue uses 40 bins and
density-normalised counts. It is a separate histogram from the one that
defines \(\tau\). One curve is the positive upper triangle of the full
distance matrix. The other is the positive upper triangle of the dense
distance matrix. Bin centres and the two density values are written to
the histogram CSV.

The dendrogram is the patch linkage. It is truncated at level 5
(`truncate_mode="level"`, `p=5`). A horizontal line is drawn at
\(\tau\), and the colour threshold of the dendrogram is \(\tau\).

Axis text requests Times New Roman. The geometry of either figure does
not depend on that font.


## Test results

The catalogue `catalog_bacground_proj.csv` is a Kuril–Kamchatka
background file: 4131 events, every `Aftershock` value false, magnitude
4.4 to 6.9, latitude 41.902° to 56.284°, longitude 144.32° to 163.97°,
depth 10 km to 270 km, from 16 August 1990 to 17 August 2026. It is not
shipped in this repository. The run that produced the numbers and the
figures is written out in [`results.md`](results.md).

## References

Ostapchuk, A.; Nugmanov, I. Background Seismicity Highlights Tectonic
Asperities. *Geosciences* 2026, *16*, 38.
<https://doi.org/10.3390/geosciences16010038>

Agayan, S. M.; Bogoutdinov, Sh. R.; Dobrovolsky, M. N. Discrete Perfect
Sets and Their Application in Cluster Analysis. *Cybernetics and Systems
Analysis* 2014, *50*(2), 176–190.
