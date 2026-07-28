"""
t2_common.py
============

Shared loading, quality control and preprocessing for the Task 2 scripts
(05_cluster_genes.py and 06_cluster_samples.py). 

"""

import numpy as np


# ---------------------------------------------------------------------------
# k-means, implemented here rather than imported from scikit-learn
# ---------------------------------------------------------------------------


def _squared_distances(points, centres):
    """Squared Euclidean distance from every point to every centre.

    Uses the expansion ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 so the whole thing
    is one matrix multiply instead of a Python loop. Tiny negative values can
    appear from floating point cancellation, so we clip at zero.
    """
    return np.maximum(
        (points ** 2).sum(axis=1)[:, None]
        - 2.0 * points @ centres.T
        + (centres ** 2).sum(axis=1)[None, :],
        0.0)


def _kmeans_plusplus(points, k, rng):
    """k-means++ seeding.

    Picks the first centre uniformly, then each subsequent centre with
    probability proportional to its squared distance from the nearest centre
    chosen so far. This spreads the starting centres out, which matters because
    plain random seeding on expression data regularly lands two centres inside
    the same dense blob and converges to a visibly wrong answer.
    """
    n_points = points.shape[0]
    centres = np.empty((k, points.shape[1]), dtype=points.dtype)
    centres[0] = points[rng.integers(n_points)]

    closest = _squared_distances(points, centres[:1]).ravel()
    for index in range(1, k):
        total = closest.sum()
        if total <= 0:
            # Every point already sits on a centre. Fill the rest at random.
            centres[index] = points[rng.integers(n_points)]
        else:
            centres[index] = points[rng.choice(n_points, p=closest / total)]
        closest = np.minimum(
            closest, _squared_distances(points, centres[index:index + 1]).ravel())
    return centres


def _lloyd(points, centres, max_iter, tol):
    """One run of Lloyd's algorithm from a given starting set of centres."""
    labels = None
    for _ in range(max_iter):
        distances = _squared_distances(points, centres)
        new_labels = np.argmin(distances, axis=1)

        if labels is not None and np.array_equal(new_labels, labels):
            break                       
        labels = new_labels

        moved = 0.0
        for cluster in range(centres.shape[0]):
            members = points[labels == cluster]
            if len(members) == 0:
                # An empty cluster would collapse the solution to fewer than k
                # groups. Re-seed it on the point currently worst served by its
                # own centre, which is the standard repair and keeps k honest.
                worst = int(np.argmax(distances[np.arange(len(labels)), labels]))
                new_centre = points[worst]
            else:
                new_centre = members.mean(axis=0)
            moved += float(np.sum((new_centre - centres[cluster]) ** 2))
            centres[cluster] = new_centre

        if moved <= tol:
            break

    distances = _squared_distances(points, centres)
    labels = np.argmin(distances, axis=1)
    inertia = float(distances[np.arange(len(labels)), labels].sum())
    return labels, centres, inertia


def kmeans(points, k, n_init=10, max_iter=300, tol=1e-8, seed=0):
    """k-means with k-means++ seeding and `n_init` restarts, best inertia wins.

    Lloyd's algorithm only finds a local optimum, and which
    one it finds depends entirely on where it started. Taking the best of several
    independent starts is what makes the result reproducible in practice rather
    than in principle.

    Returns (labels, centres, inertia).
    """
    rng = np.random.default_rng(seed)
    points = np.ascontiguousarray(points, dtype=np.float64)

    best = None
    for _ in range(n_init):
        start = _kmeans_plusplus(points, k, rng)
        labels, centres, inertia = _lloyd(points, start.copy(), max_iter, tol)
        if best is None or inertia < best[2]:
            best = (labels, centres, inertia)
    return best


def assign_to_centroids(points, centres):
    """Label every point by its nearest centre. Used by the stability bootstrap,
    which fits on a subset of genes and then labels all of them."""
    return np.argmin(_squared_distances(
        np.asarray(points, dtype=np.float64),
        np.asarray(centres, dtype=np.float64)), axis=1)


def load_matrix(path):
    """Read the tab separated expression matrix.

    Returns (gene_names, sample_names, values) where values is a float array
    shaped (n_genes, n_samples). Deliberately hand rolled rather than using
    pandas so the failure modes are obvious and the row order is guaranteed to be
    the file order.
    """
    gene_names = []
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r").split("\t")
        sample_names = header[1:]
        for line_number, line in enumerate(handle, start=2):
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) != len(header):
                raise ValueError(
                    "Line {} has {} fields but the header has {}".format(
                        line_number, len(fields), len(header)))
            gene_names.append(fields[0])
            rows.append([float(value) for value in fields[1:]])
    return np.array(gene_names), np.array(sample_names), np.array(rows, dtype=float)


def describe_raw(values, sample_names):
    """Quality control summary of the matrix as it arrives, before any changes."""
    column_sums = values.sum(axis=0)
    log_values = np.log2(values + 1)
    column_medians = np.median(log_values, axis=0)
    return {
        "n_genes": values.shape[0],
        "n_samples": values.shape[1],
        "n_zeros": int((values == 0).sum()),
        "n_all_zero_genes": int((values == 0).all(axis=1).sum()),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "column_sums": column_sums,
        "column_sum_cv": float(column_sums.std() / column_sums.mean()),
        "log_column_medians": column_medians,
        "log_median_spread": float(column_medians.max() - column_medians.min()),
        "top10_share": float(
            np.sort(values.mean(axis=1))[::-1][:10].sum() / values.mean(axis=1).sum()),
        "sample_names": sample_names,
    }


def filter_genes(values, gene_names, min_mean_expression=1.0,
                 min_samples_above=3, min_expression_floor=1.0):
    """Drop genes that cannot support a meaningful expression pattern.

    Two conditions, both of which a gene must pass:
      - its mean expression across samples is at least `min_mean_expression`
      - it is above `min_expression_floor` in at least `min_samples_above` samples

    The second condition is the more useful of the two. A gene that is loud in a
    single sample and silent everywhere else will pass a mean threshold but is
    far more likely to be a technical spike than a coherent expression pattern,
    and with only 9 samples there is no way to tell the two apart. Requiring
    presence in at least a few samples keeps the clustering focused on genes
    where a pattern could actually be observed.

    These thresholds are gentle on purpose. T2 is already a well populated
    matrix, only 51 of its 87,912 values fall below 1, so this step removes very
    littl
    """
    mean_expression = values.mean(axis=1)
    n_above_floor = (values >= min_expression_floor).sum(axis=1)
    keep = (mean_expression >= min_mean_expression) & \
           (n_above_floor >= min_samples_above)

    # A zero variance gene would make the z-score step divide by zero. Such genes
    # are flat by definition and have no pattern to cluster on, so they go too.
    keep &= values.std(axis=1) > 0

    return values[keep], gene_names[keep], keep


def normalize_columns(log_values, method="median"):
    """Remove the per-sample global offset from log2 transformed data.
    """
    if method == "none":
        return log_values.copy(), np.zeros(log_values.shape[1])

    if method == "median":
        column_medians = np.median(log_values, axis=0)
        overall_median = np.median(column_medians)
        offsets = column_medians - overall_median
        return log_values - offsets[None, :], offsets

    if method == "quantile":
        # Average the sorted values across columns to build one shared reference
        # distribution, then substitute each column's ranks into it.
        order = np.argsort(log_values, axis=0)
        sorted_columns = np.sort(log_values, axis=0)
        reference = sorted_columns.mean(axis=1)
        result = np.empty_like(log_values)
        for column in range(log_values.shape[1]):
            result[order[:, column], column] = reference
        offsets = np.median(log_values, axis=0) - np.median(result, axis=0)
        return result, offsets

    raise ValueError("Unknown normalization method: {}".format(method))


def zscore_rows(matrix):
    """Centre and scale every row to mean 0 and standard deviation 1.
    """
    means = matrix.mean(axis=1, keepdims=True)
    stds = matrix.std(axis=1, ddof=0, keepdims=True)
    safe = stds.copy()
    safe[safe == 0] = 1.0
    result = (matrix - means) / safe
    result[stds.ravel() == 0, :] = 0.0
    return result


def preprocess(path, normalize="median", min_mean_expression=1.0,
               min_samples_above=3, min_expression_floor=1.0):
    """Run the whole chain and hand back everything the analyses need.

    Returns a dictionary rather than a long tuple, because the callers want
    different subsets of it and positional unpacking of eight values is a recipe
    for silent mix-ups.
    """
    gene_names, sample_names, raw = load_matrix(path)
    qc = describe_raw(raw, sample_names)

    filtered, kept_gene_names, keep_mask = filter_genes(
        raw, gene_names,
        min_mean_expression=min_mean_expression,
        min_samples_above=min_samples_above,
        min_expression_floor=min_expression_floor)

    log_values = np.log2(filtered + 1)
    log_normalized, offsets = normalize_columns(log_values, method=normalize)
    z = zscore_rows(log_normalized)

    return {
        "gene_names": kept_gene_names,
        "sample_names": sample_names,
        "raw": filtered,
        # log2 scale with the per-sample offset removed. This is the matrix to
        # use whenever amplitude matters, for example when reporting how large a
        # cluster's fold change actually is.
        "log": log_normalized,
        # the same data with each gene standardised. This is what gets clustered.
        "z": z,
        "normalization_offsets": offsets,
        "normalization_method": normalize,
        "n_genes_before_filter": len(gene_names),
        "n_genes_removed": int((~keep_mask).sum()),
        "removed_gene_names": gene_names[~keep_mask],
        "qc": qc,
    }
