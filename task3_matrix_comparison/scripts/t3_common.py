"""
t3_common.py
============

Statistical building blocks for comparing two matrices (T3_M1 and T3_M2) where
rows are categories and columns are independent units, for example independent
samples, replicates or conditions.

The question behind Task 3
---------------------------
"Can you think of a statistical test to test for differences or similarity
between two matrices, preserving the matrix structure, given that the columns
are independent of each other."

The approach used here
-----------------------
Treat each column as its own 2-by-R contingency table (M1's column vs M2's
column, R categories), test each one on its own terms, and then combine the
five independent pieces of evidence in two complementary ways:

    Fisher's combined probability test
        Combines the five independent column p-values into one omnibus
        p-value. This is the most direct use of "the columns are
        independent": Fisher's method is only valid when the p-values being
        combined come from independent tests.

    Generalized Cochran-Mantel-Haenszel (CMH) test
        Treats each column as an independent stratum and combines the raw
        association evidence (not just the p-values) into one statistic. It
        is the standard tool from epidemiology and biostatistics for exactly
        this situation: an R-category outcome, two groups being compared,
        and several independent strata. It has more power than Fisher's
        method when the direction of the difference is consistent across
        columns, because it pools the actual effect rather than just the
        p-values.

Both of these are cross-checked with a Monte Carlo permutation calibration,
because two of the five columns have small enough counts that the usual
chi-square approximation is not fully trustworthy (an expected cell count
below 5 in two of the five columns), and simulation-based p-values do not
depend on that approximation at all.
"""

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matrix(path):
    """Read a plain tab or whitespace separated numeric matrix, no header.

    Returns a float64 numpy array. Kept deliberately simple: T3_M1.txt and
    T3_M2.txt are bare grids of non-negative counts with no row or column
    labels, so there is nothing to sniff out beyond the numbers themselves.
    """
    return np.loadtxt(path, dtype=float)


def check_same_shape(M1, M2, name1="M1", name2="M2"):
    """Refuse to compare two matrices that are not even the same shape.

    A column-by-column comparison only makes sense if column j of M1 and
    column j of M2 are the same category axis, which requires identical
    dimensions. Failing loudly here is much better than failing quietly
    three functions later with a confusing broadcasting error.
    """
    if M1.shape != M2.shape:
        raise ValueError(
            "{} has shape {} but {} has shape {}. A column-wise comparison "
            "requires both matrices to share the same rows (categories) and "
            "columns (independent units).".format(
                name1, M1.shape, name2, M2.shape))


# ---------------------------------------------------------------------------
# A single, numerically safe G-test (log-likelihood ratio test)
# ---------------------------------------------------------------------------

def g_test_2xR(group1_counts, group2_counts):
    """G-test (log-likelihood ratio) for a single 2 x R table.

    group1_counts and group2_counts are both length-R arrays, one row of the
    2 x R contingency table each. Returns (G statistic, degrees of freedom,
    asymptotic p-value against a chi-square distribution).

    The convention x * log(x) = 0 when x = 0 is applied explicitly, since a
    category that was not observed at all in one group contributes nothing to
    the statistic rather than causing a divide by zero or a log of zero.
    """
    group1_counts = np.asarray(group1_counts, dtype=float)
    group2_counts = np.asarray(group2_counts, dtype=float)
    R = len(group1_counts)

    row_totals = np.array([group1_counts.sum(), group2_counts.sum()])
    col_totals = group1_counts + group2_counts
    grand_total = row_totals.sum()

    expected1 = col_totals * row_totals[0] / grand_total
    expected2 = col_totals * row_totals[1] / grand_total

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(group1_counts > 0,
                        group1_counts * np.log(group1_counts / expected1), 0.0)
        term2 = np.where(group2_counts > 0,
                        group2_counts * np.log(group2_counts / expected2), 0.0)

    statistic = float(2.0 * (term1.sum() + term2.sum()))
    dof = R - 1
    pvalue = float(stats.chi2.sf(statistic, dof))
    return statistic, dof, pvalue, np.vstack([expected1, expected2])


def g_test_2xR_batch(group1_counts, group2_counts):
    """Vectorized version of g_test_2xR for a batch of simulated tables.

    group1_counts and group2_counts both have shape (n_replicates, R). Returns
    an array of G statistics, one per replicate. Used by the Monte Carlo
    calibration, where recomputing the test tens of thousands of times with a
    Python-level loop would be needlessly slow.
    """
    row1 = group1_counts.sum(axis=1, keepdims=True)
    row2 = group2_counts.sum(axis=1, keepdims=True)
    col_totals = group1_counts + group2_counts
    grand = row1 + row2

    expected1 = col_totals * row1 / grand
    expected2 = col_totals * row2 / grand

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(group1_counts > 0,
                        group1_counts * np.log(group1_counts / expected1), 0.0)
        term2 = np.where(group2_counts > 0,
                        group2_counts * np.log(group2_counts / expected2), 0.0)

    return 2.0 * (term1.sum(axis=1) + term2.sum(axis=1))


def chi2_pearson_2xR(group1_counts, group2_counts):
    """Ordinary Pearson chi-square statistic for a single 2 x R table.
    """
    group1_counts = np.asarray(group1_counts, dtype=float)
    group2_counts = np.asarray(group2_counts, dtype=float)
    row_totals = np.array([group1_counts.sum(), group2_counts.sum()])
    col_totals = group1_counts + group2_counts
    grand_total = row_totals.sum()
    expected1 = col_totals * row_totals[0] / grand_total
    expected2 = col_totals * row_totals[1] / grand_total
    statistic = float(((group1_counts - expected1) ** 2 / expected1).sum() +
                      ((group2_counts - expected2) ** 2 / expected2).sum())
    dof = len(group1_counts) - 1
    pvalue = float(stats.chi2.sf(statistic, dof))
    return statistic, dof, pvalue, np.vstack([expected1, expected2])


# ---------------------------------------------------------------------------
# Combining independent per-column p-values
# ---------------------------------------------------------------------------

def fisher_combine(pvalues):
    """Fisher's combined probability test.

    Statistic is -2 * sum(log(p_i)), which follows a chi-square distribution
    with 2k degrees of freedom (k = number of tests) UNDER THE ASSUMPTION THAT
    THE TESTS ARE INDEPENDENT. That assumption is exactly what "the columns
    are independent of each other" buys us here: without it, this combination
    formula would not be valid.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    # Guard against a p-value of exactly 0 from floating point underflow,
    # which would send the statistic to infinity. A tiny floor has no
    # practical effect on the result but keeps the arithmetic well defined.
    clipped = np.clip(pvalues, 1e-300, 1.0)
    statistic = float(-2.0 * np.sum(np.log(clipped)))
    dof = 2 * len(pvalues)
    pvalue = float(stats.chi2.sf(statistic, dof))
    return statistic, dof, pvalue


def stouffer_combine(pvalues, weights=None):
    """Stouffer's Z-score method for combining independent p-values.

    Reported as a cross-check on Fisher's method. The two usually agree; if
    they were to disagree strongly it would be a sign that one or two columns
    with extreme p-values are dominating Fisher's statistic (which weights the
    log of the p-value, giving outsized influence to very small p-values)
    where Stouffer's method, especially when weighted by each column's sample
    size, gives a more evenly balanced combination.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    clipped = np.clip(pvalues, 1e-300, 1 - 1e-16)
    z_scores = stats.norm.isf(clipped)
    if weights is None:
        weights = np.ones(len(pvalues))
    weights = np.asarray(weights, dtype=float)
    combined_z = float((weights * z_scores).sum() / np.sqrt((weights ** 2).sum()))
    pvalue = float(stats.norm.sf(combined_z))
    return combined_z, pvalue


def harmonic_mean_pvalue(pvalues, weights=None):
    """The harmonic mean p-value (Wilson, 2019), a robustness cross-check.

    Fisher's method combines evidence through a sum of logs, which means one
    very small p-value can dominate the combined statistic. The harmonic mean
    p-value is far less sensitive to a single extreme value and is a useful
    second opinion when Fisher's result is being driven by one standout
    column. It does not have a clean parametric null in general, so here it is
    reported as a descriptive combined figure alongside the Monte Carlo
    calibration, rather than as a test with its own asymptotic p-value.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    if weights is None:
        weights = np.ones(len(pvalues))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    return float(1.0 / np.sum(weights / pvalues))


# ---------------------------------------------------------------------------
# Generalized Cochran-Mantel-Haenszel test (R categories, K independent strata)
# ---------------------------------------------------------------------------

def generalized_cmh(group1_by_column, group2_by_column):
    """CMH general-association statistic extended from 2x2xK to RxCxK.
    """
    K = len(group1_by_column)
    R = len(group1_by_column[0])

    total_deviation = np.zeros(R - 1)
    total_covariance = np.zeros((R - 1, R - 1))

    for k in range(K):
        counts1 = np.asarray(group1_by_column[k], dtype=float)
        counts2 = np.asarray(group2_by_column[k], dtype=float)
        category_totals = counts1 + counts2
        n_group1 = counts1.sum()
        n_group2 = counts2.sum()
        n = n_group1 + n_group2
        if n <= 1:
            continue          

        expected1 = category_totals * n_group1 / n


        observed_reduced = counts1[:-1]
        expected_reduced = expected1[:-1]
        category_totals_reduced = category_totals[:-1]

        total_deviation += (observed_reduced - expected_reduced)

        variance_factor = (n_group1 * n_group2) / (n ** 2 * (n - 1))
        stratum_covariance = np.zeros((R - 1, R - 1))
        for a in range(R - 1):
            for b in range(R - 1):
                if a == b:
                    stratum_covariance[a, b] = (
                        variance_factor * category_totals_reduced[a] *
                        (n - category_totals_reduced[a]))
                else:
                    stratum_covariance[a, b] = (
                        -variance_factor * category_totals_reduced[a] *
                        category_totals_reduced[b])
        total_covariance += stratum_covariance


    statistic = float(total_deviation @ np.linalg.pinv(total_covariance) @
                      total_deviation)
    dof = R - 1
    pvalue = float(stats.chi2.sf(statistic, dof))
    return statistic, dof, pvalue


# ---------------------------------------------------------------------------
# Monte Carlo calibration
# ---------------------------------------------------------------------------

def monte_carlo_calibration(M1, M2, n_simulations=200000, seed=20260726):
    """Simulate the null distribution of the per-column and combined
    statistics directly, rather than trusting the chi-square approximation.
    """
    rng = np.random.default_rng(seed)
    R, C = M1.shape

    column_totals_1 = M1.sum(axis=0)
    column_totals_2 = M2.sum(axis=0)
    pooled_proportions = [
        (M1[:, j] + M2[:, j]) / (M1[:, j] + M2[:, j]).sum() for j in range(C)
    ]

    per_column_observed_g = np.zeros(C)
    per_column_mc_pvalue = np.zeros(C)
    combined_simulated = np.zeros(n_simulations)

    for j in range(C):
        observed_g, _, _, _ = g_test_2xR(M1[:, j], M2[:, j])
        per_column_observed_g[j] = observed_g

        simulated_group1 = rng.multinomial(
            int(column_totals_1[j]), pooled_proportions[j], size=n_simulations
        ).astype(float)
        simulated_group2 = rng.multinomial(
            int(column_totals_2[j]), pooled_proportions[j], size=n_simulations
        ).astype(float)

        simulated_g = g_test_2xR_batch(simulated_group1, simulated_group2)
        per_column_mc_pvalue[j] = (simulated_g >= observed_g).mean()
        combined_simulated += simulated_g

    combined_observed = per_column_observed_g.sum()
    # Adding 1 to numerator and denominator (a standard correction for Monte
    # Carlo p-values) avoids ever reporting an impossible p-value of exactly
    # zero: with a finite number of simulations, "we saw nothing this extreme"
    # should be reported as "less than roughly 1 in n_simulations", not as
    # literally zero.
    combined_mc_pvalue = (np.sum(combined_simulated >= combined_observed) + 1) / \
        (n_simulations + 1)

    return {
        "per_column_observed_g": per_column_observed_g,
        "per_column_mc_pvalue": per_column_mc_pvalue,
        "combined_observed": combined_observed,
        "combined_mc_pvalue": combined_mc_pvalue,
        "combined_null_distribution": combined_simulated,
        "n_simulations": n_simulations,
    }


# ---------------------------------------------------------------------------
# Effect size and similarity metrics (the "or similarity" half of the task)
# ---------------------------------------------------------------------------

def cramers_v(group1_counts, group2_counts):
    """Cramer's V effect size for a 2 x R table, from 0 (no association, the
    two columns look identical) to 1 (maximum possible association).

    A p-value only tells you whether a difference is detectable given the
    sample size, not whether it is large. Cramer's V is reported alongside
    every test so that a column with a tiny but real difference (small V,
    significant p because the counts are large) can be told apart from a
    column with a large, substantial difference.
    """
    chi2, dof_plus_one, _, _ = chi2_pearson_2xR(group1_counts, group2_counts)
    n = np.asarray(group1_counts).sum() + np.asarray(group2_counts).sum()
    r = 2                     # two groups
    k = len(group1_counts)     # categories
    min_dim = min(r - 1, k - 1)
    return float(np.sqrt(chi2 / (n * min_dim)))


def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence between two probability distributions.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(a > 0, a * np.log2(a / b), 0.0)
        return terms.sum()

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors, 1 meaning identical direction.

    Reported per column as a simple, scale-free similarity measure that a
    non-statistical reader can sanity check by eye.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    return float((a @ b) / denominator)


# ---------------------------------------------------------------------------
# The two naive comparisons kept around purely to demonstrate why they fail
# ---------------------------------------------------------------------------

def naive_flatten_test(M1, M2):
    """Paired t-test and Wilcoxon signed rank on the raw flattened cell values.
    """
    flat1 = M1.flatten()
    flat2 = M2.flatten()
    t_statistic, t_pvalue = stats.ttest_rel(flat1, flat2)
    try:
        w_statistic, w_pvalue = stats.wilcoxon(flat1, flat2)
    except ValueError:
        w_statistic, w_pvalue = float("nan"), float("nan")
    return {
        "t_statistic": float(t_statistic), "t_pvalue": float(t_pvalue),
        "wilcoxon_statistic": float(w_statistic), "wilcoxon_pvalue": float(w_pvalue),
    }


def naive_pooled_test(M1, M2):
    """Sum away the columns and run one chi-square test on the row totals.

    Kept for the same reason as naive_flatten_test: to show concretely what
    gets lost (which columns actually differ, and the fact that one column
    shows no difference at all) when the column structure is discarded before
    testing.
    """
    row_totals_1 = M1.sum(axis=1)
    row_totals_2 = M2.sum(axis=1)
    table = np.vstack([row_totals_1, row_totals_2])
    chi2, pvalue, dof, expected = stats.chi2_contingency(table)
    return {
        "chi2": float(chi2), "pvalue": float(pvalue), "dof": int(dof),
        "expected": expected, "row_totals_1": row_totals_1,
        "row_totals_2": row_totals_2,
    }
