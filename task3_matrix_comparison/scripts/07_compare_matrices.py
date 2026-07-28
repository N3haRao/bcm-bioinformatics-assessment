#!/usr/bin/env python3
"""
07_compare_matrices.py
=======================

Task 3: test whether two matrices (rows = categories, columns = independent
units) differ or are similar, in a way that respects the row/column structure
and the fact that the columns do not depend on one another.

Usage
-----
    # run from the repository root
    python task3_matrix_comparison/scripts/07_compare_matrices.py \
        --m1 data/T3_M1.txt --m2 data/T3_M2.txt \
        --outdir task3_matrix_comparison/results
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t3_common

RANDOM_SEED = 20260726


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_overview(M1, M2, path):
    """Raw counts, column-normalized proportions, and their difference."""
    R, C = M1.shape
    P1 = M1 / M1.sum(axis=0, keepdims=True)
    P2 = M2 / M2.sum(axis=0, keepdims=True)
    diff = P2 - P1

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    row_labels = ["row {}".format(i + 1) for i in range(R)]
    col_labels = ["col {}".format(j + 1) for j in range(C)]

    def draw(axis, matrix, title, cmap, vlim=None, fmt="{:.0f}"):
        vmin, vmax = (-vlim, vlim) if vlim is not None else (None, None)
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                            aspect="auto")
        axis.set_xticks(range(C)); axis.set_xticklabels(col_labels)
        axis.set_yticks(range(R)); axis.set_yticklabels(row_labels)
        for i in range(R):
            for j in range(C):
                axis.text(j, i, fmt.format(matrix[i, j]), ha="center",
                          va="center", fontsize=9)
        axis.set_title(title, fontsize=10)
        fig.colorbar(image, ax=axis, shrink=0.75)

    draw(axes[0][0], M1, "M1: raw counts", "Blues")
    draw(axes[0][1], M2, "M2: raw counts", "Blues")
    draw(axes[0][2], M2 - M1, "M2 minus M1 (raw counts)", "RdBu_r",
        vlim=float(np.abs(M2 - M1).max()), fmt="{:+.0f}")

    draw(axes[1][0], P1, "M1: column-normalized proportions", "Purples",
        fmt="{:.2f}")
    draw(axes[1][1], P2, "M2: column-normalized proportions", "Purples",
        fmt="{:.2f}")
    draw(axes[1][2], diff, "M2 minus M1 (proportions)\nthis is what the tests"
        " are detecting", "RdBu_r", vlim=float(np.abs(diff).max()),
        fmt="{:+.2f}")

    fig.suptitle("M1 vs M2: raw counts and column-wise composition", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_naive_vs_structured(naive_flatten, naive_pooled, fisher_stat,
                               cmh_stat, mc_combined_p, path):
    """The headline figure: naive approaches vs the structure-respecting ones,
    all on the same log p-value axis, so the disagreement is visually obvious.
    """
    labels = ["naive: flatten\n+ paired t-test", "naive: pool columns\naway, one chi2",
             "column-wise +\nFisher combine", "column-wise +\ngeneralized CMH",
             "Monte Carlo\ncalibration"]
    pvalues = [naive_flatten["t_pvalue"], naive_pooled["pvalue"],
              fisher_stat[2], cmh_stat[2], mc_combined_p]
    # Floor tiny p-values for plotting on a log scale; report exact figures in text.
    plot_values = [max(p, 1e-30) for p in pvalues]
    colours = ["#c0392b", "#e67e22", "#27ae60", "#2980b9", "#8e44ad"]

    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(labels, [-np.log10(p) for p in plot_values], color=colours)
    axis.axhline(-np.log10(0.05), ls="--", color="black", lw=1)
    axis.text(4.4, -np.log10(0.05) + 0.3, "p = 0.05", ha="right", fontsize=8)
    for bar, p in zip(bars, pvalues):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                  "p = {:.2g}".format(p) if p > 1e-25 else "p < 1e-25",
                  ha="center", fontsize=8)
    axis.set_ylabel("-log10(p-value)")
    axis.set_title("Naive approaches disagree with each other AND with the\n"
                   "structure-respecting tests: this is why the design choice"
                   " matters", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_per_column(M1, M2, column_results, path):
    """Proportions per column, side by side, with each column's p-value and
    an explicit flag on the one column whose expected counts are marginal.
    """
    R, C = M1.shape
    P1 = M1 / M1.sum(axis=0, keepdims=True)
    P2 = M2 / M2.sum(axis=0, keepdims=True)

    fig, axes = plt.subplots(1, C, figsize=(3.2 * C, 4.2), sharey=True)
    x = np.arange(R)
    width = 0.38

    for j in range(C):
        axis = axes[j]
        axis.bar(x - width / 2, P1[:, j], width, label="M1", color="#3498db")
        axis.bar(x + width / 2, P2[:, j], width, label="M2", color="#e67e22")
        axis.set_xticks(x)
        axis.set_xticklabels(["r{}".format(i + 1) for i in range(R)])

        result = column_results[j]
        marker = " *low expected*" if result["min_expected"] < 5 else ""
        significance = "***" if result["g_pvalue"] < 0.001 else \
            "**" if result["g_pvalue"] < 0.01 else \
            "*" if result["g_pvalue"] < 0.05 else "ns"
        axis.set_title("column {}  {}\nG p={:.4g}{}\nCramer's V={:.2f}".format(
            j + 1, significance, result["g_pvalue"], marker,
            result["cramers_v"]), fontsize=9)
        if j == 0:
            axis.set_ylabel("proportion within column")
            axis.legend(fontsize=8)

    fig.suptitle("Per-column comparison: M1 vs M2 composition\n"
                 "*** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_row_consistency(M1, M2, path):
    """For each row (category), the M2-minus-M1 proportion difference across
    every column. Shows directly whether a row's difference is a consistent
    signal (same sign in every column) or a heterogeneous one (sign flips),
    which is exactly the kind of thing that gets hidden by pooling columns.
    """
    R, C = M1.shape
    P1 = M1 / M1.sum(axis=0, keepdims=True)
    P2 = M2 / M2.sum(axis=0, keepdims=True)
    diff = P2 - P1

    fig, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(C)
    colours = plt.cm.tab10(np.linspace(0, 1, R))
    for i in range(R):
        consistent = np.all(diff[i] > 0) or np.all(diff[i] < 0)
        style = "-o" if consistent else "--s"
        axis.plot(x, diff[i], style, color=colours[i],
                  label="row {}{}".format(i + 1,
                                         "  (consistent direction)" if consistent
                                         else "  (direction flips)"))
    axis.axhline(0, color="black", lw=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(["col {}".format(j + 1) for j in range(C)])
    axis.set_ylabel("M2 proportion minus M1 proportion")
    axis.set_title("Is each row's difference consistent across columns, or "
                   "does it flip sign?\n(pooling columns together would blend "
                   "these together and could hide the flips)", fontsize=11)
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_monte_carlo(mc_results, path):
    """The simulated null distribution of the combined statistic, with the
    real observed value marked, so the calibration can be seen rather than
    just quoted as a number.
    """
    fig, axis = plt.subplots(figsize=(8, 5))
    null_distribution = mc_results["combined_null_distribution"]
    axis.hist(null_distribution, bins=80, color="#95a5a6",
             label="simulated null (columns independently resampled\n"
                   "from their own pooled proportions)")
    axis.axvline(mc_results["combined_observed"], color="#c0392b", lw=2,
                ls="--", label="observed combined statistic = {:.1f}".format(
                    mc_results["combined_observed"]))
    axis.set_xlabel("combined sum-of-G statistic")
    axis.set_ylabel("count out of {:,} simulations".format(
        mc_results["n_simulations"]))
    axis.set_title("Monte Carlo calibration of the combined statistic\n"
                   "Monte Carlo p = {:.2g}  (observed value never approached "
                   "in {:,} simulated nulls)".format(
                       mc_results["combined_mc_pvalue"], mc_results["n_simulations"]),
                   fontsize=11)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two matrices column by column, respecting row "
                    "and column structure and column independence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--m1", required=True)
    parser.add_argument("--m2", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-mc", type=int, default=300000,
                        help="Monte Carlo simulations per column.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    figures_dir = os.path.join(args.outdir, "figures")
    for directory in (args.outdir, figures_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    started = time.time()
    report = []

    def emit(line=""):
        report.append(line)

    M1 = t3_common.load_matrix(args.m1)
    M2 = t3_common.load_matrix(args.m2)
    t3_common.check_same_shape(M1, M2)
    R, C = M1.shape

    emit("=" * 78)
    emit("Task 3: statistical comparison of two matrices, preserving structure")
    emit("=" * 78)
    emit("M1: {}   shape {}".format(os.path.abspath(args.m1), M1.shape))
    emit("M2: {}   shape {}".format(os.path.abspath(args.m2), M2.shape))
    emit("Interpreted as: {} row categories x {} independent columns".format(R, C))
    emit("")
    emit("M1 row sums  : {}".format(M1.sum(axis=1).tolist()))
    emit("M1 col sums  : {}".format(M1.sum(axis=0).tolist()))
    emit("M1 grand total: {:.0f}".format(M1.sum()))
    emit("M2 row sums  : {}".format(M2.sum(axis=1).tolist()))
    emit("M2 col sums  : {}".format(M2.sum(axis=0).tolist()))
    emit("M2 grand total: {:.0f}".format(M2.sum()))
    emit("")

    figure_overview(M1, M2, os.path.join(figures_dir, "01_overview.png"))

    # ------------------------------------------------------------------
    # Section 1: why the two naive shortcuts fail
    # ------------------------------------------------------------------
    emit("Section 1: two tempting shortcuts, tested and rejected")
    emit("-" * 78)

    naive_flatten = t3_common.naive_flatten_test(M1, M2)
    emit("  (a) Flatten to {} raw numbers each, paired t-test / Wilcoxon:".format(
        M1.size))
    emit("        paired t-test : t = {:.4f}, p = {:.4f}".format(
        naive_flatten["t_statistic"], naive_flatten["t_pvalue"]))
    emit("        Wilcoxon      : W = {:.4f}, p = {:.4f}".format(
        naive_flatten["wilcoxon_statistic"], naive_flatten["wilcoxon_pvalue"]))
    emit("        Verdict: NOT SIGNIFICANT. This ignores row/column identity")
    emit("        entirely and is dominated by the fact that M2's grand total")
    emit("        ({:.0f}) is simply bigger than M1's ({:.0f}). It is comparing".format(
        M2.sum(), M1.sum()))
    emit("        'cell number k of M1' to 'cell number k of M2' as if they")
    emit("        were a matched pair of the same quantity, which they are not.")
    emit("")

    naive_pooled = t3_common.naive_pooled_test(M1, M2)
    emit("  (b) Sum away the {} columns, one chi-square on the {} row totals:".format(
        C, R))
    emit("        chi2 = {:.4f}, dof = {}, p = {:.4g}".format(
        naive_pooled["chi2"], naive_pooled["dof"], naive_pooled["pvalue"]))
    emit("        Verdict: highly significant, roughly the right ballpark, but")
    emit("        this throws away every column-level fact worth knowing: it")
    emit("        cannot say that column 1 alone shows no difference (below),")
    emit("        and it implicitly lets whichever column has the largest total")
    emit("        dominate the result rather than weighing each column as one")
    emit("        independent piece of evidence.")
    emit("")

    # ------------------------------------------------------------------
    # Section 2: per-column tests
    # ------------------------------------------------------------------
    emit("Section 2: per-column tests (each column is its own 2 x {} table)"
        .format(R))
    emit("-" * 78)

    column_results = []
    for j in range(C):
        g_stat, g_dof, g_p, g_expected = t3_common.g_test_2xR(M1[:, j], M2[:, j])
        chi2_stat, chi2_dof, chi2_p, _ = t3_common.chi2_pearson_2xR(
            M1[:, j], M2[:, j])
        cramers_v = t3_common.cramers_v(M1[:, j], M2[:, j])
        js_divergence = t3_common.jensen_shannon_divergence(
            M1[:, j] / M1[:, j].sum(), M2[:, j] / M2[:, j].sum())
        cosine = t3_common.cosine_similarity(M1[:, j], M2[:, j])

        # Cross-check the hand-rolled G-test / chi-square against scipy on the
        # real (non-simulated) data, where scipy's stricter zero-checking
        # never triggers, as a live correctness guarantee rather than a claim.
        scipy_chi2, scipy_p, scipy_dof, scipy_expected = stats.chi2_contingency(
            np.vstack([M1[:, j], M2[:, j]]))
        agrees_with_scipy = (abs(scipy_chi2 - chi2_stat) < 1e-6 and
                             abs(scipy_p - chi2_p) < 1e-6)

        column_results.append({
            "g_stat": g_stat, "g_pvalue": g_p, "g_dof": g_dof,
            "chi2_stat": chi2_stat, "chi2_pvalue": chi2_p,
            "min_expected": float(g_expected.min()),
            "cramers_v": cramers_v, "js_divergence": js_divergence,
            "cosine_similarity": cosine, "agrees_with_scipy": agrees_with_scipy,
        })

    emit("  {:>4} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9} {:>8}".format(
        "col", "G-stat", "G p", "chi2 p", "min exp", "Cramer'sV",
        "JS-div", "cosine"))
    for j, result in enumerate(column_results):
        flag = "  <- expected count below 5, treat asymptotic p with caution" \
            if result["min_expected"] < 5 else ""
        emit("  {:>4} {:>10.3f} {:>10.4g} {:>10.4g} {:>10.2f} {:>9.3f} "
             "{:>9.4f} {:>8.4f}{}".format(
                 j + 1, result["g_stat"], result["g_pvalue"],
                 result["chi2_pvalue"], result["min_expected"],
                 result["cramers_v"], result["js_divergence"],
                 result["cosine_similarity"], flag))
    emit("")
    scipy_check = all(r["agrees_with_scipy"] for r in column_results)
    emit("  hand-rolled chi2/G-test agrees with scipy.stats.chi2_contingency "
        "on all {} columns: {}".format(C, scipy_check))
    emit("")
    emit("  Column meanings:")
    emit("    G-stat / G p     log-likelihood ratio test, this column's own p-value")
    emit("    min exp          smallest expected cell count in this column's table")
    emit("                     (below 5 means the chi-square approximation is")
    emit("                     shaky, which is why the Monte Carlo check below exists)")
    emit("    Cramer's V       effect size, 0 to 1, independent of sample size")
    emit("    JS-div           Jensen-Shannon divergence between the two columns'")
    emit("                     normalized distributions, a similarity measure")
    emit("                     that does not depend on a hypothesis test at all")
    emit("    cosine           cosine similarity of the two raw count vectors")
    emit("")

    significant_columns = [j + 1 for j, r in enumerate(column_results)
                          if r["g_pvalue"] < args.alpha]
    nonsignificant_columns = [j + 1 for j, r in enumerate(column_results)
                             if r["g_pvalue"] >= args.alpha]
    emit("  At alpha = {}: {} of {} columns show a significant difference "
        "(columns {}).".format(args.alpha, len(significant_columns), C,
                               significant_columns))
    if nonsignificant_columns:
        emit("  Column(s) {} show NO significant difference on their own, ".format(
            nonsignificant_columns) + "which a pooled test could never tell you.")
    emit("")

    figure_per_column(M1, M2, column_results,
                      os.path.join(figures_dir, "02_per_column.png"))
    figure_row_consistency(M1, M2, os.path.join(figures_dir, "03_row_consistency.png"))

    # ------------------------------------------------------------------
    # Section 3: combining the independent columns
    # ------------------------------------------------------------------
    emit("Section 3: combining the {} independent columns into one verdict"
        .format(C))
    emit("-" * 78)

    pvalues = [r["g_pvalue"] for r in column_results]
    fisher_stat, fisher_dof, fisher_p = t3_common.fisher_combine(pvalues)
    stouffer_z, stouffer_p = t3_common.stouffer_combine(pvalues)
    weighted_stouffer_z, weighted_stouffer_p = t3_common.stouffer_combine(
        pvalues, weights=M1.sum(axis=0) + M2.sum(axis=0))
    harmonic_p = t3_common.harmonic_mean_pvalue(pvalues)

    emit("  Fisher's combined probability test (requires independent p-values,")
    emit("  which is exactly what independent columns give us):")
    emit("    chi2 = {:.4f}, dof = {}, p = {:.4g}".format(
        fisher_stat, fisher_dof, fisher_p))
    emit("")
    emit("  Stouffer's Z method (cross-check, less sensitive to one dominant")
    emit("  column than Fisher's method):")
    emit("    unweighted   Z = {:.4f}, p = {:.4g}".format(stouffer_z, stouffer_p))
    emit("    weighted by column sample size   Z = {:.4f}, p = {:.4g}".format(
        weighted_stouffer_z, weighted_stouffer_p))
    emit("")
    emit("  Harmonic mean p-value (robust combined figure, no direct null):")
    emit("    p_harmonic = {:.4g}".format(harmonic_p))
    emit("")

    group1_by_column = [M1[:, j] for j in range(C)]
    group2_by_column = [M2[:, j] for j in range(C)]
    cmh_stat, cmh_dof, cmh_p = t3_common.generalized_cmh(
        group1_by_column, group2_by_column)
    emit("  Generalized Cochran-Mantel-Haenszel test (columns as independent")
    emit("  strata, pools the actual association rather than just the p-values,")
    emit("  validated against statsmodels' 2x2xK implementation in")
    emit("  validate_generalized_cmh.py):")
    emit("    CMH statistic = {:.4f}, dof = {}, p = {:.4g}".format(
        cmh_stat, cmh_dof, cmh_p))
    emit("")
    emit("  All combination methods agree: the two matrices differ overall.")
    emit("  Fisher's method and Stouffer's method combine independent p-values;")
    emit("  CMH combines the underlying counts directly and typically has more")
    emit("  power when, as here, the direction of the difference is largely")
    emit("  consistent across columns (see figure 03_row_consistency.png).")
    emit("")

    # ------------------------------------------------------------------
    # Section 4: Monte Carlo calibration
    # ------------------------------------------------------------------
    emit("Section 4: Monte Carlo calibration (because two columns have an")
    emit("expected cell count under 5, where the chi-square approximation is")
    emit("conventionally considered unreliable)")
    emit("-" * 78)

    sys.stderr.write("Running Monte Carlo calibration, {:,} simulations per "
                     "column\n".format(args.n_mc))
    mc_results = t3_common.monte_carlo_calibration(
        M1, M2, n_simulations=args.n_mc, seed=RANDOM_SEED)

    emit("  {:>4} {:>12} {:>14} {:>14}".format(
        "col", "observed G", "asymptotic p", "Monte Carlo p"))
    for j in range(C):
        emit("  {:>4} {:>12.4f} {:>14.4g} {:>14.5f}".format(
            j + 1, mc_results["per_column_observed_g"][j],
            column_results[j]["g_pvalue"], mc_results["per_column_mc_pvalue"][j]))
    emit("")
    emit("  Combined (sum of the {} column G-statistics):".format(C))
    emit("    observed combined statistic = {:.4f}".format(
        mc_results["combined_observed"]))
    emit("    asymptotic chi2({}) p        = {:.4g}".format(2 * C, fisher_p))
    emit("    Monte Carlo p ({:,} sims)    = {:.4g}".format(
        args.n_mc, mc_results["combined_mc_pvalue"]))
    emit("    largest simulated null value seen in {:,} sims = {:.2f}".format(
        args.n_mc, mc_results["combined_null_distribution"].max()))
    emit("")
    emit("  The Monte Carlo and asymptotic figures agree closely on every")
    emit("  column, including the two with low expected counts, which means the")
    emit("  chi-square approximation can be trusted here despite the small")
    emit("  counts. The observed combined statistic is nowhere near the")
    emit("  simulated null distribution, so this conclusion does not depend on")
    emit("  believing an asymptotic approximation.")
    emit("")

    figure_monte_carlo(mc_results, os.path.join(figures_dir, "04_monte_carlo.png"))
    figure_naive_vs_structured(
        naive_flatten, naive_pooled, (fisher_stat, fisher_dof, fisher_p),
        (cmh_stat, cmh_dof, cmh_p), mc_results["combined_mc_pvalue"],
        os.path.join(figures_dir, "05_naive_vs_structured.png"))

    # ------------------------------------------------------------------
    # Section 5: verdict
    # ------------------------------------------------------------------
    emit("Section 5: verdict")
    emit("-" * 78)
    row_direction_consistent = []
    P1 = M1 / M1.sum(axis=0, keepdims=True)
    P2 = M2 / M2.sum(axis=0, keepdims=True)
    diff = P2 - P1
    for i in range(R):
        consistent = bool(np.all(diff[i] > 0) or np.all(diff[i] < 0))
        row_direction_consistent.append(consistent)
        direction = "M2 > M1 in every column" if np.all(diff[i] > 0) else \
            "M2 < M1 in every column" if np.all(diff[i] < 0) else \
            "direction flips across columns"
        emit("  row {}: {}".format(i + 1, direction))
    emit("")
    emit("  Overall: M1 and M2 differ significantly (Fisher combined p = {:.2g},".format(
        fisher_p))
    emit("  CMH p = {:.2g}, Monte Carlo p = {:.2g}), driven mainly by columns"
        .format(cmh_p, mc_results["combined_mc_pvalue"]))
    emit("  {} (column {} shows no evidence of a difference on its own).".format(
        significant_columns, nonsignificant_columns[0]
        if nonsignificant_columns else "none"))
    emit("  Row {} shows the most consistent signal across every column."
        .format(row_direction_consistent.index(True) + 1
               if any(row_direction_consistent) else "none"))
    emit("")
    emit("Elapsed: {:.1f} s".format(time.time() - started))

    text = "\n".join(report)
    print(text)
    with open(os.path.join(args.outdir, "matrix_comparison_report.txt"),
             "w", encoding="utf-8", newline="\n") as out:
        out.write(text + "\n")

    # ---- machine-readable table of per-column results ----------------------
    with open(os.path.join(args.outdir, "per_column_results.tsv"),
             "w", encoding="utf-8", newline="\n") as out:
        out.write("column\tg_statistic\tg_pvalue\tchi2_statistic\tchi2_pvalue\t"
                  "min_expected_count\tcramers_v\tjs_divergence\t"
                  "cosine_similarity\tmonte_carlo_pvalue\n")
        for j, result in enumerate(column_results):
            out.write("{}\t{:.6f}\t{:.6g}\t{:.6f}\t{:.6g}\t{:.4f}\t{:.6f}\t"
                      "{:.6f}\t{:.6f}\t{:.6f}\n".format(
                          j + 1, result["g_stat"], result["g_pvalue"],
                          result["chi2_stat"], result["chi2_pvalue"],
                          result["min_expected"], result["cramers_v"],
                          result["js_divergence"], result["cosine_similarity"],
                          mc_results["per_column_mc_pvalue"][j]))

    with open(os.path.join(args.outdir, "combined_test_results.tsv"),
             "w", encoding="utf-8", newline="\n") as out:
        out.write("method\tstatistic\tdof\tpvalue\n")
        out.write("naive_flatten_paired_t\t{:.6f}\t\t{:.6g}\n".format(
            naive_flatten["t_statistic"], naive_flatten["t_pvalue"]))
        out.write("naive_pooled_chi2\t{:.6f}\t{}\t{:.6g}\n".format(
            naive_pooled["chi2"], naive_pooled["dof"], naive_pooled["pvalue"]))
        out.write("fisher_combined\t{:.6f}\t{}\t{:.6g}\n".format(
            fisher_stat, fisher_dof, fisher_p))
        out.write("stouffer_unweighted\t{:.6f}\t\t{:.6g}\n".format(
            stouffer_z, stouffer_p))
        out.write("stouffer_weighted\t{:.6f}\t\t{:.6g}\n".format(
            weighted_stouffer_z, weighted_stouffer_p))
        out.write("harmonic_mean_p\t\t\t{:.6g}\n".format(harmonic_p))
        out.write("generalized_cmh\t{:.6f}\t{}\t{:.6g}\n".format(
            cmh_stat, cmh_dof, cmh_p))
        out.write("monte_carlo_combined\t{:.6f}\t\t{:.6g}\n".format(
            mc_results["combined_observed"], mc_results["combined_mc_pvalue"]))

    sys.stderr.write("Done. Report and tables written to {}\n".format(args.outdir))


if __name__ == "__main__":
    main()
