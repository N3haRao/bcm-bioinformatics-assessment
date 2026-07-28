#!/usr/bin/env python3
"""
06_cluster_samples.py
=====================

The inverse of script 5. Cluster the SAMPLES rather than the genes, then find
and plot the gene signature that defines each sample cluster.

Two problems that make sample clustering different from gene clustering
----------------------------------------------------------------------

Problem 1: raw sample correlations are nearly saturated and tell you almost
nothing.
    Correlate the samples of T2 against each other on log2 values and every pair
    comes back between 0.87 and 0.99. That is not because the samples are nearly
    identical, it is because a ribosomal gene is highly expressed in every
    sample and a quiet gene is quiet in every sample. That shared abundance
    profile dominates the correlation and drowns out the differences we actually
    care about.

    The fix is to centre each gene first, so what gets correlated is each
    sample's DEVIATION from that gene's average rather than its absolute level.
    The correlation matrix then spreads out across the full range and the group
    structure becomes obvious. Both versions are plotted side by side so the
    difference is visible rather than asserted.

Problem 2: with 9 samples, any clustering will look convincing.
    Nine points can be split into tidy looking groups by accident. Silhouette
    scores computed on 9 points are extremely noisy, and a dendrogram will always
    draw you a hierarchy whether or not one exists. So the primary evidence here
    is not the dendrogram, it is a bootstrap: resample the genes with replacement
    a thousand times, redo the sample clustering each time, and record how often
    each pair of samples ends up together. Pairs that co-cluster in 100% of
    replicates are real. Pairs that co-cluster 55% of the time are a coin flip
    being presented as a result.

    This matters concretely for T2. Eyeballing the correlation matrix suggests
    S1-S3 are clearly separate, but whether S4-S6 and S7-S9 are genuinely two
    groups or one is exactly the kind of borderline call that the bootstrap can
    settle and a dendrogram cannot.

Gene signatures
---------------
For each sample cluster we rank genes by how specifically they mark it, using the
signal-to-noise ratio (mean difference divided by the sum of the two standard
deviations). With three samples per group, an ordinary t-test is badly behaved
because a group that happens to have a tiny sample standard deviation produces an
enormous t statistic on no real evidence. 

Usage
-----
    # run from the repository root
    python task2_expression_clustering/scripts/06_cluster_samples.py \
        --matrix data/T2.txt --outdir task2_expression_clustering/results \
        --gene-clusters task2_expression_clustering/results/gene_clusters.tsv
"""

import argparse
import os
import sys
import time
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_common

RANDOM_SEED = 20260726


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_variable_genes(log_values, n_top):
    """Indices of the `n_top` most variable genes.

    Why restrict the gene set at all? Because a gene that does not change between
    samples contributes nothing but measurement noise to the distance between
    those samples. With around 9,700 genes of which the median has a coefficient
    of variation of only 0.16, including everything means the sample distances are
    mostly an average of noise. Taking the most variable genes concentrates the
    signal.

    The obvious objection is that selecting on variance and then clustering is
    circular. It is worth being clear about the limits: this selection is blind to
    any grouping of the samples, it only asks whether a gene moves at all, so it
    cannot manufacture a specific group structure. It can however make whatever
    structure exists look cleaner than it is, which is precisely why the bootstrap
    below resamples genes rather than trusting a single gene set.
    """
    variance = log_values.var(axis=1, ddof=1)
    return np.argsort(variance)[::-1][:n_top]


# ---------------------------------------------------------------------------
# Distances between samples
# ---------------------------------------------------------------------------

def sample_correlation(matrix, centre_genes):
    """Sample by sample Pearson correlation matrix.

    `centre_genes` toggles the fix described in the module docstring. With it off
    you get the saturated, uninformative version; with it on you get the version
    that actually resolves the groups.
    """
    working = matrix.copy()
    if centre_genes:
        working = working - working.mean(axis=1, keepdims=True)
    return np.corrcoef(working.T)


def correlation_distance(correlation):
    """Convert a correlation matrix into a distance matrix, as 1 - r.
    """
    return np.clip(1.0 - correlation, 0.0, None)


# ---------------------------------------------------------------------------
# How many sample clusters, and are they real
# ---------------------------------------------------------------------------

def bootstrap_coassignment(z_variable, k, n_bootstrap=1000, seed=RANDOM_SEED):
    """How often does each pair of samples cluster together under gene resampling?

    Each replicate draws a fresh set of genes with replacement, recomputes the
    sample clustering from scratch, and records which samples share a cluster.
    The result is a matrix of co-assignment frequencies between 0 and 1.
    """
    rng = np.random.default_rng(seed)
    n_genes, n_samples = z_variable.shape
    counts = np.zeros((n_samples, n_samples))

    for replicate in range(n_bootstrap):
        picked = rng.integers(0, n_genes, n_genes)
        resampled = z_variable[picked]
        # Re-centre genes after resampling so the distance means the same thing
        # as it does in the main analysis.
        resampled = resampled - resampled.mean(axis=1, keepdims=True)
        correlation = np.corrcoef(resampled.T)
        # A resampled gene set can occasionally produce a NaN column if a gene
        # got drawn in a degenerate way. Fall back to skipping the replicate
        # rather than letting NaN poison the average.
        if not np.isfinite(correlation).all():
            continue
        distance = squareform(correlation_distance(correlation), checks=False)
        labels = fcluster(linkage(distance, method="average"),
                          t=k, criterion="maxclust")
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    counts[i, j] += 1

    return counts / max(1, n_bootstrap)


def evaluate_sample_k(z_variable, distance_square, k_values):
    """Silhouette per candidate k, plus how balanced the resulting groups are."""
    condensed = squareform(distance_square, checks=False)
    link = linkage(condensed, method="average")
    rows = []
    for k in k_values:
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
            rows.append((k, float("nan"), Counter(labels)))
            continue
        score = silhouette_score(distance_square, labels, metric="precomputed")
        rows.append((k, float(score), Counter(labels)))
    return rows, link


# ---------------------------------------------------------------------------
# Gene signatures for each sample cluster
# ---------------------------------------------------------------------------

def signature_genes(log_values, gene_names, labels, n_top=25):
    """Rank genes by how specifically they mark each sample cluster.
    """
    epsilon = 0.1        
    signatures = {}

    for cluster_id in sorted(set(labels)):
        inside = labels == cluster_id
        outside = ~inside
        if inside.sum() == 0 or outside.sum() == 0:
            continue

        mean_in = log_values[:, inside].mean(axis=1)
        mean_out = log_values[:, outside].mean(axis=1)
        sd_in = log_values[:, inside].std(axis=1, ddof=1) \
            if inside.sum() > 1 else np.zeros(log_values.shape[0])
        sd_out = log_values[:, outside].std(axis=1, ddof=1) \
            if outside.sum() > 1 else np.zeros(log_values.shape[0])

        log2_fold_change = mean_in - mean_out
        snr = log2_fold_change / (sd_in + sd_out + epsilon)

        order_up = np.argsort(snr)[::-1][:n_top]
        order_down = np.argsort(snr)[:n_top]

        signatures[cluster_id] = {
            "up": [(gene_names[i], float(snr[i]), float(log2_fold_change[i]),
                    float(mean_in[i]), float(mean_out[i])) for i in order_up],
            "down": [(gene_names[i], float(snr[i]), float(log2_fold_change[i]),
                      float(mean_in[i]), float(mean_out[i])) for i in order_down],
            "snr": snr,
            "log2fc": log2_fold_change,
        }
    return signatures


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_correlation_comparison(log_all, log_variable, sample_names, path):
    """The saturated versus the gene-centred correlation matrix, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    panels = [
        (sample_correlation(log_all, centre_genes=False),
         "Naive: all genes, NOT gene-centred\n"
         "everything correlates, structure hidden"),
        (sample_correlation(log_variable, centre_genes=False),
         "Variable genes, NOT gene-centred\n"
         "still dominated by shared abundance"),
        (sample_correlation(log_variable, centre_genes=True),
         "Variable genes, gene-centred\n"
         "THIS is the one that resolves groups"),
    ]

    for axis, (correlation, title) in zip(axes, panels):
        low = float(correlation[~np.eye(len(correlation), dtype=bool)].min())
        image = axis.imshow(correlation, cmap="RdYlBu_r", vmin=low, vmax=1.0)
        axis.set_xticks(range(len(sample_names)))
        axis.set_xticklabels(sample_names, fontsize=8)
        axis.set_yticks(range(len(sample_names)))
        axis.set_yticklabels(sample_names, fontsize=8)
        axis.set_title("{}\noff-diagonal range {:.3f} to {:.3f}".format(
            title, low,
            float(correlation[~np.eye(len(correlation), dtype=bool)].max())),
            fontsize=9)
        for i in range(len(sample_names)):
            for j in range(len(sample_names)):
                axis.text(j, i, "{:.2f}".format(correlation[i, j]),
                          ha="center", va="center", fontsize=6,
                          color="black")
        fig.colorbar(image, ax=axis, shrink=0.75)

    fig.suptitle("Why gene-centring matters before comparing samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_sample_structure(link, correlation, coassignment, sample_names,
                            labels, silhouette_rows, path):
    """Dendrogram, correlation heatmap, bootstrap support and k diagnostics."""
    fig = plt.figure(figsize=(16, 9))

    # --- dendrogram ---
    axis = fig.add_subplot(2, 2, 1)
    dendrogram(link, labels=list(sample_names), ax=axis,
               color_threshold=None, above_threshold_color="#7f8c8d")
    axis.set_title("Hierarchical clustering of samples\n"
                   "(average linkage on 1 - Pearson r, gene-centred)",
                   fontsize=10)
    axis.set_ylabel("1 - r")

    # --- correlation heatmap in dendrogram order ---
    axis = fig.add_subplot(2, 2, 2)
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(link)
    reordered = correlation[np.ix_(order, order)]
    image = axis.imshow(reordered, cmap="RdYlBu_r")
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([sample_names[i] for i in order], fontsize=8)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([sample_names[i] for i in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            axis.text(j, i, "{:.2f}".format(reordered[i, j]), ha="center",
                      va="center", fontsize=6)
    axis.set_title("Sample correlation, clustered order", fontsize=10)
    fig.colorbar(image, ax=axis, shrink=0.8)

    # --- bootstrap co-assignment: the real confidence measure ---
    axis = fig.add_subplot(2, 2, 3)
    reordered_support = coassignment[np.ix_(order, order)]
    image = axis.imshow(reordered_support, cmap="Greens", vmin=0, vmax=1)
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([sample_names[i] for i in order], fontsize=8)
    axis.set_yticks(range(len(order)))
    axis.set_yticklabels([sample_names[i] for i in order], fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            value = reordered_support[i, j]
            axis.text(j, i, "{:.0f}".format(100 * value), ha="center",
                      va="center", fontsize=6,
                      color="white" if value > 0.6 else "black")
    axis.set_title("Bootstrap co-assignment, % of 1000 gene resamples\n"
                   "(100 = always together, ~30 = coin flip)", fontsize=10)
    fig.colorbar(image, ax=axis, shrink=0.8)

    # --- silhouette across k ---
    axis = fig.add_subplot(2, 2, 4)
    k_list = [row[0] for row in silhouette_rows]
    scores = [row[1] for row in silhouette_rows]
    axis.plot(k_list, scores, "o-", color="#8e44ad")
    axis.set_xticks(k_list)
    axis.set_xlabel("number of sample clusters (k)")
    axis.set_ylabel("silhouette (precomputed distance)")
    axis.set_title("Silhouette across k\n"
                   "treat with caution, only 9 samples", fontsize=10)
    axis.grid(alpha=0.3)
    for k, score, sizes in silhouette_rows:
        if np.isfinite(score):
            axis.annotate("{}".format(sorted(sizes.values(), reverse=True)),
                          (k, score), textcoords="offset points",
                          xytext=(0, 8), ha="center", fontsize=7)

    fig.suptitle("Sample clustering: structure and how much to trust it",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_sample_pca(z_variable, sample_names, labels, path):
    """PCA and the variance each component explains."""
    # Genes are the features here, so we transpose: 9 observations, many features.
    centred = z_variable - z_variable.mean(axis=1, keepdims=True)
    pca = PCA(n_components=min(len(sample_names), 5), random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(centred.T)
    variance = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    palette = plt.cm.Set1(np.linspace(0, 1, max(3, len(set(labels)))))

    for axis, (a, b) in zip(axes[:2], [(0, 1), (1, 2)]):
        for cluster_id in sorted(set(labels)):
            members = labels == cluster_id
            axis.scatter(coordinates[members, a], coordinates[members, b],
                         s=180, color=palette[cluster_id - 1],
                         edgecolor="black", zorder=3,
                         label="cluster {}".format(cluster_id))
        for index, name in enumerate(sample_names):
            axis.annotate(name, (coordinates[index, a], coordinates[index, b]),
                          textcoords="offset points", xytext=(8, 4), fontsize=9)
        axis.set_xlabel("PC{} ({:.1f}%)".format(a + 1, 100 * variance[a]))
        axis.set_ylabel("PC{} ({:.1f}%)".format(b + 1, 100 * variance[b]))
        axis.grid(alpha=0.3)
        axis.axhline(0, color="grey", lw=0.6)
        axis.axvline(0, color="grey", lw=0.6)
    axes[0].legend(fontsize=8)

    axes[2].bar(range(1, len(variance) + 1), 100 * variance, color="#2980b9")
    axes[2].plot(range(1, len(variance) + 1), 100 * np.cumsum(variance),
                 "o-", color="#c0392b", label="cumulative")
    axes[2].set_xlabel("principal component")
    axes[2].set_ylabel("% of variance")
    axes[2].set_title("Variance explained", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("Samples in PCA space (most variable genes, gene-centred)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_signature_heatmap(log_values, gene_names, labels, signatures,
                             sample_names, n_show, path):
    """Heatmap of the top signature genes for every sample cluster.
    """
    lookup = {name: index for index, name in enumerate(gene_names)}
    rows, row_labels, block_sizes, block_names = [], [], [], []

    for cluster_id in sorted(signatures.keys()):
        chosen = signatures[cluster_id]["up"][:n_show]
        for gene, snr, log2fc, _, _ in chosen:
            rows.append(lookup[gene])
            row_labels.append("{}  (SNR {:.1f}, {:+.1f} log2)".format(
                gene, snr, log2fc))
        block_sizes.append(len(chosen))
        block_names.append("cluster {}".format(cluster_id))

    matrix = log_values[rows]
    matrix = (matrix - matrix.mean(axis=1, keepdims=True)) / \
        (matrix.std(axis=1, ddof=0, keepdims=True) + 1e-9)

    # Order the columns so samples of the same cluster sit together.
    column_order = np.argsort(labels, kind="stable")

    height = max(6.0, 0.22 * len(rows) + 2.0)
    fig, axis = plt.subplots(figsize=(9, height))
    limit = float(np.percentile(np.abs(matrix), 99))
    image = axis.imshow(matrix[:, column_order], aspect="auto", cmap="RdBu_r",
                        vmin=-limit, vmax=limit, interpolation="nearest")

    axis.set_xticks(range(len(column_order)))
    axis.set_xticklabels(["{}\n(c{})".format(sample_names[i], labels[i])
                          for i in column_order], fontsize=8)
    axis.set_yticks(range(len(row_labels)))
    axis.set_yticklabels(row_labels, fontsize=6)

    cumulative = 0
    for size, name in zip(block_sizes, block_names):
        cumulative += size
        if cumulative < len(rows):
            axis.axhline(cumulative - 0.5, color="black", lw=1.2)
        axis.text(len(column_order) - 0.4, cumulative - size / 2.0 - 0.5,
                  name, rotation=270, va="center", ha="left", fontsize=8)

    # Vertical separators between sample clusters.
    previous = labels[column_order[0]]
    for position, index in enumerate(column_order):
        if labels[index] != previous:
            axis.axvline(position - 0.5, color="black", lw=1.5)
            previous = labels[index]

    axis.set_title("Gene signature for each sample cluster\n"
                   "top {} markers per cluster by signal-to-noise "
                   "ratio".format(n_show), fontsize=11)
    fig.colorbar(image, ax=axis, shrink=0.3, label="row z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_gene_cluster_bridge(gene_cluster_path, log_values, gene_names,
                               labels, sample_names, path):
    """Connect the two halves of the task.

    Takes the gene clusters found by script 5 and shows the mean profile of each
    one across the sample clusters found here. This is what ties the two analyses
    together: it names which gene programmes are responsible for separating which
    groups of samples, which neither analysis states on its own.
    """
    assignments = {}
    with open(gene_cluster_path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        column = {name: index for index, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            assignments[fields[column["gene"]]] = int(fields[column["cluster"]])

    gene_cluster_ids = np.array([assignments.get(name, -1)
                                 for name in gene_names])
    valid = gene_cluster_ids >= 0
    if not valid.any():
        return False

    z = (log_values - log_values.mean(axis=1, keepdims=True)) / \
        (log_values.std(axis=1, ddof=0, keepdims=True) + 1e-9)

    unique_gene_clusters = sorted(set(gene_cluster_ids[valid].tolist()))
    unique_sample_clusters = sorted(set(labels.tolist()))

    matrix = np.zeros((len(unique_gene_clusters), len(unique_sample_clusters)))
    for row, gene_cluster in enumerate(unique_gene_clusters):
        member_genes = gene_cluster_ids == gene_cluster
        for column_index, sample_cluster in enumerate(unique_sample_clusters):
            member_samples = labels == sample_cluster
            matrix[row, column_index] = z[np.ix_(member_genes, member_samples)].mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"width_ratios": [1, 1.4]})

    limit = float(np.abs(matrix).max())
    image = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].set_xticks(range(len(unique_sample_clusters)))
    axes[0].set_xticklabels(["sample\ncluster {}".format(c)
                             for c in unique_sample_clusters], fontsize=9)
    axes[0].set_yticks(range(len(unique_gene_clusters)))
    axes[0].set_yticklabels(["gene C{}\n(n={:,})".format(
        c, int((gene_cluster_ids == c).sum())) for c in unique_gene_clusters],
        fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[0].text(j, i, "{:+.2f}".format(matrix[i, j]), ha="center",
                         va="center", fontsize=8,
                         color="white" if abs(matrix[i, j]) > 0.6 * limit
                         else "black")
    axes[0].set_title("Mean z-score of each GENE cluster\n"
                      "within each SAMPLE cluster", fontsize=11)
    fig.colorbar(image, ax=axes[0], shrink=0.7, label="mean z")

    x = np.arange(len(sample_names))
    colours = plt.cm.tab20(np.linspace(0, 1, len(unique_gene_clusters)))
    for colour, gene_cluster in zip(colours, unique_gene_clusters):
        member_genes = gene_cluster_ids == gene_cluster
        axes[1].plot(x, z[member_genes].mean(axis=0), "o-", color=colour,
                     label="gene C{} (n={:,})".format(
                         gene_cluster, int(member_genes.sum())))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["{}\nc{}".format(name, labels[index])
                             for index, name in enumerate(sample_names)],
                            fontsize=8)
    axes[1].axhline(0, color="black", lw=0.6, ls=":")
    axes[1].set_ylabel("mean z-score")
    axes[1].set_title("Gene cluster profiles, samples annotated with\n"
                      "their sample cluster", fontsize=11)
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Bridging the two analyses: which gene programmes separate "
                 "which samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster samples and plot the gene signature of each cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--normalize", choices=["median", "quantile", "none"],
                        default="median")
    parser.add_argument("--n-variable-genes", type=int, default=2000,
                        help="How many of the most variable genes to use for the "
                             "sample distances.")
    parser.add_argument("--k", default="auto",
                        help="Number of sample clusters, or 'auto'.")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Gene resampling replicates for co-assignment support.")
    parser.add_argument("--n-signature-genes", type=int, default=20,
                        help="Markers per sample cluster to plot.")
    parser.add_argument("--gene-clusters", default=None,
                        help="Optional gene_clusters.tsv from script 05, used to "
                             "draw the figure that links the two analyses.")
    args = parser.parse_args()

    figures_dir = os.path.join(args.outdir, "figures")
    for directory in (args.outdir, figures_dir):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    started = time.time()
    report = []

    def emit(line=""):
        report.append(line)

    sys.stderr.write("Loading and preprocessing\n")
    data = t2_common.preprocess(args.matrix, normalize=args.normalize)
    log_values = data["log"]
    sample_names = data["sample_names"]
    gene_names = data["gene_names"]

    emit("=" * 76)
    emit("Task 2b: clustering samples and finding their gene signatures")
    emit("=" * 76)
    emit("Genes after filtering : {:,}".format(log_values.shape[0]))
    emit("Samples               : {}".format(", ".join(sample_names)))
    emit("Normalization         : {}".format(data["normalization_method"]))
    emit("")

    # ---- feature selection ------------------------------------------------
    variable_index = select_variable_genes(log_values, args.n_variable_genes)
    log_variable = log_values[variable_index]
    variance = log_values.var(axis=1, ddof=1)

    emit("Gene selection for the sample distances")
    emit("-" * 76)
    emit("  A gene that does not move between samples contributes only noise to")
    emit("  the distance between them, so the distances are computed on the most")
    emit("  variable genes. The selection is blind to any sample grouping, it")
    emit("  only asks whether a gene varies at all.")
    emit("")
    emit("  most variable genes used : {:,} of {:,}".format(
        len(variable_index), log_values.shape[0]))
    emit("  log2 variance cut-off    : {:.4f}".format(
        float(variance[variable_index].min())))
    emit("  median log2 variance, selected vs all : {:.4f} vs {:.4f}".format(
        float(np.median(variance[variable_index])), float(np.median(variance))))
    emit("")

    # ---- correlation structure -------------------------------------------
    naive = sample_correlation(log_values, centre_genes=False)
    centred = sample_correlation(log_variable, centre_genes=True)
    off_diagonal = ~np.eye(len(sample_names), dtype=bool)

    emit("Why the raw correlation matrix is not enough")
    emit("-" * 76)
    emit("  all genes, not gene-centred    : off-diagonal r from {:.4f} to {:.4f}"
         "  (range {:.4f})".format(naive[off_diagonal].min(),
                                   naive[off_diagonal].max(),
                                   naive[off_diagonal].max() - naive[off_diagonal].min()))
    emit("  variable genes, gene-centred   : off-diagonal r from {:.4f} to {:.4f}"
         "  (range {:.4f})".format(centred[off_diagonal].min(),
                                   centred[off_diagonal].max(),
                                   centred[off_diagonal].max() - centred[off_diagonal].min()))
    emit("  The second spreads the samples over a {:.1f}x wider range, which is".format(
        (centred[off_diagonal].max() - centred[off_diagonal].min()) /
        max(1e-9, naive[off_diagonal].max() - naive[off_diagonal].min())))
    emit("  the difference between resolving the groups and not.")
    emit("")

    figure_correlation_comparison(
        log_values, log_variable, sample_names,
        os.path.join(figures_dir, "07_sample_correlation_comparison.png"))

    # ---- how many sample clusters ----------------------------------------
    distance_square = correlation_distance(centred)
    k_candidates = list(range(2, min(6, len(sample_names))))
    silhouette_rows, link = evaluate_sample_k(
        log_variable, distance_square, k_candidates)

    emit("How many sample clusters")
    emit("-" * 76)
    emit("  {:>4} {:>14} {:>22}".format("k", "silhouette", "cluster sizes"))
    for k, score, sizes in silhouette_rows:
        emit("  {:>4} {:>14} {:>22}".format(
            k, "{:.4f}".format(score) if np.isfinite(score) else "n/a",
            str(sorted(sizes.values(), reverse=True))))
    emit("")

    best_by_silhouette = max(
        (row for row in silhouette_rows if np.isfinite(row[1])),
        key=lambda row: row[1])[0]
    chosen_k = best_by_silhouette if args.k == "auto" else int(args.k)
    labels = fcluster(link, t=chosen_k, criterion="maxclust")

    emit("  k chosen by silhouette : {}".format(best_by_silhouette))
    emit("  k used                 : {}{}".format(
        chosen_k, "" if args.k == "auto" else " (forced via --k)"))
    emit("")
    emit("  assignments:")
    for cluster_id in sorted(set(labels)):
        members = [sample_names[i] for i in range(len(labels))
                   if labels[i] == cluster_id]
        emit("    cluster {} : {}".format(cluster_id, ", ".join(members)))
    emit("")

    # ---- bootstrap support ------------------------------------------------
    sys.stderr.write("Bootstrapping sample clustering over {} gene resamples\n"
                     .format(args.bootstrap))
    z_variable = log_variable
    coassignment = bootstrap_coassignment(z_variable, chosen_k,
                                          n_bootstrap=args.bootstrap)

    emit("Bootstrap support: how often each pair of samples clusters together")
    emit("-" * 76)
    emit("  {} replicates, each resampling the {:,} genes with replacement and".format(
        args.bootstrap, len(variable_index)))
    emit("  redoing the clustering from scratch. This, not the dendrogram, is the")
    emit("  evidence for whether a grouping is real.")
    emit("")
    emit("        " + "".join("{:>7}".format(s) for s in sample_names))
    for i, name in enumerate(sample_names):
        emit("  {:<6}".format(name) +
             "".join("{:>7.0f}".format(100 * coassignment[i, j])
                     for j in range(len(sample_names))))
    emit("")

    # Within-cluster and between-cluster support, which is the summary that
    # actually answers "should I believe these groups".
    within, between = [], []
    for i in range(len(sample_names)):
        for j in range(i + 1, len(sample_names)):
            if labels[i] == labels[j]:
                within.append(coassignment[i, j])
            else:
                between.append(coassignment[i, j])
    emit("  mean support for pairs placed TOGETHER  : {:.1f}%  (min {:.1f}%)".format(
        100 * np.mean(within), 100 * np.min(within)))
    emit("  mean support for pairs placed APART     : {:.1f}%  (max {:.1f}%)".format(
        100 * np.mean(between), 100 * np.max(between)))
    emit("")
    weak = [(sample_names[i], sample_names[j], coassignment[i, j])
            for i in range(len(sample_names))
            for j in range(i + 1, len(sample_names))
            if labels[i] == labels[j] and coassignment[i, j] < 0.9]
    if weak:
        emit("  pairs grouped together on weak support (under 90%):")
        for a, b, support in weak:
            emit("    {} and {} : {:.1f}%".format(a, b, 100 * support))
    else:
        emit("  every pair placed together is supported in at least 90% of")
        emit("  replicates, so the partition is not an artefact of the gene set.")
    emit("")

    figure_sample_structure(link, centred, coassignment, sample_names, labels,
                            silhouette_rows,
                            os.path.join(figures_dir, "08_sample_structure.png"))
    figure_sample_pca(log_variable, sample_names, labels,
                      os.path.join(figures_dir, "09_sample_pca.png"))

    # ---- gene signatures --------------------------------------------------
    sys.stderr.write("Deriving gene signatures per sample cluster\n")
    signatures = signature_genes(log_values, gene_names, labels,
                                 n_top=max(50, args.n_signature_genes))

    emit("Gene signatures per sample cluster")
    emit("-" * 76)
    emit("  Ranked by signal-to-noise ratio, (mean_in - mean_out) divided by the")
    emit("  sum of the two standard deviations. Preferred over a t statistic")
    emit("  because with 3 samples per group a t test rewards a group that")
    emit("  happens to have a very small spread, which is not evidence.")
    emit("")
    for cluster_id in sorted(signatures.keys()):
        members = [sample_names[i] for i in range(len(labels))
                   if labels[i] == cluster_id]
        emit("  Cluster {} ({})".format(cluster_id, ", ".join(members)))
        emit("    top {} UP markers".format(args.n_signature_genes))
        emit("      {:<10} {:>8} {:>10} {:>10} {:>10}".format(
            "gene", "SNR", "log2FC", "mean in", "mean out"))
        for gene, snr, log2fc, mean_in, mean_out in \
                signatures[cluster_id]["up"][:args.n_signature_genes]:
            emit("      {:<10} {:>8.2f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                gene, snr, log2fc, mean_in, mean_out))
        emit("    top 5 DOWN markers")
        for gene, snr, log2fc, mean_in, mean_out in \
                signatures[cluster_id]["down"][:5]:
            emit("      {:<10} {:>8.2f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                gene, snr, log2fc, mean_in, mean_out))
        emit("")

    figure_signature_heatmap(
        log_values, gene_names, labels, signatures, sample_names,
        args.n_signature_genes,
        os.path.join(figures_dir, "10_sample_cluster_signatures.png"))

    # ---- bridge to the gene clustering ------------------------------------
    if args.gene_clusters and os.path.isfile(args.gene_clusters):
        sys.stderr.write("Drawing the bridge to the gene clustering\n")
        drawn = figure_gene_cluster_bridge(
            args.gene_clusters, log_values, gene_names, labels, sample_names,
            os.path.join(figures_dir, "11_gene_and_sample_clusters.png"))
        if drawn:
            emit("Link to the gene clustering")
            emit("-" * 76)
            emit("  figures/11_gene_and_sample_clusters.png shows the mean z-score")
            emit("  of every gene cluster from script 05 within every sample")
            emit("  cluster found here, which names the gene programmes")
            emit("  responsible for separating the sample groups.")
            emit("")

    # ---- tables -----------------------------------------------------------
    assignments_path = os.path.join(args.outdir, "sample_clusters.tsv")
    with open(assignments_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("sample\tcluster\tmean_bootstrap_support_within_cluster\n")
        for index, name in enumerate(sample_names):
            same = [coassignment[index, j] for j in range(len(sample_names))
                    if j != index and labels[j] == labels[index]]
            out.write("{}\t{}\t{:.4f}\n".format(
                name, labels[index], float(np.mean(same)) if same else float("nan")))

    signatures_path = os.path.join(args.outdir, "sample_cluster_signatures.tsv")
    with open(signatures_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("sample_cluster\tdirection\trank\tgene\tsnr\tlog2_fold_change\t"
                  "mean_log2_in_cluster\tmean_log2_outside\n")
        for cluster_id in sorted(signatures.keys()):
            for direction in ("up", "down"):
                for rank, (gene, snr, log2fc, mean_in, mean_out) in \
                        enumerate(signatures[cluster_id][direction], start=1):
                    out.write("{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                        cluster_id, direction, rank, gene, snr, log2fc,
                        mean_in, mean_out))

    support_path = os.path.join(args.outdir, "sample_bootstrap_support.tsv")
    with open(support_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("sample\t" + "\t".join(sample_names) + "\n")
        for index, name in enumerate(sample_names):
            out.write(name + "\t" + "\t".join(
                "{:.4f}".format(coassignment[index, j])
                for j in range(len(sample_names))) + "\n")

    emit("Files written")
    emit("-" * 76)
    for path in [assignments_path, signatures_path, support_path]:
        emit("  {}".format(path))
    for name in ["07_sample_correlation_comparison.png", "08_sample_structure.png",
                 "09_sample_pca.png", "10_sample_cluster_signatures.png"]:
        emit("  {}".format(os.path.join(figures_dir, name)))
    emit("")
    emit("Elapsed: {:.1f} s".format(time.time() - started))

    text = "\n".join(report)
    print(text)
    report_path = os.path.join(args.outdir, "sample_clustering_report.txt")
    with open(report_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(text + "\n")


if __name__ == "__main__":
    main()
