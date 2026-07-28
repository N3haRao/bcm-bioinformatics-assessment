#!/usr/bin/env python3
"""
05_cluster_genes.py
===================

Cluster genes by the shape of their expression across samples, then visualise a
representative profile for each cluster.

The three questions this script has to answer 
------------------------------------------------------

1. How many clusters?
   Picking k by looking at an elbow plot and guessing is the usual approach and
   it is not good enough, because the elbow in expression data is almost always
   smooth and you can talk yourself into any k you like. 

2. Is a cluster real, or is it amplified noise?
   Z-scoring is necessary (see t2_common.py) but it destroys amplitude. A gene
   that doubles and a gene that wobbles by 2% can end up with identical z
   profiles, so a tight looking cluster might be built entirely out of genes with
   no real signal. Every cluster is therefore reported with its true amplitude in
   log2 units, and there is a second figure showing the clusters on the real
   expression scale next to the z-scored one. A cluster whose mean amplitude is
   0.05 log2 units, a 3.5% swing, is noise regardless of how good its silhouette
   is.

3. Does the answer depend on the algorithm?
   k-means imposes roughly spherical, equal sized clusters. Ward hierarchical
   clustering makes different assumptions. We run both and report the adjusted
   Rand index between them. High agreement means the structure is in the data;
   low agreement means we are looking at an artefact of one method's assumptions.

Usage
-----
    # run from the repository root
    python task2_expression_clustering/scripts/05_cluster_genes.py \
        --matrix data/T2.txt --outdir task2_expression_clustering/results
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")       
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_samples,
                             silhouette_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_common

RANDOM_SEED = 20260726


# ---------------------------------------------------------------------------
# Choosing k
# ---------------------------------------------------------------------------

def evaluate_k_range(z, k_values, n_bootstrap=10, subsample_fraction=0.8,
                     silhouette_sample=4000, seed=RANDOM_SEED):
    """Score every candidate k on four indicators, including resampling stability.

    Returns a dict of lists, one entry per k, all aligned with `k_values`.
    """
    rng = np.random.default_rng(seed)
    n_genes = z.shape[0]

    results = {"k": [], "inertia": [], "silhouette": [], "calinski": [],
               "davies_bouldin": [], "stability": [], "stability_sd": []}

    for k in k_values:
        labels, centres, inertia = t2_common.kmeans(z, k, n_init=10, seed=seed)

        # Silhouette on the full 9,000 by 9 matrix means ~45 million pairwise
        # distances per k. Sub-sampling gives the same number to two decimals for
        # a fraction of the time, which matters because we are sweeping 14 values
        # of k and also bootstrapping each one.
        if n_genes > silhouette_sample:
            index = rng.choice(n_genes, silhouette_sample, replace=False)
            sil = silhouette_score(z[index], labels[index])
        else:
            sil = silhouette_score(z, labels)

        # --- stability by subsampling ---
        # Fit on a random 80% of genes, then use the resulting centroids to label
        # ALL genes. Two bootstrap replicates can then be compared directly on the
        # full gene set. Comparing only the shared genes would work too but this
        # way every pair is compared on the same footing.
        bootstrap_labels = []
        for replicate in range(n_bootstrap):
            subset = rng.choice(n_genes,
                                int(subsample_fraction * n_genes),
                                replace=False)
            _, sub_centres, _ = t2_common.kmeans(
                z[subset], k, n_init=5, seed=int(rng.integers(1 << 30)))
            bootstrap_labels.append(t2_common.assign_to_centroids(z, sub_centres))

        pairwise_ari = []
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                pairwise_ari.append(
                    adjusted_rand_score(bootstrap_labels[i], bootstrap_labels[j]))

        results["k"].append(k)
        results["inertia"].append(float(inertia))
        results["silhouette"].append(float(sil))
        results["calinski"].append(float(calinski_harabasz_score(z, labels)))
        results["davies_bouldin"].append(float(davies_bouldin_score(z, labels)))
        results["stability"].append(float(np.mean(pairwise_ari)))
        results["stability_sd"].append(float(np.std(pairwise_ari)))

        sys.stderr.write(
            "  k={:<3} silhouette={:.4f}  stability(ARI)={:.4f}+/-{:.4f}  "
            "CH={:.0f}\n".format(k, sil, results["stability"][-1],
                                 results["stability_sd"][-1],
                                 results["calinski"][-1]))
        sys.stderr.flush()

    return results


def choose_k(results, strict_floor=0.95, gate_floor=0.75):
    """Pick k.
    """
    stability = np.array(results["stability"])
    silhouette = np.array(results["silhouette"])
    k_values = np.array(results["k"])

    gated = stability >= gate_floor
    if gated.any():
        candidates = np.where(gated)[0]
        most_separated = int(k_values[candidates[np.argmax(silhouette[candidates])]])
        gate_passed = True
    else:
        most_separated = int(k_values[int(np.argmax(stability))])
        gate_passed = False

    strict = np.where(stability >= strict_floor)[0]
    if len(strict):
        most_granular = int(k_values[strict.max()])
    else:
        most_granular = most_separated

    return most_granular, most_separated, gate_passed


# ---------------------------------------------------------------------------
# Describing the clusters
# ---------------------------------------------------------------------------

def order_clusters_by_similarity(centroids):
    """Return an ordering of clusters so that similar profiles sit side by side.
    """
    if len(centroids) < 3:
        return list(range(len(centroids)))
    link = linkage(centroids, method="average", metric="correlation")
    return list(leaves_list(link))


def summarise_clusters(z, log_values, labels, sample_names, silhouette_values):
    """Per cluster statistics, including the amplitude check.

    The important columns are `mean_log2_range` and `fold_change`. They answer
    "how big is this pattern really", which the z-scored profile cannot tell you.
    """
    summaries = []
    for cluster_id in sorted(set(labels)):
        members = labels == cluster_id
        z_members = z[members]
        log_members = log_values[members]

        z_profile = z_members.mean(axis=0)
        log_profile = log_members.mean(axis=0)

        # Amplitude, measured per gene and then averaged, NOT measured on the
        # averaged profile. Averaging first would cancel out genes that move in
        # opposite directions and understate how much each gene really moves.
        per_gene_range = log_members.max(axis=1) - log_members.min(axis=1)

        # How coherent is the cluster? Correlation of each member against the
        # cluster centroid, which is a more intuitive measure than silhouette for
        # expression data.
        centroid = z_profile
        centred_members = z_members - z_members.mean(axis=1, keepdims=True)
        centred_centroid = centroid - centroid.mean()
        denominator = (np.linalg.norm(centred_members, axis=1) *
                       np.linalg.norm(centred_centroid))
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations = (centred_members @ centred_centroid) / denominator

        summaries.append({
            "cluster": cluster_id,
            "n_genes": int(members.sum()),
            "z_profile": z_profile,
            "log_profile": log_profile,
            "z_q25": np.percentile(z_members, 25, axis=0),
            "z_q75": np.percentile(z_members, 75, axis=0),
            "mean_log2_range": float(np.mean(per_gene_range)),
            "median_log2_range": float(np.median(per_gene_range)),
            "fold_change": float(2 ** np.mean(per_gene_range)),
            "mean_expression_log2": float(log_members.mean()),
            "mean_correlation_to_centroid": float(np.nanmean(correlations)),
            "mean_silhouette": float(silhouette_values[members].mean()),
            "peak_sample": sample_names[int(np.argmax(z_profile))],
            "trough_sample": sample_names[int(np.argmin(z_profile))],
        })
    return summaries


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_qc(data, path):
    """Show what the normalization step did, so it can be checked visually."""
    raw_log = np.log2(data["raw"] + 1)
    sample_names = data["sample_names"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].boxplot([raw_log[:, i] for i in range(raw_log.shape[1])],
                    labels=sample_names, showfliers=False)
    axes[0].set_title("Before: log2 expression per sample\n"
                      "(note S1-S3 sit lower)")
    axes[0].set_ylabel("log2(expression + 1)")

    axes[1].boxplot([data["log"][:, i] for i in range(data["log"].shape[1])],
                    labels=sample_names, showfliers=False)
    axes[1].set_title("After: {} normalization\n"
                      "(medians aligned)".format(data["normalization_method"]))
    axes[1].set_ylabel("log2(expression + 1), normalized")

    before = np.median(raw_log, axis=0)
    after = np.median(data["log"], axis=0)
    x = np.arange(len(sample_names))
    axes[2].plot(x, before, "o-", label="before", color="#c0392b")
    axes[2].plot(x, after, "s-", label="after", color="#27ae60")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sample_names)
    axes[2].set_ylabel("column median, log2")
    axes[2].set_title("Column medians\n"
                      "spread {:.3f} -> {:.3f} log2".format(
                          before.max() - before.min(), after.max() - after.min()))
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("Quality control: removing the per-sample global offset",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_k_selection(results, chosen_k, path):
    """Four panels of evidence for the choice of k."""
    k_values = results["k"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        (axes[0][0], "inertia", "Inertia (within-cluster sum of squares)",
         "lower is better, look for the elbow", "#2980b9"),
        (axes[0][1], "silhouette", "Silhouette score",
         "higher is better, max 1", "#8e44ad"),
        (axes[1][0], "calinski", "Calinski-Harabasz index",
         "higher is better", "#16a085"),
        (axes[1][1], "stability", "Resampling stability (adjusted Rand index)",
         "higher is better, this is the gate", "#d35400"),
    ]

    for axis, key, title, subtitle, colour in panels:
        axis.plot(k_values, results[key], "o-", color=colour)
        if key == "stability":
            axis.fill_between(
                k_values,
                np.array(results["stability"]) - np.array(results["stability_sd"]),
                np.array(results["stability"]) + np.array(results["stability_sd"]),
                alpha=0.2, color=colour)
            axis.axhline(0.75, ls="--", color="grey", lw=1)
            axis.text(k_values[-1], 0.76, "stability floor 0.75",
                      ha="right", va="bottom", fontsize=8, color="grey")
        axis.axvline(chosen_k, ls=":", color="black", lw=1.5)
        axis.set_title("{}\n{}".format(title, subtitle), fontsize=10)
        axis.set_xlabel("number of clusters (k)")
        axis.set_xticks(k_values)
        axis.grid(alpha=0.3)

    fig.suptitle("Choosing k: chosen k = {} (dotted line)".format(chosen_k),
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_cluster_profiles(summaries, order, sample_names, z, labels, path):
    """Small multiples: the representative z-scored profile of every cluster.

    Each panel shows a thin sample of individual gene traces in the background so
    the reader can see the spread rather than only a mean line, the interquartile
    ribbon, and the cluster mean on top.
    """
    n = len(order)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.9 * n_rows),
                             squeeze=False, sharex=True, sharey=True)
    x = np.arange(len(sample_names))
    rng = np.random.default_rng(RANDOM_SEED)

    for position, cluster_index in enumerate(order):
        summary = summaries[cluster_index]
        axis = axes[position // n_cols][position % n_cols]

        # A background sample of individual genes. Capping at 150 keeps the
        # figure legible and the file small; drawing 3,000 lines would just be a
        # solid block of colour that hides the mean.
        members = np.where(labels == summary["cluster"])[0]
        show = rng.choice(members, min(150, len(members)), replace=False)
        for gene_index in show:
            axis.plot(x, z[gene_index], color="#95a5a6", alpha=0.10, lw=0.6)

        axis.fill_between(x, summary["z_q25"], summary["z_q75"],
                          color="#3498db", alpha=0.35, label="IQR")
        axis.plot(x, summary["z_profile"], "o-", color="#c0392b", lw=2.2,
                  label="cluster mean")
        axis.axhline(0, color="black", lw=0.6, ls=":")

        axis.set_title(
            "Cluster {}   n = {:,}\n{:.2f} log2 range ({:.2f}x)   r={:.2f}".format(
                summary["cluster"], summary["n_genes"],
                summary["mean_log2_range"], summary["fold_change"],
                summary["mean_correlation_to_centroid"]),
            fontsize=9)
        axis.set_xticks(x)
        axis.set_xticklabels(sample_names, fontsize=8)
        if position % n_cols == 0:
            axis.set_ylabel("z-score")

    # Blank out any unused panels so there are no empty axes with stray ticks.
    for position in range(n, n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis("off")

    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Representative expression profile per gene cluster "
                 "(z-scored: pattern SHAPE)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_cluster_amplitude(summaries, order, sample_names, path):
    """The same clusters on the real log2 scale, so amplitude is visible.

    This is the companion to the z-scored figure and the reason both exist. Two
    clusters can look equally dramatic after z-scoring while one moves fourfold
    and the other moves by 3%. Only this view distinguishes them.
    """
    fig = plt.figure(figsize=(15, 6))
    grid = GridSpec(1, 3, width_ratios=[2.0, 1.0, 1.0], wspace=0.32)

    x = np.arange(len(sample_names))
    colours = plt.cm.tab20(np.linspace(0, 1, len(order)))

    # Panel 1: absolute expression level per cluster.
    axis = fig.add_subplot(grid[0])
    for colour, cluster_index in zip(colours, order):
        summary = summaries[cluster_index]
        axis.plot(x, summary["log_profile"], "o-", color=colour,
                  label="C{} (n={:,})".format(summary["cluster"],
                                              summary["n_genes"]))
    axis.set_xticks(x)
    axis.set_xticklabels(sample_names)
    axis.set_ylabel("mean log2(expression + 1), normalized")
    axis.set_title("Cluster profiles on the REAL expression scale\n"
                   "(vertical position = abundance, slope = actual change)",
                   fontsize=10)
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.3)

    # Panel 2: amplitude ranking, the noise-versus-signal check.
    axis = fig.add_subplot(grid[1])
    ordered = sorted(summaries, key=lambda s: s["mean_log2_range"])
    labels_text = ["C{}".format(s["cluster"]) for s in ordered]
    values = [s["mean_log2_range"] for s in ordered]
    bar_colours = ["#c0392b" if v < 0.5 else "#27ae60" for v in values]
    axis.barh(range(len(values)), values, color=bar_colours)
    axis.set_yticks(range(len(values)))
    axis.set_yticklabels(labels_text, fontsize=8)
    axis.axvline(0.5, ls="--", color="black", lw=1)
    axis.text(0.52, 0.3, "0.5 log2 = 1.41x", fontsize=7, rotation=90)
    axis.set_xlabel("mean per-gene log2 range")
    axis.set_title("Amplitude: is the pattern real?\n"
                   "red = under 1.41x, likely noise", fontsize=10)

    # Panel 3: cluster size.
    axis = fig.add_subplot(grid[2])
    ordered_size = sorted(summaries, key=lambda s: s["n_genes"])
    axis.barh(range(len(ordered_size)),
              [s["n_genes"] for s in ordered_size], color="#7f8c8d")
    axis.set_yticks(range(len(ordered_size)))
    axis.set_yticklabels(["C{}".format(s["cluster"]) for s in ordered_size],
                         fontsize=8)
    axis.set_xlabel("genes")
    axis.set_title("Cluster size", fontsize=10)

    fig.suptitle("Amplitude check: z-scoring hides how big each pattern is",
                 fontsize=13)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figure_heatmap(z, labels, order, sample_names, path):
    """Genes by samples heatmap with genes ordered and blocked by cluster."""
    # Build the row order: clusters in the display order, and within each cluster
    # sort genes by their own profile so the block has internal structure rather
    # than looking like static.
    row_order = []
    boundaries = []
    for cluster_index in order:
        members = np.where(labels == cluster_index)[0]
        if len(members) > 1:
            local_link = linkage(z[members], method="average",
                                 metric="correlation")
            members = members[leaves_list(local_link)]
        row_order.extend(members.tolist())
        boundaries.append(len(row_order))

    matrix = z[row_order]

    fig, axis = plt.subplots(figsize=(6.5, 11))
    limit = float(np.percentile(np.abs(matrix), 99))
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-limit, vmax=limit, interpolation="nearest")
    for boundary in boundaries[:-1]:
        axis.axhline(boundary, color="black", lw=0.8)

    # Label each block at its midpoint.
    previous = 0
    tick_positions, tick_labels = [], []
    for boundary, cluster_index in zip(boundaries, order):
        tick_positions.append((previous + boundary) / 2.0)
        tick_labels.append("C{} ({:,})".format(cluster_index,
                                               boundary - previous))
        previous = boundary
    axis.set_yticks(tick_positions)
    axis.set_yticklabels(tick_labels, fontsize=8)

    axis.set_xticks(np.arange(len(sample_names)))
    axis.set_xticklabels(sample_names, rotation=0)
    axis.set_title("All {:,} genes, z-scored, grouped by cluster".format(
        matrix.shape[0]), fontsize=11)
    fig.colorbar(image, ax=axis, shrink=0.4, label="z-score")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure_gene_pca(z, labels, order, path):
    """PCA of the genes, coloured by cluster.

    A sanity check on the clustering rather than a result in itself. If the
    clusters form coherent territories in the first two components, the partition
    is describing the dominant structure. If they are shuffled together, k-means
    has cut across the grain of the data.
    """
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(z)
    variance = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colours = plt.cm.tab20(np.linspace(0, 1, len(order)))
    lookup = {cluster_index: colour for cluster_index, colour in
              zip(order, colours)}

    for axis, (a, b) in zip(axes, [(0, 1), (0, 2)]):
        for cluster_index in order:
            members = labels == cluster_index
            axis.scatter(coordinates[members, a], coordinates[members, b],
                         s=3, alpha=0.45, color=lookup[cluster_index],
                         label="C{}".format(cluster_index), rasterized=True)
        axis.set_xlabel("PC{} ({:.1f}% of variance)".format(a + 1,
                                                           100 * variance[a]))
        axis.set_ylabel("PC{} ({:.1f}% of variance)".format(b + 1,
                                                           100 * variance[b]))
        axis.grid(alpha=0.3)
    axes[1].legend(fontsize=7, markerscale=3, ncol=2, loc="best")
    fig.suptitle("Genes in PCA space, coloured by cluster "
                 "(structure check, not a result)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster genes by expression pattern and plot cluster profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--matrix", required=True, help="Tab separated matrix.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--normalize", choices=["median", "quantile", "none"],
                        default="median",
                        help="Per-sample offset correction. See t2_common.py for "
                             "why 'none' is a bad idea on this dataset.")
    parser.add_argument("--k", default="auto",
                        help="Number of clusters, or 'auto' to select by the "
                             "stability-gated rule.")
    parser.add_argument("--k-max", type=int, default=15,
                        help="Largest k to evaluate during selection.")
    parser.add_argument("--bootstrap", type=int, default=10,
                        help="Resampling replicates per k for the stability score.")
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
    z = data["z"]
    sample_names = data["sample_names"]
    qc = data["qc"]

    emit("=" * 76)
    emit("Task 2a: clustering genes by expression pattern")
    emit("=" * 76)
    emit("Input matrix   : {}".format(os.path.abspath(args.matrix)))
    emit("Shape as read  : {:,} genes x {} samples".format(
        qc["n_genes"], qc["n_samples"]))
    emit("Samples        : {}".format(", ".join(sample_names)))
    emit("")
    emit("Quality control")
    emit("-" * 76)
    emit("  exact zeros in matrix        : {:,}".format(qc["n_zeros"]))
    emit("  genes zero in every sample   : {:,}".format(qc["n_all_zero_genes"]))
    emit("  value range                  : {:,.1f} to {:,.1f}".format(
        qc["value_min"], qc["value_max"]))
    emit("  top 10 genes' share of signal: {:.2f}%".format(100 * qc["top10_share"]))
    emit("  column sum CV                : {:.4f}".format(qc["column_sum_cv"]))
    emit("  log2 column median spread    : {:.4f} log2 units "
         "({:.1f}% linear)".format(qc["log_median_spread"],
                                   100 * (2 ** qc["log_median_spread"] - 1)))
    emit("  genes removed by filtering   : {:,} of {:,}".format(
        data["n_genes_removed"], data["n_genes_before_filter"]))
    if data["n_genes_removed"]:
        emit("    ({})".format(", ".join(data["removed_gene_names"][:12]) +
                               (" ..." if data["n_genes_removed"] > 12 else "")))
    emit("  genes carried into clustering: {:,}".format(z.shape[0]))
    emit("  normalization applied        : {}".format(data["normalization_method"]))
    emit("  per-sample offsets removed   : {}".format(
        ", ".join("{}={:+.4f}".format(s, o) for s, o
                  in zip(sample_names, data["normalization_offsets"]))))
    emit("")

    figure_qc(data, os.path.join(figures_dir, "01_qc_normalization.png"))

    # ---- choose k ---------------------------------------------------------
    sys.stderr.write("Evaluating k from 2 to {}\n".format(args.k_max))
    k_values = list(range(2, args.k_max + 1))
    selection = evaluate_k_range(z, k_values, n_bootstrap=args.bootstrap)

    auto_k, most_separated_k, gate_passed = choose_k(selection)
    if args.k == "auto":
        chosen_k = auto_k
    else:
        chosen_k = int(args.k)

    emit("Choosing the number of clusters")
    emit("-" * 76)
    emit("  Stability = adjusted Rand index between {} independent refits on".format(
        args.bootstrap))
    emit("  random 80% subsets of the genes. It measures whether the same genes")
    emit("  keep landing together, which is the property we actually need.")
    emit("")
    emit("  Two rules, both fixed in advance, because they answer different")
    emit("  questions:")
    emit("    most separated       highest silhouette among k with stability")
    emit("                         above 0.75. Answers 'what is the single")
    emit("                         cleanest split'. Beware: silhouette")
    emit("                         structurally favours small k, so on")
    emit("                         continuous data it usually lands on k=2.")
    emit("    most granular stable largest k whose stability still reaches 0.95.")
    emit("                         Answers 'how much structure can we resolve")
    emit("                         and still reproduce'. This is the default,")
    emit("                         because the task is to describe expression")
    emit("                         patterns rather than find one dichotomy.")
    emit("")
    emit("  {:>4} {:>12} {:>12} {:>12} {:>14} {:>10}".format(
        "k", "inertia", "silhouette", "Calinski-H", "stability ARI", "gate"))
    for index, k in enumerate(selection["k"]):
        emit("  {:>4} {:>12,.0f} {:>12.4f} {:>12,.0f} {:>8.4f}+/-{:.3f} {:>10}".format(
            k, selection["inertia"][index], selection["silhouette"][index],
            selection["calinski"][index], selection["stability"][index],
            selection["stability_sd"][index],
            "pass" if selection["stability"][index] >= 0.75 else "fail"))
    emit("")
    if not gate_passed:
        emit("  WARNING: no k reached the stability floor. Falling back to the")
        emit("  most stable k. Treat the partition as provisional.")
    emit("  most separated k        : {} (silhouette {:.4f})".format(
        most_separated_k,
        selection["silhouette"][selection["k"].index(most_separated_k)]))
    emit("  most granular stable k  : {} (stability {:.4f})".format(
        auto_k, selection["stability"][selection["k"].index(auto_k)]))
    emit("  k used                  : {}{}".format(
        chosen_k, "" if args.k == "auto" else " (forced via --k)"))
    emit("")
    emit("  Read the two together. k={} is the dominant axis of variation in".format(
        most_separated_k))
    emit("  this matrix and everything else is structure layered on top of it.")
    emit("  k={} is as fine as the data can be cut while still reproducing".format(
        auto_k))
    emit("  under resampling.")
    emit("")

    figure_k_selection(selection, chosen_k,
                       os.path.join(figures_dir, "02_k_selection.png"))

    # ---- final clustering -------------------------------------------------
    sys.stderr.write("Final k-means with k={}\n".format(chosen_k))
    # More restarts than during the sweep, since this is the answer we keep.
    labels, final_centres, final_inertia = t2_common.kmeans(
        z, chosen_k, n_init=50, seed=RANDOM_SEED)

    # Cross-check with a method that makes different assumptions. Ward minimises
    # within-cluster variance in a hierarchy rather than around k moving centres,
    # so agreement between the two is meaningful evidence.
    sys.stderr.write("Cross-checking against Ward hierarchical clustering\n")
    ward_link = linkage(z, method="ward")
    ward_labels = fcluster(ward_link, t=chosen_k, criterion="maxclust")
    agreement = adjusted_rand_score(labels, ward_labels)

    silhouette_values = silhouette_samples(z, labels)
    summaries = summarise_clusters(z, data["log"], labels, sample_names,
                                   silhouette_values)
    centroids = np.array([s["z_profile"] for s in summaries])
    order = order_clusters_by_similarity(centroids)

    emit("Cluster results (k = {})".format(chosen_k))
    emit("-" * 76)
    emit("  Agreement with Ward hierarchical clustering (ARI): {:.4f}".format(
        agreement))
    emit("  Overall silhouette: {:.4f}".format(float(silhouette_values.mean())))
    emit("")
    emit("  {:>3} {:>7} {:>9} {:>8} {:>7} {:>7} {:>6} {:>6}".format(
        "C", "genes", "log2 rng", "fold", "r-cent", "silh", "peak", "trough"))
    for cluster_index in order:
        s = summaries[cluster_index]
        emit("  {:>3} {:>7,} {:>9.3f} {:>7.2f}x {:>7.3f} {:>7.3f} {:>6} {:>6}".format(
            s["cluster"], s["n_genes"], s["mean_log2_range"], s["fold_change"],
            s["mean_correlation_to_centroid"], s["mean_silhouette"],
            s["peak_sample"], s["trough_sample"]))
    emit("")
    emit("  Column meanings:")
    emit("    log2 rng  mean per-gene log2 range across samples (the amplitude)")
    emit("    fold      the same number as a linear fold change")
    emit("    r-cent    mean Pearson correlation of members to cluster centroid")
    emit("    silh      mean silhouette, how well separated from other clusters")
    emit("")

    low_amplitude = [s for s in summaries if s["mean_log2_range"] < 0.5]
    emit("  Amplitude verdict")
    emit("  " + "-" * 74)
    emit("    Z-scoring makes every cluster look equally dramatic, so amplitude")
    emit("    has to be checked separately. Using 0.5 log2 (1.41x) as the line")
    emit("    below which a pattern is not worth interpreting:")
    emit("      clusters above the line: {} ({:,} genes)".format(
        chosen_k - len(low_amplitude),
        sum(s["n_genes"] for s in summaries if s["mean_log2_range"] >= 0.5)))
    emit("      clusters below the line: {} ({:,} genes)".format(
        len(low_amplitude), sum(s["n_genes"] for s in low_amplitude)))
    if low_amplitude:
        emit("      low amplitude clusters: {}".format(
            ", ".join("C{} ({:.2f} log2, {:.2f}x)".format(
                s["cluster"], s["mean_log2_range"], s["fold_change"])
                for s in low_amplitude)))
    emit("")

    # ---- figures ----------------------------------------------------------
    sys.stderr.write("Drawing figures\n")
    figure_cluster_profiles(summaries, order, sample_names, z, labels,
                            os.path.join(figures_dir,
                                         "03_gene_cluster_profiles.png"))
    figure_cluster_amplitude(summaries, order, sample_names,
                             os.path.join(figures_dir,
                                          "04_gene_cluster_amplitude.png"))
    figure_heatmap(z, labels, order, sample_names,
                   os.path.join(figures_dir, "05_gene_cluster_heatmap.png"))
    figure_gene_pca(z, labels, order,
                    os.path.join(figures_dir, "06_gene_pca.png"))

    # ---- tables -----------------------------------------------------------
    assignments_path = os.path.join(args.outdir, "gene_clusters.tsv")
    with open(assignments_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("gene\tcluster\tsilhouette\tmean_log2_expression\t"
                  "log2_range\tcorrelation_to_centroid\t"
                  + "\t".join("z_" + s for s in sample_names) + "\t"
                  + "\t".join("log2_" + s for s in sample_names) + "\n")
        log_ranges = data["log"].max(axis=1) - data["log"].min(axis=1)
        for index, gene in enumerate(data["gene_names"]):
            cluster_id = labels[index]
            centroid = summaries[cluster_id]["z_profile"]
            a = z[index] - z[index].mean()
            b = centroid - centroid.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            correlation = float(a @ b / denominator) if denominator else 0.0
            out.write("{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\n".format(
                gene, cluster_id, silhouette_values[index],
                float(data["log"][index].mean()), float(log_ranges[index]),
                correlation,
                "\t".join("{:.4f}".format(v) for v in z[index]),
                "\t".join("{:.4f}".format(v) for v in data["log"][index])))

    profiles_path = os.path.join(args.outdir, "gene_cluster_profiles.tsv")
    with open(profiles_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("cluster\tn_genes\tmean_log2_range\tfold_change\t"
                  "mean_correlation_to_centroid\tmean_silhouette\t"
                  "peak_sample\ttrough_sample\t"
                  + "\t".join("zmean_" + s for s in sample_names) + "\t"
                  + "\t".join("log2mean_" + s for s in sample_names) + "\n")
        for cluster_index in order:
            s = summaries[cluster_index]
            out.write("{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t{}\t{}\t{}\n".format(
                s["cluster"], s["n_genes"], s["mean_log2_range"],
                s["fold_change"], s["mean_correlation_to_centroid"],
                s["mean_silhouette"], s["peak_sample"], s["trough_sample"],
                "\t".join("{:.4f}".format(v) for v in s["z_profile"]),
                "\t".join("{:.4f}".format(v) for v in s["log_profile"])))

    selection_path = os.path.join(args.outdir, "k_selection_metrics.tsv")
    with open(selection_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("k\tinertia\tsilhouette\tcalinski_harabasz\t"
                  "davies_bouldin\tstability_ari\tstability_sd\n")
        for index, k in enumerate(selection["k"]):
            out.write("{}\t{:.4f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(
                k, selection["inertia"][index], selection["silhouette"][index],
                selection["calinski"][index], selection["davies_bouldin"][index],
                selection["stability"][index], selection["stability_sd"][index]))

    emit("Files written")
    emit("-" * 76)
    for path in [assignments_path, profiles_path, selection_path]:
        emit("  {}".format(path))
    for name in ["01_qc_normalization.png", "02_k_selection.png",
                 "03_gene_cluster_profiles.png", "04_gene_cluster_amplitude.png",
                 "05_gene_cluster_heatmap.png", "06_gene_pca.png"]:
        emit("  {}".format(os.path.join(figures_dir, name)))
    emit("")
    emit("Elapsed: {:.1f} s".format(time.time() - started))

    text = "\n".join(report)
    print(text)
    report_path = os.path.join(args.outdir, "gene_clustering_report.txt")
    with open(report_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(text + "\n")


if __name__ == "__main__":
    main()
