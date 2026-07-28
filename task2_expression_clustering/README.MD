# Task 2: Clustering genes and samples in an expression matrix

Input `data/T2.txt`: 9,768 genes x 9 samples (S1-S9), tab separated, no missing
values.
## Quick start

```bash
# Cluster genes, plot a representative profile per cluster  
python task2_expression_clustering/scripts/05_cluster_genes.py \
      --matrix data/T2.txt --outdir task2_expression_clustering/results

# Cluster samples, derive and plot gene signatures 
python task2_expression_clustering/scripts/06_cluster_samples.py \
      --matrix data/T2.txt --outdir task2_expression_clustering/results \
      --gene-clusters task2_expression_clustering/results/gene_clusters.tsv
```

`scripts/t2_common.py` holds the shared loading, QC and preprocessing so both
directions see an identical matrix.

## Outputs

`results/` contains three reports (`gene_clustering_report.txt`,
`sample_clustering_report.txt`), five tables, and eleven figures under
`results/figures/`:

| Figure | Shows |
|---|---|
| `01_qc_normalization.png` | the per-sample offset, before and after |
| `02_k_selection.png` | inertia, silhouette, Calinski-Harabasz, stability vs k |
| `03_gene_cluster_profiles.png` | **representative profile per gene cluster** (z-scored) |
| `04_gene_cluster_amplitude.png` | the same clusters on the real expression scale |
| `05_gene_cluster_heatmap.png` | all 9,767 genes, blocked by cluster |
| `06_gene_pca.png` | genes in PCA space, coloured by cluster |
| `07_sample_correlation_comparison.png` | why gene-centring is required |
| `08_sample_structure.png` | dendrogram, correlation, bootstrap support, silhouette |
| `09_sample_pca.png` | samples in PCA space |
| `10_sample_cluster_signatures.png` | **gene signature per sample cluster** |
| `11_gene_and_sample_clusters.png` | links the two analyses |

## Preprocessing, and why each step is there

`log2(x+1)` then **median-centre the columns** then **z-score each gene**.

T2's column sums range 36.8M to 46.0M (CV
7.3%), and in log2 space the column medians span 0.233 units, a 17.5% linear
difference. Critically that offset is **confounded with the sample groups**: S1-S3
all sit low, S4-S9 all sit high. Left in place, gene-wise z-scoring launders it
into fake biology. Measured directly: take the 3,000 least variable genes, the
ones with nothing to say, and z-score them without column correction. Their mean
profile comes out with a magnitude of **1.14 z units**, tracking library size
exactly, which would have formed a large and entirely spurious cluster. Median
centering halves it to 0.60. Pass `--normalize none` to reproduce the artefact.

This is deliberately conservative: nothing can distinguish a genuine genome-wide
transcriptional shift from a depth difference without spike-ins, so global shifts
are treated as technical. It is safe here because the real signal (1.1 log2, 2.2x)
is roughly five times larger than the offset being removed (0.23 log2).

Z-scoring is needed so clusters reflect pattern shape rather than abundance tier.
A useful consequence: once profiles are z-scored, squared Euclidean distance is
`2n(1-r)`, so **k-means on this matrix already is correlation clustering**.

Note: k-means is implemented from scratch in `t2_common.py` rather than imported
from scikit-learn, because the scikit-learn build in the original development
environment crashed on `KMeans` due to a `threadpoolctl` version mismatch (every
other scikit-learn function used here, silhouette, adjusted Rand index, PCA,
worked fine). The hand-rolled version (k-means++ seeding, multiple restarts) is
validated against `scipy.cluster.vq.kmeans2` in the analysis: identical or lower
inertia at every k tested.

## A note on the T2 sample structure

The sample names imply three triplets (S1-3, S4-6, S7-9). **The data does not
support that.** The primary split {S1,S2,S3} vs {S4-S9} is absolute (100% vs 0%
bootstrap co-assignment over 1,000 gene resamples, silhouette 0.727). But forcing
k=3 gives **{S1,S2,S3}, {S4,S6}, {S5,S7,S8,S9}**, not the naming-implied triplets,
at 95-96% support. S5 correlates better with S7 (0.550) than with S4 (0.366) or S6
(0.445). 
