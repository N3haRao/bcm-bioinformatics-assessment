# Bioinformatics Assistant Technical Assessment
### Baylor College of Medicine

Solutions to a four-part technical assessment: transcriptome feature
extraction and annotation from a GTF file, unsupervised clustering of an
RNA-seq expression matrix, the design of a statistical test for comparing
two structured count matrices, and a small demo showing how an LLM can
answer questions grounded in an uploaded file.

## About this repository

This repository contains my solutions for the Bioinformatics Assistant
technical assessment. Each task lives in its own folder with a complete
write-up: methodology, the reasoning behind each design decision,
correctness checks, and interpretation of the results.

| Task | Folder | What it does |
|---|---|---|
| 1 | [task1_gtf_features/](task1_gtf_features/README.md) | Extracts genes, transcripts, exons, UTRs and introns from a GENCODE GTF into BED files, then maps an external region set onto them |
| 2 | [task2_expression_clustering/](task2_expression_clustering/README.md) | Clusters genes by expression pattern and samples by gene signature, in both directions |
| 3 | [task3_matrix_comparison/](task3_matrix_comparison/README.md) | Designs and validates a statistical test to compare two matrices while respecting row/column structure and column independence |
| 4 | [task4_llm_demo/](task4_llm_demo/README.md) | A Streamlit app: upload a file, ask Claude questions about it, and see proof the answers are grounded in the file rather than guessed |

Click through to each task's README for the full write-up: reasoning behind
every design decision, correctness checks, figures, and interpretation of the
results. This top-level README is just the map and the setup instructions.

---

## Repository structure

```
.
├── README.md                       <- you are here
├── requirements.txt                 numpy, scipy, matplotlib, scikit-learn, statsmodels
├── .gitignore
├── data/
│   ├── README.md                    ENCODE GTF download instructions
│   ├── T1.bed.txt                   Task 1 input: 311,493 single-nucleotide sites
│   ├── T2.txt                       Task 2 input: 9,768 genes x 9 samples
│   ├── T3_M1.txt, T3_M2.txt         Task 3 inputs: two 4x5 count matrices
├── task1_gtf_features/
│   ├── README.md
│   ├── scripts/                     01_gtf_to_bed.py, 02_map_regions_to_features.py,
│   │                                03_diagnose_coordinate_convention.py,
│   │                                04_analyse_annotation.py
│   ├── bed_features/                generated, gitignored (~490 MB)
│   └── results/                     small reports committed; large regenerated files gitignored
├── task2_expression_clustering/
│   ├── README.md
│   ├── scripts/                     t2_common.py, 05_cluster_genes.py, 06_cluster_samples.py
│   └── results/                     reports, tables, and 11 figures 
├── task3_matrix_comparison/
│   ├── README.md
│   ├── scripts/                     t3_common.py, validate_generalized_cmh.py,
│   │                                07_compare_matrices.py
│   └── results/                     reports, tables, and 5 figures 
└── task4_llm_demo/
    ├── README.md
    ├── requirements.txt              streamlit, anthropic, pypdf, pandas (separate from the root file)
    ├── app.py                        Streamlit UI
    ├── file_loader.py                 .txt/.csv/.tsv/.pdf -> model context + preview
    ├── llm_client.py                   Anthropic API calls, system prompt, caching
    └── sample_data/                    two small invented demo files
```

---

## Setup

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

Task 1 additionally needs the GENCODE GTF, which is not committed to keep the
repository small (see [data/README.md](data/README.md) for the one-line
download). Tasks 2 and 3 need nothing beyond `pip install -r requirements.txt`;
their input matrices are already in `data/`. Task 4 has its own
`requirements.txt` (`pip install -r task4_llm_demo/requirements.txt`) and needs
an Anthropic API key set as `ANTHROPIC_API_KEY` before running.

All commands in every task README are written to be run from this repository
root.

---

## One-paragraph summary of each task's findings

**Task 1.** Derived every UTR from `exon minus CDS` rather than trusting
GENCODE's own UTR lines, then used those same lines as a correctness check:
100.0000% agreement across 311,868 blocks. Discovered, rather than assumed,
that the external BED file's coordinates are 1-based (21,984 sites land
exactly on an annotated transcript 3' end, a spike absent from the same test
against 5' ends) and that ~29% of "unmapped" sites are not noise but sit
immediately downstream of annotated gene ends, consistent with T1 being a
polyadenylation-site dataset that GENCODE's 3' UTR annotation does not fully
capture.

**Task 2.** Median-centring the columns before z-scoring genes was not
optional: without it, the 3,000 flattest (least informative) genes acquire a
spurious coherent z-profile of magnitude 1.14 that tracks sequencing depth
exactly, because that depth difference is confounded with the sample groups.
Chose the number of gene clusters (k=4) using a stability-gated rule rather
than raw silhouette, which structurally favours k=2 on continuous data and
would have missed real substructure. Found that the sample groups do not match
what the S1-S9 naming implies: the true k=3 split is {S1,S2,S3}, {S4,S6},
{S5,S7,S8,S9}, with S5 worth checking for a batch effect or sample swap.

**Task 3.** The question was a design problem, not a lookup: two shortcuts
(flattening to a paired test, and pooling away the columns before testing)
were shown to fail or mislead on the real data, then replaced with a
column-by-column analysis whose five independent results are combined via
Fisher's method and a hand-derived, statsmodels-validated generalized
Cochran-Mantel-Haenszel test, both cross-checked against a 1.5-million-sample
Monte Carlo calibration. All three agree the matrices differ significantly,
and the analysis also surfaces which specific column shows no difference and
which category's difference is consistent across every column.

**Task 4.** A Streamlit app answers natural-language questions about an
uploaded file using Claude, with the actual design problem being how to prove
the answers are grounded rather than guessed: a system prompt instructs the
model to say plainly when something is not in the document, a transparency
panel shows exactly what text the model received, and a "prove it's grounded"
toggle runs every question a second time with no file at all so the two
answers can be compared side by side. Tables too large to embed in full get a
verbatim sample of 50 rows plus exact summary statistics computed with pandas
over the entire table, so aggregate questions (counts, averages) can still be
answered precisely rather than estimated from a preview.
