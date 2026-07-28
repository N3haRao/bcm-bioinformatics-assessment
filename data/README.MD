# Data

| File | Used by | Committed? | Notes |
|---|---|:---:|---|
| `T1.bed.txt` | Task 1 | yes (8 MB) | 311,493 single-nucleotide sites, 4 columns (chrom, position, position, strand) |
| `T2.txt` | Task 2 | yes (1 MB) | 9,768 genes x 9 samples expression matrix |
| `T3_M1.txt` | Task 3 | yes (54 bytes) | 4 x 5 count matrix |
| `T3_M2.txt` | Task 3 | yes (53 bytes) | 4 x 5 count matrix |
| `gencode.v33.annotation.gtf.gz` | Task 1 | **no** | see below |

## Getting the GENCODE GTF

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_33/gencode.v33.annotation.gtf.gz
```

or fetch it manually from the
[GENCODE release 33 page](https://www.gencodegenes.org/human/release_33.html)
("Comprehensive gene annotation" GTF for the GRCh38 primary assembly) and save
it as `data/gencode.v33.annotation.gtf.gz`.

