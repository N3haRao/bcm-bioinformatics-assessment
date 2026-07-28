# Task 3: comparing two matrices while preserving structure and column independence

Input: `data/T3_M1.txt` and `data/T3_M2.txt`, both 4 rows (categories) x 5
columns (independent units), non-negative integer counts, no headers.

## Quick start

```bash
# Correctness check for the hand-derived generalized CMH test 
python task3_matrix_comparison/scripts/validate_generalized_cmh.py

# The full comparison 
python task3_matrix_comparison/scripts/07_compare_matrices.py \
      --m1 data/T3_M1.txt --m2 data/T3_M2.txt \
      --outdir task3_matrix_comparison/results
```

`scripts/t3_common.py` holds every statistical building block, each with a
docstring explaining why it is there; `scripts/07_compare_matrices.py` is the
driver that runs the whole analysis and writes the report and figures.

## The question

"Preserving the matrix structure" and "the columns are independent of each
other" are not throwaway phrases, they are the two constraints that rule out
the two shortcuts most people reach for first:

- **Flatten both matrices to 20 numbers and run a paired test.** This throws
  away row and column identity completely and just asks "is number k of M1
  different from number k of M2". On the real data this gives **p = 0.53**,
  a confident "no difference", because it is secretly dominated by M2 simply
  having a bigger grand total (471 vs 417), not by any real compositional
  difference.
- **Sum away the columns and run one chi-square on the row totals.** This
  keeps the category axis but destroys the column axis, silently letting
  whichever column has the biggest counts dominate, and hiding the fact that
  one of the five columns shows no difference at all.

The fix: treat each column as its own independent 2 x 4 contingency table,
test each one on its own terms, and combine the five independent results using
methods whose validity is built on that independence.

## What "preserving structure, columns independent" buys you

| Approach | Result | Verdict |
|---|---|---|
| Naive: flatten + paired t-test | p = 0.53 | wrong, ignores structure entirely |
| Naive: pool columns away, one chi2 | p = 2.5x10^-22 | right ballpark, hides everything column-level |
| **Column-wise + Fisher's combined test** | **p = 2.8x10^-23** | correct, uses column independence directly |
| **Column-wise + generalized CMH** | **p = 7.8x10^-20** | correct, pools effect rather than p-values |
| Monte Carlo calibration (1.5M sims) | p = 3.3x10^-6 | confirms the above without any asymptotic assumption |

## Per-column results

| Column | G p-value | Cramer's V | Note |
|---|---:|---:|---|
| 1 | 0.371 | 0.14 | **not significant on its own** |
| 2 | 2.4x10^-6 | 0.32 | significant |
| 3 | 7.2x10^-15 | 0.56 | significant, largest effect |
| 4 | 1.9x10^-5 | 0.36 | significant, expected count borderline (7.79) |
| 5 | 2.8x10^-4 | 0.47 | significant, expected count borderline (1.25) |

Column 1 showing no evidence of a difference is a genuine finding that the
pooled test could never surface, since it sums that column's counts in with
four others before testing anything.

## The two combination methods, and why both are used

**Fisher's combined probability test** sums `-2 * ln(p)` across the five
column p-values, which follows a chi-square distribution with `2 x 5 = 10`
degrees of freedom **only when the p-values being combined are independent**.
That is precisely the condition the task states, and precisely why this
method is valid here rather than being a statistical trick.

**Generalized Cochran-Mantel-Haenszel test** treats each column as an
independent stratum and pools the actual observed-minus-expected counts
(not just the p-values) into one statistic. It is the standard tool from
epidemiology for "R-category outcome, two groups, several independent strata"
and typically has more power than Fisher's method when the direction of the
difference is consistent across strata, which is checked directly in
`results/figures/03_row_consistency.png`.

Since scipy only ships a 2x2xK version of CMH (via
`statsmodels.stats.contingency_tables.StratifiedTable`), the R-category
extension used here is derived from the underlying multivariate hypergeometric
formula and validated in
[scripts/validate_generalized_cmh.py](scripts/validate_generalized_cmh.py):
collapsed to 2 categories, it matches statsmodels exactly across 20 random
stratified tables of varying size.

## A note on similarity, not just difference

The task also asks about similarity, which a p-value alone cannot answer,
since a p-value only says whether a difference is detectable given the sample
size, not whether it is large. Every column result also reports:

- **Cramer's V** (0 to 1, sample-size independent effect size)
- **Jensen-Shannon divergence** between the two columns' normalized
  distributions (a similarity measure with no hypothesis test attached at all)
- **cosine similarity** of the raw count vectors

so a column can be flagged as "statistically different but practically very
similar" (small V, significant p because of large counts) rather than
collapsing everything into significant/not-significant.

## The one substantive finding

Row 3 is the only category whose direction is **consistent across every
single column**: M2's proportion exceeds M1's in all five columns
(`results/figures/03_row_consistency.png`). Rows 1, 2 and 4 all flip sign at
least once across the five columns. This is exactly the kind of pattern that
pooling columns together would blend into a single number and obscure, and it
is the main reason the generalized CMH test (which rewards a consistent
direction) agrees so closely with Fisher's method (which does not need one)
here.
