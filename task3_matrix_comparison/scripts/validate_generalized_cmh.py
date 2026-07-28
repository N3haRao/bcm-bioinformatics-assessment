#!/usr/bin/env python3
"""
validate_generalized_cmh.py
============================

Correctness check for the generalized_cmh() function in t3_common.py.

t2_common's CMH test is written from scratch because statsmodels only ships a
2x2xK version (StratifiedTable), while our data needs the R-category
extension (R = 4 rows, K = 5 columns as strata). Rather than trust a hand
derived formula on faith, this script collapses the general R-category
statistic down to R = 2 categories on many random stratified tables, where it
must reduce to exactly the same answer as statsmodels' well established
implementation, and checks that it does.

Run it with:
    python validate_generalized_cmh.py
"""

import sys
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
import t3_common


def main():
    rng = np.random.default_rng(0)
    n_trials = 20
    all_matched = True

    print("Validating the R-category generalized CMH statistic against")
    print("statsmodels' StratifiedTable (which implements the standard, well")
    print("tested 2x2xK case) by collapsing to R = 2 categories.")
    print("=" * 70)

    for trial in range(n_trials):
        n_strata = int(rng.integers(3, 8))
        tables = []
        group1_by_stratum, group2_by_stratum = [], []

        for _ in range(n_strata):
            a, b, c, d = rng.integers(2, 50, size=4)
            tables.append(np.array([[a, b], [c, d]]))
            # category axis has 2 levels here (the two rows of the 2x2 table);
            # group 1 is column 0, group 2 is column 1.
            group1_by_stratum.append(np.array([a, c], dtype=float))
            group2_by_stratum.append(np.array([b, d], dtype=float))

        reference = StratifiedTable(tables).test_null_odds()
        mine_stat, mine_dof, mine_p = t3_common.generalized_cmh(
            group1_by_stratum, group2_by_stratum)

        matched = np.isclose(reference.statistic, mine_stat, atol=1e-6)
        all_matched &= bool(matched)
        print("trial {:>2}  strata={}   statsmodels={:>9.5f}  mine={:>9.5f}"
             "   match={}".format(trial + 1, n_strata, reference.statistic,
                                 mine_stat, matched))

    print("=" * 70)
    if all_matched:
        print("PASS: the generalized CMH statistic matches statsmodels exactly")
        print("on every trial when collapsed to the 2-category case.")
    else:
        print("FAIL: at least one trial disagreed. Do not trust the R-category")
        print("extension until this is resolved.")
    return 0 if all_matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
