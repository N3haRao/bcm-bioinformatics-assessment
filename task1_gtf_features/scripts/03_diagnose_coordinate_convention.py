#!/usr/bin/env python3
"""
03_diagnose_coordinate_convention.py
====================================

A focused diagnostic that answers one question: when T1.bed writes the same
number into the start and the end column, is that number a 0-based BED start or
a 1-based genomic position?

The test
--------
The last base of a 3'UTR is the last base of the transcript, which is the
cleavage and polyadenylation site. So we take every annotated transcript 3' end,
and for each T1 site we measure the signed distance to the nearest one. If the
sites really are transcript 3' ends, we will see a sharp spike in that
histogram, and the offset at which the spike sits tells us the convention:

    spike at  0   the number in T1 is a 1-based position   -> --point-mode pos1
    spike at -1   the number in T1 is a 0-based BED start  -> --point-mode bed0

Strand matters here. On a plus strand transcript the 3' end is the high
coordinate, on a minus strand transcript it is the low coordinate, so the two
have to be gathered separately.

Usage
-----
    # run from the repository root
    python task1_gtf_features/scripts/03_diagnose_coordinate_convention.py \
        --regions data/T1.bed.txt --features task1_gtf_features/bed_features
"""

import argparse
import bisect
import os
import sys
from collections import Counter, defaultdict


def log(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def load_reference_positions(bed_path, which_end):
    """Collect one reference position per BED row, as a 1-based coordinate.

    `which_end` is either "three_prime" or "five_prime", interpreted in
    transcript orientation so that strand is taken into account.
    """
    positions = defaultdict(list)
    with open(bed_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            strand = fields[5]

            if which_end == "three_prime":
                # 3' end: high coordinate on +, low coordinate on -
                pos = end if strand != "-" else start + 1
            else:
                # 5' end: low coordinate on +, high coordinate on -
                pos = start + 1 if strand != "-" else end

            positions[(chrom, strand)].append(pos)

    for key in positions:
        positions[key].sort()
    return positions


def nearest_signed_distance(sorted_positions, query):
    """Signed distance from `query` to the closest value in `sorted_positions`.

    Negative means the reference sits to the right of the query. Uses binary
    search because these lists hold hundreds of thousands of positions and we
    query them hundreds of thousands of times.
    """
    if not sorted_positions:
        return None
    index = bisect.bisect_left(sorted_positions, query)
    best = None
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(sorted_positions):
            distance = query - sorted_positions[candidate_index]
            if best is None or abs(distance) < abs(best):
                best = distance
    return best


def run_test(regions, reference_positions, label, window=10):
    """Histogram the signed distance from every region to the nearest reference."""
    histogram = Counter()
    total = 0
    within_window = 0

    for chrom, position, strand in regions:
        key = (chrom, strand)
        if key not in reference_positions:
            continue
        total += 1
        distance = nearest_signed_distance(reference_positions[key], position)
        if distance is None:
            continue
        if abs(distance) <= window:
            within_window += 1
            histogram[distance] += 1

    print("")
    print("Signed distance from each T1 value to the nearest {}".format(label))
    print("-" * 72)
    print("  (distance 0 means the T1 number, read as a 1-based coordinate,")
    print("   IS that reference base exactly)")
    print("")
    print("  {:>9}  {:>10}  {:>8}   {}".format(
        "offset", "sites", "% of all", "bar"))
    peak_offset, peak_count = None, -1
    for offset in range(-window, window + 1):
        count = histogram.get(offset, 0)
        if count > peak_count:
            peak_offset, peak_count = offset, count
        bar = "#" * min(60, int(60.0 * count / max(1, max(histogram.values()))))
        print("  {:>9}  {:>10,}  {:>7.3f}%   {}".format(
            offset, count, 100.0 * count / total if total else 0.0, bar))
    print("")
    print("  sites tested                      : {:,}".format(total))
    print("  within +/-{} bp of a reference    : {:,} ({:.2f}%)".format(
        window, within_window, 100.0 * within_window / total if total else 0.0))
    print("  peak offset                       : {} ({:,} sites, {:.2f}%)".format(
        peak_offset, peak_count, 100.0 * peak_count / total if total else 0.0))
    return peak_offset, peak_count, total


def main():
    parser = argparse.ArgumentParser(
        description="Work out which coordinate convention T1.bed uses.")
    parser.add_argument("--regions", required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    log("Reading regions")
    regions = []
    with open(args.regions, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped or stripped[0] == "#":
                continue
            fields = stripped.split("\t")
            if len(fields) < 4:
                continue
            # Take column 2 exactly as written, with no reinterpretation. The
            # whole point of this script is to discover how to interpret it.
            regions.append((fields[0], int(fields[1]), fields[3]))
    log("  {:,} regions".format(len(regions)))

    transcripts_bed = os.path.join(args.features, "transcripts.bed")

    print("=" * 72)
    print("Coordinate convention diagnostic for {}".format(
        os.path.basename(args.regions)))
    print("=" * 72)

    log("Testing against transcript 3' ends")
    three_prime = load_reference_positions(transcripts_bed, "three_prime")
    run_test(regions, three_prime, "annotated transcript 3' END "
                                   "(cleavage / polyA site)")
    del three_prime

    # The 5' end test is the control. If the sites were, say, transcription
    # start sites or 5' cap related, the spike would appear here instead. Seeing
    # a flat histogram here while the 3' test spikes is what turns a suggestive
    # result into a convincing one.
    log("Testing against transcript 5' ends (control)")
    five_prime = load_reference_positions(transcripts_bed, "five_prime")
    run_test(regions, five_prime, "annotated transcript 5' END "
                                  "(TSS) -- CONTROL")

    print("")
    print("=" * 72)
    print("Reading the result")
    print("-" * 72)
    print("A spike at offset 0 in the 3' end test means the numbers in T1 are")
    print("1-based genomic positions, so --point-mode pos1 is the correct")
    print("setting. A spike at offset -1 would mean they are 0-based BED")
    print("starts, so --point-mode bed0 would be correct. A flat histogram in")
    print("the 5' control confirms the 3' signal is real and not just an")
    print("artefact of sites being dense near genes.")
    print("=" * 72)


if __name__ == "__main__":
    main()
