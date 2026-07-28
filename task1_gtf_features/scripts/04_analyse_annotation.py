#!/usr/bin/env python3
"""
04_analyse_annotation.py
========================

Four analyses, each answering a specific question.

A. Enrichment against available sequence space
   The raw percentages from script 2 cannot be read directly. 3'UTRs contain
   vastly more sequence than 5'UTRs do, and introns contain more than everything
   else combined, so "29.65% of sites are in 3'UTRs" might just be telling us
   that 3'UTRs are big. To fix this we build the null properly: partition the
   genome into the same five classes using the same priority order script 02
   uses, count how many base pairs each class actually owns, and compare the
   observed site counts against what random placement across that space would
   give. Only after that normalisation can we talk about enrichment.

B. Where the unmapped (UN) sites sit
   Roughly a third of the sites do not touch any annotated transcript feature.
   That could mean the data is noisy, or it could mean the annotation is
   incomplete. These have very different implications, and they are easy to tell
   apart: if UN sites cluster just downstream of annotated transcript 3' ends,
   they are unannotated 3'UTR extensions rather than noise. So we measure the
   distance from every UN site to the nearest annotated 3' end and check whether
   they sit downstream.

C. Metagene position inside the 3'UTR
   For the sites that did land in a 3'UTR, we ask where in the 3'UTR they sit,
   measured in spliced transcript coordinates from the stop codon (0.0) to the
   polyA site (1.0). Different classes of single nucleotide data have very
   different signatures here. Something that piles up hard against 1.0 is
   telling you about 3' end formation. Something piling up against 0.0 would be
   pointing at the stop codon instead.

D. Gene biotype breakdown
   Which kinds of genes are collecting these sites. A clean protein coding
   skew means one thing, a pile of pseudogene hits usually means mapping
   artefacts.

Usage
-----
    # run from the repository root
    python task1_gtf_features/scripts/04_analyse_annotation.py \
        --detailed task1_gtf_features/results/T1_annotated_detailed.tsv \
        --features task1_gtf_features/bed_features \
        --out task1_gtf_features/results/T1_interpretation.txt
"""

import argparse
import bisect
import os
import sys
import time
from collections import Counter, defaultdict


# GRCh38 primary assembly total length. Only used to put an approximate
# expectation on the intergenic fraction, never for any coordinate arithmetic,
# so a small inaccuracy here cannot corrupt any of the mapping results.
GRCH38_PRIMARY_ASSEMBLY_BP = 3088269832


def log(message):
    sys.stderr.write("[{}] {}\n".format(time.strftime("%H:%M:%S"), message))
    sys.stderr.flush()


def natural_chrom_key(chrom):
    """Kept identical to scripts 01 and 02 so chromosome blocks line up."""
    name = chrom[3:] if chrom.startswith("chr") else chrom
    if name.isdigit():
        return (0, int(name), "")
    if name in ("X", "Y"):
        return (1, 0, name)
    if name in ("M", "MT"):
        return (2, 0, name)
    return (3, 0, chrom)


class ChromBlockReader(object):
    """Streams a coordinate sorted BED file one chromosome at a time.

    This is deliberately a copy of the class in script 0 rather than an import.
    Python cannot import a module whose name starts with a digit without jumping
    through importlib hoops, and keeping each script independently runnable is
    worth more here than avoiding thirty duplicated lines.
    """

    def __init__(self, path):
        self.path = path
        self.exists = os.path.isfile(path)
        self._iter = self._rows() if self.exists else iter(())
        self._pending = next(self._iter, None)

    def _rows(self):
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line[0] == "#":
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 6:
                    yield fields

    def take(self, chrom):
        if self._pending is None:
            return []
        target = natural_chrom_key(chrom)
        while self._pending is not None and \
                natural_chrom_key(self._pending[0]) < target:
            skipping = self._pending[0]
            while self._pending is not None and self._pending[0] == skipping:
                self._pending = next(self._iter, None)
        rows = []
        while self._pending is not None and self._pending[0] == chrom:
            rows.append(self._pending)
            self._pending = next(self._iter, None)
        return rows

    def all_chroms_remaining(self):
        """Peek at which chromosome is next, or None when exhausted."""
        return self._pending[0] if self._pending is not None else None


# ---------------------------------------------------------------------------
# Interval set algebra, used to build the non-redundant class partition
# ---------------------------------------------------------------------------

def merge(intervals):
    """Collapse a list of (start, end) into a sorted, non-overlapping list."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1][1] = end
        else:
            out.append([start, end])
    return [(s, e) for s, e in out]


def subtract(base, holes):
    """Remove `holes` from `base`. Both must be sorted and non-overlapping.

    Walks the two lists together like a merge join, which keeps it linear rather
    than the quadratic behaviour you get from repeatedly subtracting one hole at
    a time from a growing result list.
    """
    if not holes:
        return list(base)
    result = []
    hole_index = 0
    total_holes = len(holes)
    for start, end in base:
        cursor = start
        # Skip holes that finish before this interval even starts.
        while hole_index < total_holes and holes[hole_index][1] <= cursor:
            hole_index += 1
        scan = hole_index
        while scan < total_holes and holes[scan][0] < end:
            hole_start, hole_end = holes[scan]
            if hole_start > cursor:
                result.append((cursor, min(hole_start, end)))
            cursor = max(cursor, hole_end)
            if cursor >= end:
                break
            scan += 1
        if cursor < end:
            result.append((cursor, end))
    return result


def total_bp(intervals):
    return sum(end - start for start, end in intervals)


# ---------------------------------------------------------------------------
# Analysis A: how much sequence does each class actually own
# ---------------------------------------------------------------------------

CLASS_ORDER = ["CDS", "5UTR", "3UTR", "noncoding_exon", "intron"]


def measure_class_space(features_dir, chroms_needed):
    """Base pairs owned by each feature class, after priority deduplication.

    The priority order has to match script 2 exactly, otherwise the null we
    build here would not correspond to the labels we are testing. So a base that
    is CDS in one isoform and intron in another counts once, towards CDS, in
    both places.

    Everything is computed per chromosome AND per strand, because the annotation
    is strand specific and script 2 matched strand aware. The plus strand and
    the minus strand of the same chromosome are, for our purposes, two separate
    searchable spaces.
    """
    readers = {
        "CDS": ChromBlockReader(os.path.join(features_dir, "coding_exons.bed")),
        "5UTR": ChromBlockReader(os.path.join(features_dir, "five_prime_utrs.bed")),
        "3UTR": ChromBlockReader(os.path.join(features_dir, "three_prime_utrs.bed")),
        "intron": ChromBlockReader(os.path.join(features_dir, "introns.bed")),
    }
    exon_reader = ChromBlockReader(os.path.join(features_dir, "exons.bed"))

    bp_by_class = Counter()

    for chrom in sorted(chroms_needed, key=natural_chrom_key):
        raw = {name: reader.take(chrom) for name, reader in readers.items()}
        exon_rows = exon_reader.take(chrom)

        # Same non-coding exon definition as script 02: exons of transcripts
        # that carry no CDS anywhere.
        coding_tx = {fields[7] for fields in raw["CDS"] if len(fields) > 7}
        raw["noncoding_exon"] = [fields for fields in exon_rows
                                 if len(fields) > 7 and fields[7] not in coding_tx]
        del exon_rows

        for strand in ("+", "-"):
            claimed = []      # everything already assigned to a higher priority
            for class_name in CLASS_ORDER:
                blocks = merge([
                    (int(f[1]), int(f[2])) for f in raw[class_name]
                    if f[5] == strand
                ])
                # Whatever survives after removing the higher priority classes
                # is what this class uniquely owns.
                unique = subtract(blocks, claimed)
                bp_by_class[class_name] += total_bp(unique)
                claimed = merge(claimed + unique)
        del raw

    return bp_by_class


# ---------------------------------------------------------------------------
# Analysis B and the polyA distance work
# ---------------------------------------------------------------------------

def load_transcript_three_prime_ends(features_dir):
    """1-based coordinate of the last base of every transcript, by chrom+strand."""
    ends = defaultdict(list)
    path = os.path.join(features_dir, "transcripts.bed")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                continue
            start, end, strand = int(fields[1]), int(fields[2]), fields[5]
            # 3' end is the high coordinate on the plus strand and the low
            # coordinate on the minus strand.
            ends[(fields[0], strand)].append(end if strand != "-" else start + 1)
    for key in ends:
        ends[key].sort()
    return ends


def signed_distance_downstream(sorted_ends, position, strand):
    """Distance from `position` to the nearest transcript 3' end, oriented so
    that a POSITIVE number means the site is downstream of the gene end.

    Orienting by strand rather than by raw coordinate is the whole point. A site
    5 kb to the left of a minus strand gene's 3' end is downstream of that gene,
    whereas the same offset on a plus strand gene would be upstream. Reporting
    raw coordinate differences would mix the two and wash the signal out.
    """
    if not sorted_ends:
        return None
    index = bisect.bisect_left(sorted_ends, position)
    best = None
    for candidate in (index - 1, index):
        if 0 <= candidate < len(sorted_ends):
            raw = position - sorted_ends[candidate]
            oriented = raw if strand != "-" else -raw
            if best is None or abs(oriented) < abs(best):
                best = oriented
    return best


# ---------------------------------------------------------------------------
# Analysis C: metagene position within the 3'UTR
# ---------------------------------------------------------------------------

def load_three_prime_utr_blocks(features_dir, wanted_transcripts):
    """3'UTR blocks for the transcripts we actually need, in 5' to 3' order.

    Restricting to the transcripts referenced by the annotated sites keeps this
    small. Loading all 158,588 3'UTR rows would work too but there is no reason
    to when a few thousand will do.
    """
    blocks = defaultdict(list)
    path = os.path.join(features_dir, "three_prime_utrs.bed")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            tx_id = fields[7]
            if tx_id not in wanted_transcripts:
                continue
            blocks[tx_id].append((int(fields[1]), int(fields[2]), fields[5]))

    ordered = {}
    for tx_id, items in blocks.items():
        strand = items[0][2]
        # Sort into transcription order. On the minus strand that is descending
        # genomic coordinate.
        items.sort(key=lambda b: b[0], reverse=(strand == "-"))
        ordered[tx_id] = (strand, [(s, e) for s, e, _ in items])
    return ordered


def relative_position_in_utr(utr_blocks, strand, site_bed_start):
    """Where in the spliced 3'UTR a site sits, as a fraction from 0.0 to 1.0.
    """
    offset = 0
    total = sum(end - start for start, end in utr_blocks)
    if total == 0:
        return None
    for start, end in utr_blocks:
        length = end - start
        if start <= site_bed_start < end:
            within = (site_bed_start - start) if strand != "-" \
                else (end - 1 - site_bed_start)
            return (offset + within) / float(total)
        offset += length
    return None


# ---------------------------------------------------------------------------
# Reading the annotated table
# ---------------------------------------------------------------------------

def read_detailed(path):
    """Load the detailed output of script 02 into a list of light tuples."""
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        column = {name: index for index, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(header):
                continue
            rows.append((
                fields[column["chrom"]],
                int(fields[column["query_start_bed"]]),
                int(fields[column["query_end_bed"]]),
                fields[column["strand"]],
                fields[column["feature_type"]],
                fields[column["feature_detail"]],
                fields[column["gene_id"]],
                fields[column["transcript_ids"]],
            ))
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interpret the output of script 02.")
    parser.add_argument("--detailed", required=True,
                        help="The *_annotated_detailed.tsv from script 02.")
    parser.add_argument("--features", required=True,
                        help="Directory of BED files from script 01.")
    parser.add_argument("--out", required=True, help="Report file to write.")
    parser.add_argument("--genome-size", type=int,
                        default=GRCH38_PRIMARY_ASSEMBLY_BP,
                        help="Assembly size, used only for the approximate "
                             "intergenic expectation.")
    args = parser.parse_args()

    started = time.time()
    report = []

    def emit(line=""):
        report.append(line)

    log("Reading annotated table")
    rows = read_detailed(args.detailed)
    total_sites = len(rows)
    log("  {:,} sites".format(total_sites))

    detail_counts = Counter(row[5] for row in rows)
    chroms_needed = {row[0] for row in rows}

    emit("=" * 74)
    emit("Interpretation of {}".format(os.path.basename(args.detailed)))
    emit("=" * 74)
    emit("Sites analysed: {:,}".format(total_sites))
    emit("")

    # ---- Analysis A -------------------------------------------------------
    log("Analysis A: measuring how much sequence each feature class owns")
    bp_by_class = measure_class_space(args.features, chroms_needed)
    annotated_bp = sum(bp_by_class.values())

    # The searchable space is strand specific, so both strands of the assembly
    # are in play. That is why the intergenic estimate uses twice the genome.
    stranded_genome_bp = 2 * args.genome_size
    intergenic_bp = max(0, stranded_genome_bp - annotated_bp)

    mapped_sites = total_sites - detail_counts.get("UN", 0)

    emit("A. Enrichment against available sequence space")
    emit("-" * 74)
    emit("   Every base is assigned to exactly ONE class using the same priority")
    emit("   order script 02 used (CDS > 5UTR > 3UTR > noncoding exon > intron),")
    emit("   counted separately per strand. Expected counts assume the sites were")
    emit("   scattered at random across that space.")
    emit("")
    emit("   {:<17} {:>14} {:>8} {:>10} {:>10} {:>8}".format(
        "class", "unique bp", "% space", "observed", "expected", "log2 FC"))
    for class_name in CLASS_ORDER:
        bp = bp_by_class[class_name]
        space_fraction = bp / float(annotated_bp) if annotated_bp else 0.0
        observed = detail_counts.get(class_name, 0)
        expected = space_fraction * mapped_sites
        if expected > 0 and observed > 0:
            import math
            fold = math.log(observed / expected, 2)
            fold_text = "{:+.2f}".format(fold)
        else:
            fold_text = "n/a"
        emit("   {:<17} {:>14,} {:>7.2f}% {:>10,} {:>10,.0f} {:>8}".format(
            class_name, bp, 100.0 * space_fraction, observed, expected, fold_text))
    emit("   {:<17} {:>14,} {:>8} {:>10,}".format(
        "TOTAL annotated", annotated_bp, "100.00%", mapped_sites))
    emit("")

    un_observed = detail_counts.get("UN", 0)
    un_expected = total_sites * intergenic_bp / float(stranded_genome_bp)
    import math
    emit("   Unmapped sites for comparison (approximate, uses assembly size):")
    emit("   {:<17} {:>14,} {:>7.2f}% {:>10,} {:>10,.0f} {:>8}".format(
        "UN / intergenic", intergenic_bp,
        100.0 * intergenic_bp / stranded_genome_bp, un_observed, un_expected,
        "{:+.2f}".format(math.log(un_observed / un_expected, 2))
        if un_expected > 0 and un_observed > 0 else "n/a"))
    emit("")

    # ---- Analysis B -------------------------------------------------------
    log("Analysis B: locating the unmapped sites relative to gene 3' ends")
    three_prime_ends = load_transcript_three_prime_ends(args.features)

    # Distance bins, oriented so positive means downstream of the gene 3' end.
    bins = [(0, 100), (100, 500), (500, 1000), (1000, 5000),
            (5000, 10000), (10000, 50000), (50000, 10 ** 12)]
    downstream_counts = Counter()
    upstream_counts = Counter()
    no_reference = 0

    # We do the same for ALL sites too, so the UN pattern can be compared
    # against the background rather than read in isolation.
    all_exact = 0
    all_within_50 = 0
    all_tested = 0

    for chrom, bed_start, bed_end, strand, feature_type, detail, _, _ in rows:
        # Convert the BED interval back to the 1-based coordinate of its base.
        position = bed_end
        key = (chrom, strand)
        if key not in three_prime_ends:
            if feature_type == "UN":
                no_reference += 1
            continue
        distance = signed_distance_downstream(
            three_prime_ends[key], position, strand)
        if distance is None:
            continue

        all_tested += 1
        if distance == 0:
            all_exact += 1
        if abs(distance) <= 50:
            all_within_50 += 1

        if feature_type != "UN":
            continue
        target = downstream_counts if distance >= 0 else upstream_counts
        magnitude = abs(distance)
        for low, high in bins:
            if low <= magnitude < high:
                target[(low, high)] += 1
                break

    emit("B. Where the unmapped (UN) sites sit relative to annotated gene ends")
    emit("-" * 74)
    emit("   Distance is oriented by strand, so 'downstream' really means past")
    emit("   the 3' end of the nearest transcript, not merely at a higher")
    emit("   coordinate.")
    emit("")
    total_un_binned = sum(downstream_counts.values()) + sum(upstream_counts.values())
    emit("   {:<22} {:>12} {:>9} {:>12} {:>9}".format(
        "distance from 3' end", "downstream", "%", "upstream", "%"))
    for low, high in bins:
        down = downstream_counts.get((low, high), 0)
        up = upstream_counts.get((low, high), 0)
        label = "{:,} to {:,} bp".format(low, high) if high < 10 ** 12 \
            else "> {:,} bp".format(low)
        emit("   {:<22} {:>12,} {:>8.2f}% {:>12,} {:>8.2f}%".format(
            label, down,
            100.0 * down / total_un_binned if total_un_binned else 0.0,
            up, 100.0 * up / total_un_binned if total_un_binned else 0.0))
    down_total = sum(downstream_counts.values())
    up_total = sum(upstream_counts.values())
    emit("")
    emit("   UN sites downstream of a gene end : {:,} ({:.2f}%)".format(
        down_total, 100.0 * down_total / total_un_binned if total_un_binned else 0.0))
    emit("   UN sites upstream of a gene end   : {:,} ({:.2f}%)".format(
        up_total, 100.0 * up_total / total_un_binned if total_un_binned else 0.0))
    emit("   UN sites within 10 kb downstream  : {:,} ({:.2f}% of all UN)".format(
        sum(downstream_counts.get(b, 0) for b in bins[:5]),
        100.0 * sum(downstream_counts.get(b, 0) for b in bins[:5]) / un_observed
        if un_observed else 0.0))
    emit("")
    emit("   For context, across ALL {:,} sites:".format(all_tested))
    emit("     exactly on an annotated transcript 3' end : {:,} ({:.2f}%)".format(
        all_exact, 100.0 * all_exact / all_tested if all_tested else 0.0))
    emit("     within 50 bp of one                       : {:,} ({:.2f}%)".format(
        all_within_50, 100.0 * all_within_50 / all_tested if all_tested else 0.0))
    emit("")

    # ---- Analysis C -------------------------------------------------------
    log("Analysis C: metagene position within the 3'UTR")
    utr_sites = [row for row in rows if row[5] == "3UTR"]
    wanted = set()
    for row in utr_sites:
        first_tx = row[7].split(",")[0]
        if first_tx.startswith("ENST"):
            wanted.add(first_tx)
    utr_blocks = load_three_prime_utr_blocks(args.features, wanted)

    n_bins = 20
    metagene = [0] * n_bins
    placed = 0
    for row in utr_sites:
        first_tx = row[7].split(",")[0]
        entry = utr_blocks.get(first_tx)
        if entry is None:
            continue
        strand, blocks = entry
        fraction = relative_position_in_utr(blocks, strand, row[1])
        if fraction is None:
            continue
        index = min(n_bins - 1, int(fraction * n_bins))
        metagene[index] += 1
        placed += 1

    emit("C. Metagene position of the 3'UTR sites")
    emit("-" * 74)
    emit("   Position measured in SPLICED transcript coordinates, from the stop")
    emit("   codon (0.00) to the polyA site (1.00), using the representative")
    emit("   transcript for each site.")
    emit("")
    peak = max(metagene) if metagene else 1
    for index in range(n_bins):
        low = index / float(n_bins)
        high = (index + 1) / float(n_bins)
        bar = "#" * int(58.0 * metagene[index] / peak) if peak else ""
        emit("   {:.2f}-{:.2f} {:>9,} {:>7.2f}%  {}".format(
            low, high, metagene[index],
            100.0 * metagene[index] / placed if placed else 0.0, bar))
    emit("")
    emit("   sites placed on a metagene: {:,} of {:,} 3'UTR sites".format(
        placed, len(utr_sites)))
    if placed:
        last_decile = metagene[-2] + metagene[-1]
        first_decile = metagene[0] + metagene[1]
        emit("   in the final 10% of the 3'UTR : {:,} ({:.2f}%)".format(
            last_decile, 100.0 * last_decile / placed))
        emit("   in the first 10% of the 3'UTR : {:,} ({:.2f}%)".format(
            first_decile, 100.0 * first_decile / placed))
        emit("   ratio last : first = {:.2f}".format(
            last_decile / float(first_decile) if first_decile else float("inf")))
    emit("")

    # ---- Analysis D -------------------------------------------------------
    log("Analysis D: gene biotype breakdown")
    gene_types = {}
    with open(os.path.join(args.features, "genes.bed"), "r",
              encoding="utf-8") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) > 8:
                gene_types[fields[6]] = fields[8]

    biotype_counts = Counter()
    for row in rows:
        if row[4] == "UN":
            continue
        first_gene = row[6].split(",")[0]
        biotype_counts[gene_types.get(first_gene, "unknown")] += 1

    emit("D. Biotype of the genes the sites landed in")
    emit("-" * 74)
    biotype_total = sum(biotype_counts.values())
    for biotype, count in biotype_counts.most_common(15):
        emit("   {:<40} {:>10,} {:>7.2f}%".format(
            biotype, count, 100.0 * count / biotype_total if biotype_total else 0.0))
    emit("")
    emit("Elapsed: {:.1f} s".format(time.time() - started))

    text = "\n".join(report)
    print(text)
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write(text + "\n")
    log("Report written to {}".format(args.out))


if __name__ == "__main__":
    main()
