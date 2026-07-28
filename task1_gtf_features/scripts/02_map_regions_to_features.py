#!/usr/bin/env python3
"""
02_map_regions_to_features.py
=============================

Take an external BED file of regions (here T1.bed) and work out, for every
region, which transcriptome feature it falls in. Writes the original file back
out with an extra column holding the feature type:

    exon, 5UTR, 3UTR, intron, UN

where UN means the region did not overlap any annotated transcript feature.

Usage
-----
    # run from the repository root
    python task1_gtf_features/scripts/02_map_regions_to_features.py \
        --regions data/T1.bed.txt \
        --features task1_gtf_features/bed_features \
        --outdir task1_gtf_features/results \
        --prefix T1

Only the Python standard library is used.
"""

import argparse
import heapq
import os
import sys
import time
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Shared helpers (kept identical to script 01 so the two agree on chrom order)
# ---------------------------------------------------------------------------

def log(message):
    sys.stderr.write("[{}] {}\n".format(time.strftime("%H:%M:%S"), message))
    sys.stderr.flush()


def natural_chrom_key(chrom):

    name = chrom[3:] if chrom.startswith("chr") else chrom
    if name.isdigit():
        return (0, int(name), "")
    if name in ("X", "Y"):
        return (1, 0, name)
    if name in ("M", "MT"):
        return (2, 0, name)
    return (3, 0, chrom)


# ---------------------------------------------------------------------------
# Feature classes and how they collapse to the labels the task asks for
# ---------------------------------------------------------------------------

# Priority order. Index 0 wins over index 1 and so on. Storing the class as a
# small integer means "which of these two hits is better" is just a comparison,
# which matters when we do it a few million times.
CLASS_CDS = 0
CLASS_5UTR = 1
CLASS_3UTR = 2
CLASS_NC_EXON = 3
CLASS_INTRON = 4

CLASS_DETAIL_NAME = {
    CLASS_CDS: "CDS",
    CLASS_5UTR: "5UTR",
    CLASS_3UTR: "3UTR",
    CLASS_NC_EXON: "noncoding_exon",
    CLASS_INTRON: "intron",
}

# The five label vocabulary the task asked for. Note that both a CDS block and a
# non-coding exon are, quite literally, exons, so both map to "exon".
CLASS_REQUIRED_LABEL = {
    CLASS_CDS: "exon",
    CLASS_5UTR: "5UTR",
    CLASS_3UTR: "3UTR",
    CLASS_NC_EXON: "exon",
    CLASS_INTRON: "intron",
}

UNMAPPED_LABEL = "UN"


# ---------------------------------------------------------------------------
# Streaming a coordinate sorted BED file one chromosome at a time
# ---------------------------------------------------------------------------

class ChromBlockReader(object):
    """Hands out the rows of a coordinate sorted BED file, one chromosome block
    at a time.
    """

    def __init__(self, path):
        self.path = path
        self.exists = os.path.isfile(path)
        self._iter = self._row_iterator() if self.exists else iter(())
        self._pending = next(self._iter, None)

    def _row_iterator(self):
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line[0] == "#":
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6:
                    continue
                yield fields

    def take(self, chrom):
        """Return every row for `chrom`. Any chromosomes that sort before the
        requested one are skipped, since we will never be asked for them again."""
        if self._pending is None:
            return []
        target_key = natural_chrom_key(chrom)

        # Fast forward past chromosomes we have already moved beyond.
        while self._pending is not None and \
                natural_chrom_key(self._pending[0]) < target_key:
            skipping = self._pending[0]
            while self._pending is not None and self._pending[0] == skipping:
                self._pending = next(self._iter, None)

        rows = []
        while self._pending is not None and self._pending[0] == chrom:
            rows.append(self._pending)
            self._pending = next(self._iter, None)
        return rows


# Column positions in the BED files produced by script 01. All the transcript
# level files share one layout, which is deliberate and makes this parsing
# trivial.
COL_CHROM, COL_START, COL_END, COL_NAME, COL_SCORE, COL_STRAND = 0, 1, 2, 3, 4, 5
COL_TRANSCRIPT_ID = 7
COL_GENE_ID = 9
COL_GENE_NAME = 10

# genes.bed has a shorter layout of its own.
COL_GENE_ID_IN_GENES = 6
COL_GENE_NAME_IN_GENES = 7


# ---------------------------------------------------------------------------
# Building the searchable interval list for one chromosome
# ---------------------------------------------------------------------------

def build_chrom_intervals(blocks_by_class, string_pool):
    """Turn raw BED rows for one chromosome into a compact, sorted interval list.

    `blocks_by_class` maps a class constant to the list of BED rows for it.

    Two things happen here that are worth explaining.

    First, deduplication. Isoforms of the same gene share exons constantly, so
    the same interval turns up over and over with only the transcript ID
    differing. We collapse rows that share (start, end, strand, class, gene) into
    a single entry, remember one representative transcript ID, and count how many
    transcripts supported it. .

    Second, string pooling. Gene names and IDs repeat thousands of times. Keeping
    one shared Python string object per distinct name instead of thousands of
    equal copies is a large and free memory saving.
    """
    aggregated = {}

    for class_id, rows in blocks_by_class.items():
        for fields in rows:
            start = int(fields[COL_START])
            end = int(fields[COL_END])
            strand = fields[COL_STRAND]
            gene_id = fields[COL_GENE_ID] if len(fields) > COL_GENE_ID else "NA"
            gene_name = fields[COL_GENE_NAME] if len(fields) > COL_GENE_NAME else "NA"
            tx_id = fields[COL_TRANSCRIPT_ID] if len(fields) > COL_TRANSCRIPT_ID else "NA"

            gene_id = string_pool.setdefault(gene_id, gene_id)
            gene_name = string_pool.setdefault(gene_name, gene_name)

            key = (start, end, strand, class_id, gene_id)
            existing = aggregated.get(key)
            if existing is None:
                # [gene_name, representative transcript, transcript count]
                aggregated[key] = [gene_name, tx_id, 1]
            else:
                existing[2] += 1

    # Flatten into a list sorted by start, which is what the sweep needs.
    intervals = []
    for (start, end, strand, class_id, gene_id), value in aggregated.items():
        intervals.append((start, end, class_id, strand, gene_id,
                          value[0], value[1], value[2]))
    intervals.sort(key=lambda iv: iv[0])
    return intervals


# ---------------------------------------------------------------------------
# The overlap search itself
# ---------------------------------------------------------------------------

def sweep_overlaps(queries, intervals):
    """Find, for each query interval, every feature interval it overlaps.

    Both inputs must be sorted by start. Yields (query, hits) pairs.
    """
    heap = []
    pointer = 0
    total = len(intervals)

    for query in queries:
        q_start, q_end = query[0], query[1]

        # Admit every feature that begins before this query ends.
        while pointer < total and intervals[pointer][0] < q_end:
            iv = intervals[pointer]
            # (end, start, class, strand, gene_id, gene_name, tx, n_tx)
            heapq.heappush(heap, (iv[1], iv[0], iv[2], iv[3], iv[4],
                                  iv[5], iv[6], iv[7]))
            pointer += 1

        # Retire every feature that ended at or before this query began. The
        # heap is ordered by end coordinate, so the ones we need to drop are
        # always sitting right at the top.
        while heap and heap[0][0] <= q_start:
            heapq.heappop(heap)
          
        hits = [entry for entry in heap if entry[1] < q_end]
        yield query, hits


# ---------------------------------------------------------------------------
# Reading the query regions
# ---------------------------------------------------------------------------

def read_regions(path, point_mode):
    """Read the external BED file into a list of query records.

    Each record is (start, end, strand, original_line_index, original_fields).
    We keep the original fields and the original line order so the output can be
    written back in exactly the same order the user gave us, which makes it easy
    to paste the new column next to whatever else they have.
    """
    regions = []
    zero_length_rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            if stripped[0] == "#" or stripped.startswith(("track", "browser")):
                continue
            fields = stripped.split("\t")
            if len(fields) < 3:
                continue

            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            # Column 6 is strand in the BED spec, but this file only has four
            # columns with strand sitting in column 4, so accept either.
            if len(fields) >= 6 and fields[5] in ("+", "-"):
                strand = fields[5]
            elif len(fields) >= 4 and fields[3] in ("+", "-"):
                strand = fields[3]
            else:
                strand = "."

            if end <= start:
                # The zero length case described in the module docstring.
                zero_length_rows += 1
                if point_mode == "pos1":
                    start, end = start - 1, start
                else:
                    start, end = start, start + 1

            regions.append((chrom, start, end, strand, index, fields))

    return regions, zero_length_rows


# ---------------------------------------------------------------------------
# Main annotation driver
# ---------------------------------------------------------------------------

def annotate(regions, features_dir, require_strand_match):
    """Annotate every region. Returns a list of result dicts in input order.

    Note that strandedness is handled without doing the work twice. We collect
    every overlap regardless of strand, then decide per hit whether the strands
    agree. That lets us produce both the strand aware answer and the strand
    agnostic answer from a single pass, which is what makes the strand
    sensitivity section of the report essentially free.
    """
    # Group the query regions by chromosome, sorted by start within each.
    regions_by_chrom = defaultdict(list)
    for region in regions:
        regions_by_chrom[region[0]].append(region[1:])
    for chrom in regions_by_chrom:
        regions_by_chrom[chrom].sort(key=lambda r: (r[0], r[1]))

    # Open one streaming reader per feature file.
    readers = {
        CLASS_CDS: ChromBlockReader(os.path.join(features_dir, "coding_exons.bed")),
        CLASS_5UTR: ChromBlockReader(os.path.join(features_dir, "five_prime_utrs.bed")),
        CLASS_3UTR: ChromBlockReader(os.path.join(features_dir, "three_prime_utrs.bed")),
        CLASS_INTRON: ChromBlockReader(os.path.join(features_dir, "introns.bed")),
    }
    exon_reader = ChromBlockReader(os.path.join(features_dir, "exons.bed"))
    gene_reader = ChromBlockReader(os.path.join(features_dir, "genes.bed"))

    for class_id, reader in list(readers.items()) + [(None, exon_reader),
                                                     (None, gene_reader)]:
        if not reader.exists:
            raise SystemExit("Missing expected feature file: {}".format(reader.path))

    results = [None] * len(regions)
    string_pool = {}

    # Walk chromosomes in the same order the BED files are sorted in.
    ordered_chroms = sorted(regions_by_chrom.keys(), key=natural_chrom_key)

    for chrom in ordered_chroms:
        queries = regions_by_chrom[chrom]
        log("  {}: {:,} regions".format(chrom, len(queries)))

        blocks = {class_id: reader.take(chrom)
                  for class_id, reader in readers.items()}
        all_exon_rows = exon_reader.take(chrom)
        gene_rows = gene_reader.take(chrom)

        coding_transcripts = set()
        for fields in blocks[CLASS_CDS]:
            if len(fields) > COL_TRANSCRIPT_ID:
                coding_transcripts.add(fields[COL_TRANSCRIPT_ID])

        blocks[CLASS_NC_EXON] = [
            fields for fields in all_exon_rows
            if len(fields) > COL_TRANSCRIPT_ID
            and fields[COL_TRANSCRIPT_ID] not in coding_transcripts
        ]
        del all_exon_rows

        intervals = build_chrom_intervals(blocks, string_pool)
        del blocks

        # A separate, much smaller list for gene body context. This is what
        # lets us tell an intergenic desert apart from a spot that sits inside
        # a gene's footprint but outside every one of its transcripts.
        gene_intervals = []
        for fields in gene_rows:
            gene_intervals.append((
                int(fields[COL_START]), int(fields[COL_END]),
                fields[COL_STRAND],
                fields[COL_GENE_ID_IN_GENES] if len(fields) > COL_GENE_ID_IN_GENES else "NA",
                fields[COL_GENE_NAME_IN_GENES] if len(fields) > COL_GENE_NAME_IN_GENES else "NA",
            ))
        gene_intervals.sort(key=lambda iv: iv[0])
        del gene_rows

        # --- pass 1: transcript level features ---
        for query, hits in sweep_overlaps(queries, intervals):
            q_strand = query[2]
            original_index = query[3]

            # Split the hits by whether the strand agrees with the query.
            same_strand_hits = []
            for entry in hits:
                # entry = (end, start, class, strand, gene_id, gene_name, tx, n)
                if q_strand == "." or entry[3] == q_strand:
                    same_strand_hits.append(entry)

            chosen_hits = same_strand_hits if require_strand_match else hits

            record = summarise_hits(chosen_hits)

            other_view = summarise_hits(hits if require_strand_match
                                        else same_strand_hits)
            record["alt_strand_label"] = other_view["feature_type"]

            results[original_index] = record

        # --- pass 2: gene body context ---
        for query, hits in sweep_gene_context(queries, gene_intervals):
            original_index = query[3]
            q_strand = query[2]
            same_strand = [g for g in hits if q_strand == "." or g[2] == q_strand]
            relevant = same_strand if require_strand_match else hits
            record = results[original_index]
            if relevant:
                record["gene_context"] = ",".join(
                    sorted({g[4] for g in relevant}))
            else:
                record["gene_context"] = "intergenic"

    return results


def sweep_gene_context(queries, gene_intervals):
    """Same sweep as above but over the much smaller gene body list.

    Written as its own small function because the gene rows have a different
    tuple shape and mixing the two would make both harder to read.
    """
    heap = []
    pointer = 0
    total = len(gene_intervals)
    for query in queries:
        q_start, q_end = query[0], query[1]
        while pointer < total and gene_intervals[pointer][0] < q_end:
            iv = gene_intervals[pointer]
            heapq.heappush(heap, (iv[1], iv[0], iv[2], iv[3], iv[4]))
            pointer += 1
        while heap and heap[0][0] <= q_start:
            heapq.heappop(heap)
        yield query, [entry for entry in heap if entry[1] < q_end]


def summarise_hits(hits):
    """Collapse a list of overlapping features into one annotation record.
    """
    if not hits:
        return {
            "feature_type": UNMAPPED_LABEL,
            "feature_detail": UNMAPPED_LABEL,
            "gene_ids": "NA",
            "gene_names": "NA",
            "transcript_ids": "NA",
            "n_transcripts": 0,
            "n_genes": 0,
            "all_features_hit": UNMAPPED_LABEL,
            "gene_context": "NA",
        }

    best_class = min(entry[2] for entry in hits)
    winning = [entry for entry in hits if entry[2] == best_class]

    gene_ids = sorted({entry[4] for entry in winning})
    gene_names = sorted({entry[5] for entry in winning})
    transcript_ids = sorted({entry[6] for entry in winning})
    n_transcripts = sum(entry[7] for entry in winning)

    classes_present = sorted({entry[2] for entry in hits})
    all_hit = ";".join(CLASS_DETAIL_NAME[c] for c in classes_present)


    if len(transcript_ids) > 3:
        transcript_field = ",".join(transcript_ids[:3]) + \
            ",(+{} more)".format(len(transcript_ids) - 3)
    else:
        transcript_field = ",".join(transcript_ids)

    return {
        "feature_type": CLASS_REQUIRED_LABEL[best_class],
        "feature_detail": CLASS_DETAIL_NAME[best_class],
        "gene_ids": ",".join(gene_ids),
        "gene_names": ",".join(gene_names),
        "transcript_ids": transcript_field,
        "n_transcripts": n_transcripts,
        "n_genes": len(gene_ids),
        "all_features_hit": all_hit,
        "gene_context": "NA",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_outputs(regions, results, outdir, prefix, point_mode,
                  require_strand_match, zero_length_rows, regions_path,
                  elapsed):
    """Write the simple annotated BED, the detailed table, and the summary."""

    # --- the file the task asked for: original columns plus one label -------
    simple_path = os.path.join(outdir, "{}_annotated.bed".format(prefix))
    with open(simple_path, "w", encoding="utf-8", newline="\n") as out:
        for region, record in zip(regions, results):
            original_fields = region[5]
            out.write("\t".join(original_fields) + "\t" +
                      record["feature_type"] + "\n")

    # --- the richer table, for actually interpreting the result -------------
    detailed_path = os.path.join(outdir, "{}_annotated_detailed.tsv".format(prefix))
    header = ["chrom", "start", "end", "strand",
              "query_start_bed", "query_end_bed",
              "feature_type", "feature_detail",
              "gene_id", "gene_name", "transcript_ids",
              "n_transcripts", "n_genes", "all_features_hit", "gene_context"]
    with open(detailed_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("\t".join(header) + "\n")
        for region, record in zip(regions, results):
            chrom, q_start, q_end, strand = region[0], region[1], region[2], region[3]
            original_fields = region[5]
            out.write("\t".join([
                chrom,
                original_fields[1],
                original_fields[2],
                strand,
                str(q_start), str(q_end),
                record["feature_type"], record["feature_detail"],
                record["gene_ids"], record["gene_names"],
                record["transcript_ids"],
                str(record["n_transcripts"]), str(record["n_genes"]),
                record["all_features_hit"], record["gene_context"],
            ]) + "\n")

    # --- summary ------------------------------------------------------------
    total = len(results)
    required_counts = Counter(r["feature_type"] for r in results)
    detail_counts = Counter(r["feature_detail"] for r in results)
    combo_counts = Counter(r["all_features_hit"] for r in results)
    context_counts = Counter(r["gene_context"] if r["feature_type"] == UNMAPPED_LABEL
                             else "(mapped)" for r in results)
    strand_flips = sum(1 for r in results
                       if r["feature_type"] != r["alt_strand_label"])
    multi_gene = sum(1 for r in results if r["n_genes"] > 1)

    lines = []
    lines.append("Region to feature mapping report")
    lines.append("=" * 66)
    lines.append("Input regions      : {}".format(os.path.abspath(regions_path)))
    lines.append("Regions read       : {:,}".format(total))
    lines.append("Zero length rows   : {:,} (expanded to 1 bp, mode '{}')".format(
        zero_length_rows, point_mode))
    lines.append("Strand handling    : {}".format(
        "strand aware" if require_strand_match else "strand agnostic"))
    lines.append("Elapsed            : {:.1f} s".format(elapsed))
    lines.append("")

    lines.append("Requested feature_type column")
    lines.append("-" * 66)
    for label in ["exon", "5UTR", "3UTR", "intron", UNMAPPED_LABEL]:
        count = required_counts.get(label, 0)
        lines.append("  {:<10} {:>10,}   {:>6.2f}%".format(
            label, count, 100.0 * count / total if total else 0.0))
    lines.append("  {:<10} {:>10,}".format("TOTAL", total))
    lines.append("")

    lines.append("Finer breakdown (feature_detail column)")
    lines.append("-" * 66)
    for label in ["CDS", "5UTR", "3UTR", "noncoding_exon", "intron", UNMAPPED_LABEL]:
        count = detail_counts.get(label, 0)
        lines.append("  {:<16} {:>10,}   {:>6.2f}%".format(
            label, count, 100.0 * count / total if total else 0.0))
    lines.append("")

    lines.append("Isoform ambiguity: which class combinations were hit")
    lines.append("-" * 66)
    lines.append("  (a region hits several classes when different isoforms of a")
    lines.append("   gene, or overlapping genes, disagree about that position)")
    for combo, count in combo_counts.most_common(15):
        lines.append("  {:<44} {:>10,}".format(combo, count))
    lines.append("")
    lines.append("  regions overlapping more than one gene: {:,} ({:.2f}%)".format(
        multi_gene, 100.0 * multi_gene / total if total else 0.0))
    lines.append("")

    lines.append("Where did the unmapped (UN) regions land")
    lines.append("-" * 66)
    for context, count in context_counts.most_common(10):
        if context == "(mapped)":
            continue
        lines.append("  {:<44} {:>10,}".format(context[:44], count))
    intergenic = context_counts.get("intergenic", 0)
    un_total = required_counts.get(UNMAPPED_LABEL, 0)
    lines.append("")
    lines.append("  UN total                 {:>10,}".format(un_total))
    lines.append("  of which truly intergenic{:>10,}".format(intergenic))
    lines.append("  of which inside a gene   {:>10,}".format(un_total - intergenic))
    lines.append("  (the second group sits within a gene's footprint but outside")
    lines.append("   every one of that gene's transcripts, which happens where a")
    lines.append("   gene has two non-overlapping isoforms)")
    lines.append("")

    lines.append("Strand sensitivity")
    lines.append("-" * 66)
    lines.append("  regions whose label would change if the strand rule were")
    lines.append("  flipped: {:,} ({:.2f}%)".format(
        strand_flips, 100.0 * strand_flips / total if total else 0.0))
    lines.append("")

    lines.append("Per chromosome feature_type counts")
    lines.append("-" * 66)
    per_chrom = defaultdict(Counter)
    for region, record in zip(regions, results):
        per_chrom[region[0]][record["feature_type"]] += 1
    lines.append("  {:<26} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "chrom", "exon", "5UTR", "3UTR", "intron", "UN"))
    for chrom in sorted(per_chrom.keys(), key=natural_chrom_key):
        counter = per_chrom[chrom]
        lines.append("  {:<26} {:>8,} {:>8,} {:>8,} {:>8,} {:>8,}".format(
            chrom, counter.get("exon", 0), counter.get("5UTR", 0),
            counter.get("3UTR", 0), counter.get("intron", 0),
            counter.get(UNMAPPED_LABEL, 0)))

    # Which genes collected the most regions. A quick way to spot whether the
    # dataset is dominated by a handful of loci.
    gene_counter = Counter()
    for record in results:
        if record["gene_names"] not in ("NA", ""):
            for gene in record["gene_names"].split(","):
                gene_counter[gene] += 1
    lines.append("")
    lines.append("Top 20 genes by number of mapped regions")
    lines.append("-" * 66)
    for gene, count in gene_counter.most_common(20):
        lines.append("  {:<30} {:>8,}".format(gene, count))

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.path.join(outdir, "{}_annotation_summary.txt".format(prefix))
    with open(summary_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(summary + "\n")

    return simple_path, detailed_path, summary_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Map an external BED file onto transcriptome features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--regions", required=True,
                        help="External BED file to annotate.")
    parser.add_argument("--features", required=True,
                        help="Directory holding the BED files from script 01.")
    parser.add_argument("--outdir", required=True, help="Where to write results.")
    parser.add_argument("--prefix", default="regions",
                        help="Prefix for the output file names.")
    parser.add_argument("--point-mode", choices=["pos1", "bed0"], default="pos1",
                        help="How to expand zero length rows into one base. "
                             "pos1 treats the value as a 1-based genomic "
                             "position, bed0 treats it as a 0-based BED start. "
                             "pos1 is the default because script 03 shows T1's "
                             "coordinates are 1-based; use bed0 for a genuinely "
                             "0-based BED file.")
    parser.add_argument("--ignore-strand", action="store_true",
                        help="Match features on either strand. By default a "
                             "region only matches features on its own strand.")
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    started = time.time()

    log("Reading regions from {}".format(args.regions))
    regions, zero_length_rows = read_regions(args.regions, args.point_mode)
    log("Read {:,} regions ({:,} were zero length and were expanded to 1 bp)"
        .format(len(regions), zero_length_rows))

    log("Annotating against feature BEDs in {}".format(args.features))
    results = annotate(regions, args.features, not args.ignore_strand)

    simple_path, detailed_path, summary_path = write_outputs(
        regions, results, args.outdir, args.prefix, args.point_mode,
        not args.ignore_strand, zero_length_rows, args.regions,
        time.time() - started)

    log("Wrote {}".format(simple_path))
    log("Wrote {}".format(detailed_path))
    log("Wrote {}".format(summary_path))


if __name__ == "__main__":
    main()
