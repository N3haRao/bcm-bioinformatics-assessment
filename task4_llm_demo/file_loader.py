"""
file_loader.py
===============

Turns a raw uploaded file (.txt, .csv, .tsv, or optionally .pdf) into two
different things:

  model_context   the text that actually gets sent to Claude as "the document"
  preview          a shorter version shown to the user in the app, so they can
                   see for themselves what the model is actually working from

Showing the user the preview is not a cosmetic nicety. The whole point of this
demo is proving the model's answers come from the uploaded file rather than
its own general knowledge, and the easiest way to undercut that proof by
accident is to silently feed the model something different from what the user
thinks they uploaded (a truncated file, a mis-parsed table). Showing the exact
model_context, or an honest description of how it was reduced, closes that
gap.

The size problem, and how it is handled
----------------------------------------
A .txt file's content can simply be embedded in full: there is no structure to
lose. A .csv/.tsv table or a .pdf is different, because past a certain size we
cannot reasonably paste the whole thing into the prompt. Two options exist:
truncate silently (bad: the model will confidently answer questions about rows
or pages it never saw), or truncate and say so (better, but the model still
cannot answer whole-table questions).

For tables specifically there is a third option that is strictly better than
either: keep a small sample of rows for the model to read verbatim, but also
compute exact summary statistics (counts, means, min/max, most common
categories) over the WHOLE table with pandas, and hand those over as text
too. That way a question like "what is the average salary" or "how many
people are in Sales" can still be answered exactly even when the model never
saw every row, because the arithmetic was done by pandas on the full data, not
guessed by the model from a sample. This is the same "small vs large input"
judgment call already made for the T2 expression matrix in Task 2, applied
here to whatever table the user uploads.

PDFs get the simpler version of the same idea (full text if it fits, a
truncated prefix with an honest page count if it does not), since there is no
equivalent of "summary statistics" for free-form document text.
"""

import csv
import io

import pandas as pd


# Rough character budget for "small enough to embed verbatim". This is not
# tuned to a token count precisely, it is a conservative estimate (roughly
# 4 characters per token for English-like text) chosen so that even a fairly
# talkative document stays comfortably inside the context window alongside
# the system prompt wrapper and the running chat history.
CHAR_BUDGET = 15000

# How many sample rows / pages to show verbatim when a table or PDF is too
# big to embed in full.
SAMPLE_ROWS = 50
SAMPLE_PAGES = 20


def load_file(filename, raw_bytes):
    """Dispatch to the right loader based on the file extension.

    Returns a dict with:
        filename      the original name, unchanged
        kind          "txt" | "csv" | "tsv" | "pdf"
        model_context the text handed to Claude as the document
        preview       a shorter version shown to the user in the app
        is_full       True if model_context is the complete file, False if it
                      is a reduced sample plus summary
        metadata      a small dict of facts about the file (row/column counts,
                      page counts, character counts), used in the UI caption
    """
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        return _load_table(filename, raw_bytes, "csv", delimiter=",")
    if lower_name.endswith((".tsv", ".tab")):
        return _load_table(filename, raw_bytes, "tsv", delimiter="\t")
    if lower_name.endswith(".pdf"):
        return _load_pdf(filename, raw_bytes)
    # Anything else, including a bare ".txt", is treated as plain text. This
    # is also the fallback for an unrecognised extension, on the theory that
    # showing the user readable text they can sanity check beats refusing
    # to load the file at all.
    return _load_text(filename, raw_bytes)


def _decode(raw_bytes):
    """Decode bytes to text, tolerating the odd non-UTF-8 byte rather than
    crashing on it. A demo script should not fail on a stray smart quote."""
    return raw_bytes.decode("utf-8", errors="replace")


def _load_text(filename, raw_bytes):
    text = _decode(raw_bytes)
    is_full = len(text) <= CHAR_BUDGET
    model_context = text if is_full else (
        text[:CHAR_BUDGET] +
        "\n\n[TRUNCATED: this text file is {:,} characters long; only the "
        "first {:,} characters above were included. Treat anything past "
        "that point as unknown rather than guessing.]".format(
            len(text), CHAR_BUDGET)
    )
    preview = text[:2000] + ("..." if len(text) > 2000 else "")
    return {
        "filename": filename,
        "kind": "txt",
        "model_context": model_context,
        "preview": preview,
        "is_full": is_full,
        "metadata": {"characters": len(text)},
    }


def _sniff_delimiter(raw_text, default_delimiter):
    """Best-effort delimiter sniff, falling back to the extension's default.

    Guards against the common case of a ".csv" file that was actually saved
    tab-separated (or vice versa) by a spreadsheet program set to the wrong
    export format. If sniffing fails for any reason, the extension-implied
    delimiter is used, which is still correct the overwhelming majority of
    the time.
    """
    try:
        sample = raw_text[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return default_delimiter


def _load_table(filename, raw_bytes, kind, delimiter):
    raw_text = _decode(raw_bytes)
    delimiter = _sniff_delimiter(raw_text, delimiter)

    try:
        dataframe = pd.read_csv(io.StringIO(raw_text), sep=delimiter)
    except Exception as error:
        # A malformed table (ragged rows, a stray header, binary garbage) is
        # still worth showing the model as plain text rather than crashing
        # the app outright. Fall back to the text loader and say why.
        fallback = _load_text(filename, raw_bytes)
        fallback["kind"] = kind
        fallback["metadata"]["parse_error"] = str(error)
        return fallback

    row_count, col_count = dataframe.shape
    is_full = len(raw_text) <= CHAR_BUDGET

    if is_full:
        model_context = (
            "The full {} table ({} rows, {} columns) follows, exactly as "
            "uploaded:\n\n{}".format(kind.upper(), row_count, col_count, raw_text)
        )
    else:
        sample = dataframe.head(SAMPLE_ROWS)
        summary = _summarize_dataframe(dataframe)
        model_context = (
            "This {} table has {} rows and {} columns, which is too large to "
            "include in full. Below is a verbatim sample of the first {} "
            "rows, followed by summary statistics computed over ALL {} rows "
            "(not just the sample shown), so you can answer questions about "
            "counts, totals, and averages exactly. For anything that depends "
            "on a specific row outside the sample and outside these "
            "summaries, say plainly that you cannot verify it from a "
            "preview rather than guessing.\n\n"
            "--- sample of first {} rows ---\n{}\n\n"
            "--- summary statistics over the full table ---\n{}".format(
                kind.upper(), row_count, col_count, len(sample), row_count,
                len(sample), sample.to_csv(index=False), summary)
        )

    preview_rows = dataframe.head(10)
    preview = "{} rows x {} columns\n\n{}".format(
        row_count, col_count, preview_rows.to_string(index=False))

    return {
        "filename": filename,
        "kind": kind,
        "model_context": model_context,
        "preview": preview,
        "is_full": is_full,
        "metadata": {"rows": row_count, "columns": col_count,
                    "column_names": list(dataframe.columns)},
    }


def _summarize_dataframe(dataframe):
    """Exact summary statistics over the whole table, not just a sample.

    Numeric columns get pandas' own describe() (count, mean, std, min,
    quartiles, max). Everything else gets a count of distinct values and the
    most common ones with their exact counts, so questions like "how many
    departments are there" or "what is the most common role" can be answered
    precisely without the model ever seeing every row.
    """
    lines = []
    numeric_columns = dataframe.select_dtypes(include="number").columns
    if len(numeric_columns) > 0:
        lines.append("Numeric columns (exact, over all rows):")
        lines.append(dataframe[numeric_columns].describe().to_string())

    other_columns = dataframe.select_dtypes(exclude="number").columns
    for column in other_columns:
        value_counts = dataframe[column].value_counts()
        top_values = ", ".join(
            "{!r}: {}".format(value, count)
            for value, count in value_counts.head(8).items()
        )
        lines.append(
            "\nColumn '{}': {} distinct values across all rows. Most common: "
            "{}".format(column, dataframe[column].nunique(), top_values)
        )
    return "\n".join(lines)


def _load_pdf(filename, raw_bytes):
    # Imported lazily so that a machine with no PDF in sight never needs
    # pypdf installed to run the txt/csv path of this demo.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    total_pages = len(page_texts)

    full_text = "\n\n".join(
        "--- page {} ---\n{}".format(index + 1, text)
        for index, text in enumerate(page_texts)
    )
    is_full = len(full_text) <= CHAR_BUDGET

    if is_full:
        model_context = (
            "The full PDF ({} pages) follows, extracted as plain text:\n\n{}"
            .format(total_pages, full_text)
        )
    else:
        included_pages = page_texts[:SAMPLE_PAGES]
        included_text = "\n\n".join(
            "--- page {} ---\n{}".format(index + 1, text)
            for index, text in enumerate(included_pages)
        )
        model_context = (
            "This PDF has {} pages, which is too much text to include in "
            "full. Only the first {} pages are included below. Treat any "
            "later page as unknown rather than guessing at its "
            "contents.\n\n{}".format(total_pages, len(included_pages),
                                    included_text)
        )

    preview = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
    return {
        "filename": filename,
        "kind": "pdf",
        "model_context": model_context,
        "preview": preview,
        "is_full": is_full,
        "metadata": {"pages": total_pages},
    }
