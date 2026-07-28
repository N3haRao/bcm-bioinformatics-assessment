# Task 4: simple LLM demo on uploaded files

A small Streamlit app: upload a `.txt`, `.csv`/`.tsv`, or `.pdf` file, ask
Claude questions about it in plain English, and see concrete proof that the
answers come from the file rather than from Claude's own general knowledge.

Uses the Anthropic API directly.

## Quick start

```bash
pip install -r task4_llm_demo/requirements.txt

# set your key first 
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell

streamlit run task4_llm_demo/app.py
```

If you'd rather not set an environment variable, leave it unset and start the
app anyway: the sidebar shows a password-masked field and you can paste the
key directly into the running app instead. It stays in that browser session
only, and is never written to disk or logged.

## What it does

```
task4_llm_demo/
├── README.md
├── requirements.txt        streamlit, anthropic, pypdf, pandas
├── app.py                   Streamlit UI: sidebar, chat, grounding-check toggle
├── file_loader.py            turns any of the three formats into what Claude reads
├── llm_client.py              the Anthropic API calls
└── sample_data/
    ├── rnaseq_study_notes.txt          a mock RNA-seq DE study writeup
    └── differential_expression_results.csv   its accompanying results table
```

The two bundled sample files form one coherent (invented) study, so the demo
works immediately without needing to find your own test file: `A549` lung
adenocarcinoma cells treated with a fictional compound ("BCM-119") vs a DMSO
control, analyzed with a standard STAR + featureCounts + DESeq2 pipeline. The
gene identities and coordinates in the CSV are **real GENCODE v33 values**,
reused directly from Task 1's `genes.bed` output, for a curated panel of 28
well known genes (EGFR/MAPK/PI3K-AKT pathway members, common oncogenes and
tumor suppressors, two standard reference genes, and a couple of
tissue-irrelevant genes included as a sanity check). The expression values
themselves, and every experimental detail in the notes file, are invented for
this demo.

Three suggested questions per file:

1. **A precise lookup question** whose answer only exists in the file and is
   verifiable against the raw data (proof the model actually read it). One of
   these is deliberately layered: "how many genes meet the study's
   significance criteria of padj < 0.05 AND |log2FoldChange| > 0.5" has a
   different answer (6) than the simpler "how many genes have padj < 0.05"
   (7), since one gene clears the p-value bar but not the fold-change bar.
   Getting this right tests whether the model is actually reading the
   document's stated criteria rather than pattern-matching to the more
   common single-threshold version of this question.
2. **A verifiable factual question** about the study design (cell line,
   compound, treatment duration; which two genes served as reference
   controls).
3. **An out-of-scope control question** about something the file never
   mentions (an IC50 value, an RNA integrity number) to prove the model says
   so instead of guessing from pretraining knowledge, which real gene names
   like EGFR and TP53 make an easier trap to fall into than a fictional
   company name would.

## How grounding is actually shown, not just claimed

Three separate things in this app work together to make "the answer came
from the file" a checked fact rather than an assertion:

1. **The system prompt instructs the model explicitly** to answer only from
   the document and to say plainly when something is not covered, rather
   than filling the gap with general knowledge (`llm_client.py`,
   `SYSTEM_PROMPT_TEMPLATE`). Without this instruction the model would happily
   answer an out-of-scope question from pretraining knowledge, sometimes even
   correctly, which would make the whole exercise unfalsifiable.
2. **A transparency panel** in the app shows exactly what text was sent to
   the model, not just a claim that "the file was loaded" (`app.py`,
   `render_document_panel`). If the loader had a bug that mangled the file,
   this is where it would be visible.
3. **A "prove it's grounded" toggle** runs every question a second time with
   no document and no system prompt at all, and shows both answers side by
   side. When the two differ, that difference is the concrete evidence that
   the file changed the answer.

## The loader's size problem, and how it is handled

A `.txt` file's content can simply be embedded whole: there is no structure to
lose. A `.csv`/`.tsv` table or a `.pdf` is different, because past a certain
size the whole thing cannot reasonably be pasted into the prompt. Truncating
silently would let the model confidently answer questions about rows or pages
it never saw, which defeats the entire point of this demo.

## Prompt caching

The uploaded document gets re-sent as part of the system prompt on every
question in a chat, because the Anthropic API has no server-side "remember
this file" state between requests. The system prompt block is marked
`cache_control: {"type": "ephemeral"}` so that the second and later questions
in a session only pay full price for the new question, not for re-sending the
whole document again. On Claude Sonnet 5 the minimum prefix Anthropic will
actually cache is 1024 tokens, so this has no effect on the two small bundled
samples, but costs nothing to include and pays off on any real document-sized
upload, which is worth doing given the demo runs on the user's own API
credits.

## Verification performed

- `file_loader.py` tested directly against both bundled sample files: the
  28-row differential expression CSV and the study notes text both load as
  complete, untruncated (`is_full: True`).
- The gene coordinates in the CSV were cross-checked against Task 1's own
  `genes.bed` output (`grep`-style lookup by gene symbol), confirming they
  match the real GENCODE v33 annotation rather than being independently
  invented.
- The intended answers to the bundled example questions were computed
  directly from the CSV with plain Python/`csv`, independent of the app, to
  confirm they are unambiguous: 7 genes have `padj < 0.05`, but only 6 also
  satisfy `|log2FoldChange| > 0.5` (STAT3 clears the p-value bar but not the
  fold-change bar), and MYC is unambiguously the single largest magnitude
  change (-1.38, versus the next largest at -0.95).
- The large-table fallback path was exercised against a synthetic 2,000-row
  CSV: confirmed `is_full: False`, and confirmed the exact summary statistics
  (mean, std, min, max) are computed over all 2,000 rows even though only 50
  rows are shown to the model verbatim.
- All three modules byte-compile cleanly (`python -m py_compile`).
- The app was launched live and inspected via the accessibility tree: the
  sidebar correctly shows the password-masked API key field (with a warning
  when no key is present), the model choice radio buttons, the grounding-check
  toggle, and the file source picker; the main panel correctly shows the
  loaded CSV's row/column count, the "entire file was sent, nothing was cut"
  confirmation, and all three suggested questions for that file.
- Not exercised in this session: an actual Claude API call (no API key was
  available in the sandbox this was built in). The system prompt, message
  history handling, and usage/cache-field parsing in `llm_client.py` follow
  the Anthropic Python SDK's own documented request and response shapes
  directly, but you should run at least one live question after adding your
  key to confirm the end-to-end call, which is exactly what the "prove it's
  grounded" toggle is there to make convincing to look at.
