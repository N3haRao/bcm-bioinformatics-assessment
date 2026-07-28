"""
app.py
======

Streamlit demo: upload a file, ask Claude natural-language questions about it,
and see proof that the answers actually come from the file rather than from
the model's own general knowledge.

Run it with (from the repository root):

    streamlit run task4_llm_demo/app.py

Streamlit reruns this entire script top to bottom every time the user
interacts with the page (uploads a file, types a message, flips a toggle).
Nothing survives between reruns except what is explicitly stashed in
st.session_state, so that dictionary is where the loaded document, the chat
transcript, and the raw API message history all live. Reading this file
top to bottom in the order things happen on screen (sidebar setup, then the
document preview, then the chat) is the easiest way to follow it.
"""

import os

import anthropic
import streamlit as st

import file_loader
import llm_client


SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

# One precise lookup question and one deliberately out-of-scope question per
# sample file. The lookup question has an answer that is verifiable against
# the raw file (proof the model actually read the data); the out-of-scope
# question asks about something the file never mentions (proof the model
# says so instead of guessing from general knowledge). Together these two
# are the core of "show that the model's answers are based on the uploaded
# data" without requiring the user to think of good questions themselves.
EXAMPLE_QUESTIONS = {
    "rnaseq_study_notes.txt": [
        "What cell line and compound were used in this study, and for how long were cells treated?",
        "Which two genes were used as reference controls, and did their expression change?",
        "What was the RNA integrity number (RIN) of the samples?",
    ],
    "differential_expression_results.csv": [
        "Which gene shows the largest magnitude change in expression, and is it up- or down-regulated?",
        "How many genes meet the study's significance criteria of padj < 0.05 AND |log2FoldChange| > 0.5?",
        "What is the IC50 of the treatment compound?",
    ],
}


def init_session_state():
    defaults = {
        "doc": None,            # the dict returned by file_loader.load_file
        "display_turns": [],     # what's rendered: list of {question, answer, baseline}
        "api_history": [],       # raw Anthropic message list, replayed on every ask()
        "loaded_source": None,   # which file (name) produced the current doc, to detect changes
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    """Renders the sidebar and returns (client_or_None, model_id, prove_grounded)."""
    st.sidebar.header("1. Claude API access")

    env_key_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if env_key_present:
        st.sidebar.success("Using ANTHROPIC_API_KEY from the environment.")
        api_key = None
    else:
        # Fallback only: typed directly into this local app by the person
        # running it, held only in Streamlit's session state, and used
        # solely to construct the Anthropic client below. It is never
        # written to disk or logged.
        api_key = st.sidebar.text_input(
            "Anthropic API key",
            type="password",
            help="Not found in the ANTHROPIC_API_KEY environment variable. "
                "You can paste it here instead; it stays in this browser "
                "session only.",
        )
        if not api_key:
            st.sidebar.warning("Enter an API key to run the demo.")

    st.sidebar.header("2. Model")
    model_label = st.sidebar.radio(
        "Model",
        options=["Claude Sonnet 5 (recommended)", "Claude Haiku 4.5 (faster, cheaper)"],
        label_visibility="collapsed",
    )
    model_id = llm_client.MODEL_DEFAULT if "Sonnet" in model_label else llm_client.MODEL_FAST

    st.sidebar.header("3. Grounding check")
    prove_grounded = st.sidebar.checkbox(
        "Also answer without the file (side by side)",
        value=True,
        help="Runs every question a second time with no document and no "
            "system prompt at all, so you can see exactly what the "
            "uploaded file changed about the answer.",
    )

    client = None
    if env_key_present or api_key:
        try:
            client = llm_client.get_client(api_key=api_key)
        except Exception as error:
            st.sidebar.error("Could not create the Anthropic client: {}".format(error))

    return client, model_id, prove_grounded


def render_file_picker():
    """Renders the file source picker and returns a (filename, raw_bytes)
    tuple, or None if nothing is selected yet."""
    st.sidebar.header("4. Choose a file")
    source = st.sidebar.radio(
        "Source", options=["Use a sample file", "Upload my own"],
        label_visibility="collapsed")

    if source == "Use a sample file":
        sample_name = st.sidebar.selectbox(
            "Sample file", options=sorted(os.listdir(SAMPLE_DIR)))
        with open(os.path.join(SAMPLE_DIR, sample_name), "rb") as handle:
            return sample_name, handle.read()

    uploaded = st.sidebar.file_uploader(
        "Upload a .txt, .csv, .tsv, or .pdf file",
        type=["txt", "csv", "tsv", "tab", "pdf"])
    if uploaded is not None:
        return uploaded.name, uploaded.getvalue()
    return None


def render_document_panel(doc):
    """The transparency panel: proves to the user what the model is actually
    seeing, rather than asking them to take it on faith."""
    st.subheader("What Claude is actually reading")

    caption_bits = ["**{}**".format(doc["filename"]), doc["kind"].upper()]
    metadata = doc["metadata"]
    if "rows" in metadata:
        caption_bits.append("{:,} rows x {} columns".format(
            metadata["rows"], metadata["columns"]))
    if "pages" in metadata:
        caption_bits.append("{} pages".format(metadata["pages"]))
    if "characters" in metadata:
        caption_bits.append("{:,} characters".format(metadata["characters"]))
    st.caption(" | ".join(caption_bits))

    if doc["is_full"]:
        st.success("The entire file was sent to the model, nothing was cut.")
    else:
        st.info(
            "This file is larger than the demo's per-question budget, so "
            "the model receives a sample plus exact summary statistics "
            "computed over the full file rather than the whole thing "
            "verbatim. See the design notes in file_loader.py for why "
            "that is still trustworthy for aggregate questions.")

    with st.expander("Show the preview Claude is working from"):
        st.text(doc["preview"])


def render_example_questions(filename):
    """Suggested questions, shown only for the two bundled sample files."""
    questions = EXAMPLE_QUESTIONS.get(filename)
    if not questions:
        return None
    st.caption("Try one of these, or type your own below:")
    columns = st.columns(len(questions))
    for column, question in zip(columns, questions):
        if column.button(question, use_container_width=True):
            return question
    return None


def render_chat_history():
    for turn in st.session_state.display_turns:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("baseline") is not None:
                with st.expander("Compare: what Claude says with NO file at all"):
                    st.write(turn["baseline"])


def handle_question(client, doc, model_id, prove_grounded, question):
    """Runs one question through the grounded call (and, if requested, the
    baseline call too), updating session state. Errors are caught here so a
    single bad request shows a message instead of crashing the whole app."""
    try:
        answer, updated_history, usage = llm_client.ask(
            client, doc["filename"], doc["model_context"], question,
            history=st.session_state.api_history, model=model_id)
        st.session_state.api_history = updated_history
    except anthropic.AuthenticationError:
        st.error("That API key was rejected. Double check it in the sidebar.")
        return
    except anthropic.RateLimitError:
        st.error("Rate limited by the Anthropic API. Wait a moment and try again.")
        return
    except anthropic.APIStatusError as error:
        st.error("Anthropic API error ({}): {}".format(error.status_code, error.message))
        return
    except anthropic.APIConnectionError:
        st.error("Could not reach the Anthropic API. Check your network connection.")
        return

    baseline = None
    if prove_grounded:
        try:
            baseline = llm_client.ask_baseline(client, question, model=model_id)
        except Exception:
            baseline = "(baseline call failed, see the grounded answer above)"

    st.session_state.display_turns.append({
        "question": question, "answer": answer, "baseline": baseline,
    })


def main():
    st.set_page_config(page_title="Ask your file (Claude demo)", layout="wide")
    init_session_state()

    st.title("Ask your file")
    st.caption(
        "Upload a .txt, .csv/.tsv, or .pdf file, ask questions about it in "
        "plain English, and see proof the answers come from your file "
        "rather than Claude's general knowledge.")

    client, model_id, prove_grounded = render_sidebar()
    picked = render_file_picker()

    if picked is None:
        st.info("Pick a sample file or upload your own from the sidebar to begin.")
        return

    filename, raw_bytes = picked
    # Reset the chat whenever the selected file actually changes, so a new
    # document does not get quizzed with questions and history left over
    # from a previous one.
    if filename != st.session_state.loaded_source:
        st.session_state.doc = file_loader.load_file(filename, raw_bytes)
        st.session_state.display_turns = []
        st.session_state.api_history = []
        st.session_state.loaded_source = filename

    doc = st.session_state.doc
    render_document_panel(doc)

    st.subheader("Ask a question")
    render_chat_history()

    clicked_question = render_example_questions(filename)
    typed_question = st.chat_input("Ask something about this file...")
    question = clicked_question or typed_question

    if question:
        if client is None:
            st.error("Add an Anthropic API key in the sidebar first.")
        else:
            with st.spinner("Asking Claude..."):
                handle_question(client, doc, model_id, prove_grounded, question)
            st.rerun()


if __name__ == "__main__":
    main()
