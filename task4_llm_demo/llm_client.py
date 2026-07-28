"""
llm_client.py
=============

Thin wrapper around the Anthropic API for the file Q&A demo. Everything in
here is deliberately small: the interesting part of this task is proving the
model's answers are grounded in the uploaded file, not building an elaborate
client, so this module does exactly three things.

  ask()           answer a question, grounded in the uploaded document
  ask_baseline()  answer the SAME question with no document and no system
                  prompt at all, so the app can show the two side by side
  get_client()    build the Anthropic client from an API key

Model choice
------------
Default model is Claude Sonnet 5 (model id "claude-sonnet-5"), Anthropic's
current mid-tier model: strong quality for a document Q&A task like this one
without Opus-level cost. "claude-haiku-4-5" is offered in the app as a
faster, cheaper alternative for anyone testing on a budget. Both accept the
same request shape used here.

Why the system prompt explicitly tells the model to say "not in the
document"
-----------------------------------------------------------------------------
Grounding a model in a document is not just a matter of pasting text into the
prompt. If the model is not told to prefer the document over its own general
knowledge, it will happily answer a question from pretraining knowledge even
when the document says nothing about it, and the answer might even happen to
be correct, which would make the grounding demo meaningless (a right answer
that did not actually come from the file proves nothing). The instruction
below is what makes the "ask about something not in the file" test in the app
a real test rather than a coin flip.

Why the document text is marked cache_control: ephemeral
----------------------------------------------------------
The same document gets re-sent as part of the system prompt on every question
in a chat session, because the Anthropic API is stateless per request (there
is no server-side "remember this file" state to lean on). Marking that block
cacheable means the second, third, and later questions about the same
document only pay full input-token price for the new question and the
running chat history, not for re-sending the whole document again. On Claude
Sonnet 5 the minimum prefix Anthropic will actually cache is 1024 tokens, so
for a very short uploaded file this marker is harmless but has no effect;
for anything document-sized it is close to free to add and saves real money
across a multi-question session, which is directly relevant advice for
someone paying for this out of their own API credits.
"""

import anthropic

MODEL_DEFAULT = "claude-sonnet-5"
MODEL_FAST = "claude-haiku-4-5"

# Deliberately modest. Both models here think adaptively by default, and
# thinking tokens draw from the same max_tokens budget as the visible answer,
# so this is set higher than a bare "one paragraph of text" estimate would
# need, to leave headroom for that. A file Q&A answer has no business running
# past a couple of paragraphs, so this is a deliberate cap, not an oversight.
MAX_TOKENS_DEFAULT = 2048

SYSTEM_PROMPT_TEMPLATE = """You are answering questions about a document a user has uploaded. Answer using ONLY the information in the document below.

If the question asks about something the document does not cover, say plainly that the document does not contain that information. Do not fill the gap with your own general knowledge, and do not guess. Being wrong in a way that sounds confident is worse than saying "the document doesn't say."

When you do answer from the document, you may quote or closely paraphrase the relevant part so the user can see where the answer came from.

--- BEGIN DOCUMENT: {filename} ---
{content}
--- END DOCUMENT ---
"""


def get_client(api_key=None):
    """Build an Anthropic client.

    If api_key is given, it is used directly (this is how the app's sidebar
    fallback field reaches the SDK). Otherwise the bare constructor is used,
    which makes the SDK read ANTHROPIC_API_KEY from the environment itself,
    the same way every other Anthropic SDK example does it.
    """
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return anthropic.Anthropic()


def _extract_text(content_blocks):
    """Pull the plain text out of a response's content blocks.

    A response can contain non-text blocks too (thinking blocks, since both
    models here think adaptively by default). Only the text blocks are what
    the user should see or what should be replayed as the assistant's turn in
    the next request, which is the same "extract just the text" pattern the
    Anthropic SDK's own multi-turn conversation example uses.
    """
    return "".join(block.text for block in content_blocks if block.type == "text")


def ask(client, filename, document_text, question, history, model=MODEL_DEFAULT,
       max_tokens=MAX_TOKENS_DEFAULT):
    """Ask one question, grounded in document_text.

    `history` is the running list of {"role": ..., "content": ...} turns from
    earlier in this chat (empty on the first question). Returns
    (answer_text, updated_history, usage_dict). The caller is expected to
    store updated_history and pass it back in on the next call, the same way
    the Anthropic SDK's ConversationManager example does.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        filename=filename, content=document_text)

    messages = list(history) + [{"role": "user", "content": question}]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=messages,
    )

    answer = _extract_text(response.content)
    updated_history = messages + [{"role": "assistant", "content": answer}]

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        # Both of these are 0 on a request that did not touch the cache
        # (nothing written, nothing read), which is the normal case for a
        # document short enough to fall under Sonnet 5's 1024 token cache
        # minimum. They are still reported so the UI can show real cache
        # activity when the uploaded document is large enough to trigger it.
        "cache_creation_input_tokens": getattr(
            response.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(
            response.usage, "cache_read_input_tokens", 0) or 0,
    }
    return answer, updated_history, usage


def ask_baseline(client, question, model=MODEL_DEFAULT, max_tokens=MAX_TOKENS_DEFAULT):
    """The "without the file" control call.

    Same question, no system prompt, no document, no chat history. This
    exists purely to be shown next to the grounded answer in the app: if the
    two differ, that difference is the concrete, visual proof that the
    uploaded file is what changed the answer, not the model's own general
    knowledge.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": question}],
    )
    return _extract_text(response.content)
