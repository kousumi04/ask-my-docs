"""
Prompt construction for citation-enforced, grounded generation.

THE CORE ANTI-HALLUCINATION TECHNIQUE: number every retrieved chunk in
the context and instruct the model to cite the number(s) supporting
each claim, inline, using [n] markers. This does two things at once:

  1. It gives the model an easy, low-effort mechanical action (copy a
     bracketed number) rather than an open-ended one -- models are
     measurably more likely to comply with "cite [n] after each claim"
     than with vaguer instructions like "be accurate" or "don't make
     things up," which have no concrete action attached.
  2. It makes the output machine-verifiable after the fact (Phase 8):
     every [n] in the answer can be checked against the actual
     numbered context, so a hallucinated citation (a number that
     doesn't exist, or doesn't support the claim) becomes detectable
     rather than just an unfalsifiable claim of "the model was right."

The second half of the prompt -- explicit permission to say "I don't
know" -- matters more than it looks. Without it, a model under
instruction-following pressure to "answer the question" will often
generate a plausible-sounding but ungrounded answer rather than push
back on an unanswerable question. Giving an explicit, named escape
hatch reduces that pressure.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a documentation assistant that answers questions using ONLY the numbered context passages provided. Follow these rules strictly:

1. Every factual claim in your answer must be immediately followed by a citation marker like [1] or [2][3] referencing the passage(s) that support it.
2. Only cite passage numbers that actually appear in the context below. Never invent a citation number.
3. If the context does not contain enough information to answer the question, say so explicitly -- do not guess, and do not answer from general knowledge outside the provided passages.
4. Do not repeat the passages verbatim at length; synthesize an answer and cite your sources."""


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        location = f"{chunk['source']}"
        if chunk.get("page_number"):
            location += f", page {chunk['page_number']}"
        lines.append(f"[{i}] ({location})\n{chunk['text']}")
    return "\n\n".join(lines)


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Returns a chat-style message list: [{role, content}, ...], the
    format both Groq's and Ollama's chat APIs expect."""
    context = _format_context(chunks)
    user_content = f"""Context passages:

{context}

Question: {query}

Answer the question using only the passages above, with citation markers like [1] after each claim."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]