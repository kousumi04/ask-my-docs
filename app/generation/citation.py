"""
Parses [n] citation markers out of a generated answer and validates
them against the actual numbered context the model was given.

This is deliberately separate from prompts.py: asking the model to
cite correctly (prompts.py) and VERIFYING that it did (this module)
are two different concerns, and only the second one can actually catch
a model that ignores the instruction. A prompt is a request, not a
guarantee -- this module is what makes that request auditable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class CitationCheck:
    cited_numbers: set[int]        # every [n] found in the answer, deduplicated
    valid_numbers: set[int]        # citations that refer to a real context passage
    invalid_numbers: set[int]      # citations that DON'T -- i.e. hallucinated references
    has_any_citation: bool         # whether the answer cited anything at all
    is_fully_grounded: bool        # has_any_citation AND no invalid_numbers


def extract_citations(answer: str) -> set[int]:
    return {int(n) for n in _CITATION_RE.findall(answer)}


def validate_citations(answer: str, num_context_chunks: int) -> CitationCheck:
    cited = extract_citations(answer)
    valid_range = set(range(1, num_context_chunks + 1))

    valid = cited & valid_range
    invalid = cited - valid_range

    return CitationCheck(
        cited_numbers=cited,
        valid_numbers=valid,
        invalid_numbers=invalid,
        has_any_citation=len(cited) > 0,
        is_fully_grounded=len(cited) > 0 and len(invalid) == 0,
    )


def cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Map an answer's valid citation numbers back to their source metadata,
    for displaying 'Sources: queuely_docs.md, architecture_guide.docx' etc."""
    check = validate_citations(answer, len(chunks))
    return [
        {
            "marker": n,
            "source": chunks[n - 1]["source"],
            "page_number": chunks[n - 1].get("page_number"),
        }
        for n in sorted(check.valid_numbers)
    ]