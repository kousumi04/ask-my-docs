from __future__ import annotations

from app.generation.citation import cited_sources, extract_citations, validate_citations


def test_extract_citations_finds_all_markers_deduplicated():
    answer = "Broker timeouts are described in [1]. Scheduling is separate [2][2]."
    assert extract_citations(answer) == {1, 2}


def test_validate_citations_all_valid():
    answer = "This is grounded [1] and also here [2]."
    check = validate_citations(answer, num_context_chunks=2)
    assert check.is_fully_grounded is True
    assert check.invalid_numbers == set()


def test_validate_citations_detects_hallucinated_reference():
    """The model was only given 2 context passages but cites [5] -- a
    citation number that doesn't exist. This must be caught, not trusted."""
    answer = "According to the docs [5], this is how it works."
    check = validate_citations(answer, num_context_chunks=2)

    assert check.is_fully_grounded is False
    assert check.invalid_numbers == {5}
    assert check.has_any_citation is True


def test_validate_citations_flags_zero_citation_answer():
    answer = "This is just an answer with no citations at all."
    check = validate_citations(answer, num_context_chunks=3)

    assert check.has_any_citation is False
    assert check.is_fully_grounded is False


def test_cited_sources_maps_back_to_chunk_metadata():
    chunks = [
        {"source": "a.md", "page_number": None, "text": "..."},
        {"source": "b.pdf", "page_number": 4, "text": "..."},
    ]
    answer = "Claim one [1], claim two [2]."
    sources = cited_sources(answer, chunks)

    assert sources == [
        {"marker": 1, "source": "a.md", "page_number": None},
        {"marker": 2, "source": "b.pdf", "page_number": 4},
    ]


def test_cited_sources_excludes_hallucinated_numbers():
    chunks = [{"source": "a.md", "page_number": None, "text": "..."}]
    answer = "Claim [1] and a fake one [9]."
    sources = cited_sources(answer, chunks)
    assert sources == [{"marker": 1, "source": "a.md", "page_number": None}]