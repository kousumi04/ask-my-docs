from __future__ import annotations

from app.generation.prompts import build_prompt


def test_build_prompt_numbers_chunks_and_includes_instructions():
    chunks = [
        {"source": "queuely_docs.md", "page_number": None, "text": "E1101 means broker timeout."},
        {"source": "api_reference.pdf", "page_number": 2, "text": "Scheduling uses periodic_task."},
    ]
    messages = build_prompt("What does E1101 mean?", chunks)

    assert messages[0]["role"] == "system"
    assert "cite" in messages[0]["content"].lower()
    assert "don't know" in messages[0]["content"].lower() or "do not" in messages[0]["content"].lower()

    user_content = messages[1]["content"]
    assert "[1] (queuely_docs.md)" in user_content
    assert "[2] (api_reference.pdf, page 2)" in user_content
    assert "What does E1101 mean?" in user_content


def test_build_prompt_handles_missing_page_number():
    chunks = [{"source": "notes.txt", "page_number": None, "text": "some text"}]
    messages = build_prompt("q", chunks)
    assert "page" not in messages[1]["content"].split("[1]")[1].split("\n")[0]