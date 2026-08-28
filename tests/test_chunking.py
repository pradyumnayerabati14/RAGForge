import pytest

from ragforge.chunking import chunk_text


def test_chunking_is_stable_and_overlapping():
    text = " ".join(f"token-{i}" for i in range(300))
    first = chunk_text(text, chunk_size=100, overlap=20)
    second = chunk_text(text, chunk_size=100, overlap=20)
    assert len(first) > 1
    assert [c.content_hash for c in first] == [c.content_hash for c in second]
    assert all(len(c.text) <= 100 for c in first)


def test_invalid_chunk_options():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=10, overlap=10)

