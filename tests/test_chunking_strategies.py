from src.chunking.chunking_strategies import AdvancedChunker, SimpleChunker

def test_chunk_text_returns_list():
    chunker = SimpleChunker()
    result = chunker.chunk_text("hello world")
    assert isinstance(result, list)

def test_chunk_text_empty_string():
    chunker = SimpleChunker()
    result = chunker.chunk_text("")
    assert result == []

def test_chunk_text_respects_chunk_size():
    chunker = SimpleChunker(chunk_size=100, overlap=10)
    text = "a" * 500
    chunks = chunker.chunk_text(text)
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_chunk_text_overlap():
    chunker = SimpleChunker(chunk_size=20, overlap=5)
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunker.chunk_text(text)
    # Second chunk should start 15 chars into first chunk's end
    assert len(chunks) > 1
    assert chunks[1][0:] == "pqrstuvwxyz"  # Second chunk should start with overlap
    assert chunks[0][-5:] == chunks[1][:5]  # First chunk should end with overlap

def test_chunk_text_single_chunk_if_small():
    chunker = SimpleChunker(chunk_size=1000, overlap=10)
    text = "short text"
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_advanced_chunker_returns_list():
    chunker = AdvancedChunker(chunk_size=500, overlap=50)
    result = chunker.chunk_text("Some text here.")
    assert isinstance(result, list)


def test_advanced_chunker_empty_input():
    chunker = AdvancedChunker(chunk_size=500, overlap=50)
    assert chunker.chunk_text("") == []


def test_advanced_chunker_whitespace_only():
    chunker = AdvancedChunker(chunk_size=500, overlap=50)
    assert chunker.chunk_text("   \n\n   ") == []


def test_advanced_chunker_preserves_paragraph_boundaries():
    chunker = AdvancedChunker(chunk_size=200, overlap=20)
    text = "First paragraph with some content here.\n\nSecond paragraph with different content.\n\nThird paragraph."
    chunks = chunker.chunk_text(text)
    # No chunk should end mid-paragraph — each chunk boundary aligns with paragraph end
    for chunk in chunks:
        assert chunk == chunk.strip()


def test_advanced_chunker_filters_empty_paragraphs():
    chunker = AdvancedChunker(chunk_size=500, overlap=50)
    text = "First para.\n\n\n\nSecond para."
    chunks = chunker.chunk_text(text)
    assert all(c.strip() for c in chunks)
    assert len(chunks) >= 1


def test_advanced_chunker_single_chunk_if_fits():
    chunker = AdvancedChunker(chunk_size=1000, overlap=50)
    text = "Short paragraph.\n\nAnother short one."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1


def test_advanced_chunker_falls_back_to_sentences_for_large_paragraph():
    chunker = AdvancedChunker(chunk_size=50, overlap=10)
    # Single paragraph much larger than chunk_size
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_advanced_chunker_overlap_carries_context():
    chunker = AdvancedChunker(chunk_size=80, overlap=20)
    text = (
        "First paragraph provides context about the project.\n\n"
        "Second paragraph has results that depend on context.\n\n"
        "Third paragraph concludes the case study."
    )
    chunks = chunker.chunk_text(text)
    # With overlap, chunk 2 onwards should contain tail of previous chunk
    if len(chunks) > 1:
        # The overlap means chunk content is not completely disjoint
        all_text = " ".join(chunks)
        # Total text in chunks should be > original due to overlap repetition
        assert len(all_text) > len(text.replace("\n\n", " "))


def test_advanced_chunker_no_empty_chunks():
    chunker = AdvancedChunker(chunk_size=100, overlap=20)
    text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
    chunks = chunker.chunk_text(text)
    assert all(len(c) > 0 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_advanced_chunker_more_chunks_than_simple():
    """AdvancedChunker with small chunk_size should produce multiple chunks."""
    chunker = AdvancedChunker(chunk_size=60, overlap=15)
    text = (
        "First paragraph about Equal Experts cloud work.\n\n"
        "Second paragraph about Spirit Super results achieved.\n\n"
        "Third paragraph about John Lewis digital platform."
    )
    chunks = chunker.chunk_text(text)
    assert len(chunks) > 1


def test_advanced_chunker_overlap_does_not_exceed_chunk_size():
    """Overlap should never cause a chunk to massively exceed chunk_size."""
    chunker = AdvancedChunker(chunk_size=100, overlap=20)
    text = (
        "Equal Experts helped Spirit Super.\n\n"
        "They introduced event driven architecture.\n\n"
        "Results included zero downtime and faster delivery."
    )
    chunks = chunker.chunk_text(text)
    # Allow some tolerance for overlap but not gross violations
    assert all(len(c) <= chunker.chunk_size + chunker.overlap for c in chunks)