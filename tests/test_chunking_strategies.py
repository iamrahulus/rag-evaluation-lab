from src.chunking.chunking_strategies import SimpleChunker

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