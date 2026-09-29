import pytest
from app.retrieval.chunking import TextChunker

# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------

def _contents(chunks) -> list[str]:
    """Chunk listesinden sadece content'leri çeker."""
    return [c.content for c in chunks]


# ---------------------------------------------------------------------------
# 1. Temel bölme davranışı
# ---------------------------------------------------------------------------

def test_short_text_returns_single_chunk():
    """Chunk boyutundan kısa metin → tek parça döner."""
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split("Kısa metin.", source="test.pdf")
    assert len(chunks) == 1
    assert chunks[0].content == "Kısa metin."
    assert chunks[0].source == "test.pdf"


def test_empty_or_whitespace_text_returns_no_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("", source="empty.pdf") == []
    assert chunker.split("  \n\n  ", source="empty.pdf") == []


def test_long_text_is_split_into_multiple_chunks():
    """Uzun metin birden fazla parçaya bölünmeli."""
    text = "Kelime " * 200  # ~1400 karakter
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split(text, source="doc.pdf")
    assert len(chunks) > 1
    for c in chunks:
        assert c.source == "doc.pdf"


def test_chunk_ids_are_unique():
    """Her chunk farklı UUID almalı."""
    text = "Paragraf bir.\n\nParagraf iki.\n\nParagraf üç.\n\n" * 10
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.split(text, source="doc.pdf")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk ID'leri unique olmalı"


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=-1)


# ---------------------------------------------------------------------------
# 2. Ayraçların korunması (regresyon: kelimeler birbirine yapışıyordu)
# ---------------------------------------------------------------------------

def test_words_are_not_glued_together():
    """Kelime ayraçları korunmalı; her chunk'taki kelimeler orijinal metinde olmalı."""
    text = ("The quick brown fox jumps over the lazy dog.\n\n"
            "Second paragraph here with more words to split apart.")
    chunker = TextChunker(chunk_size=30, chunk_overlap=10)
    original_words = set(text.split())
    for chunk in chunker.split_text(text):
        for word in chunk.split():
            assert word in original_words, f"Yapışık/bozuk kelime: {word!r}"


def test_line_breaks_are_preserved_within_chunk():
    """PDF satır sonları (\\n) birleştirmede kaybolmamalı."""
    text = "\n".join(f"satir {i} icerik" for i in range(50))
    chunker = TextChunker(chunk_size=100, chunk_overlap=0)
    for chunk in chunker.split_text(text):
        assert "icerik\nsatir" in chunk or chunk.count("satir") == 1


def test_small_paragraphs_are_merged_with_separator():
    chunker = TextChunker(chunk_size=50, chunk_overlap=0)
    assert chunker.split_text("AAA\n\nBBB\n\nCCC") == ["AAA\n\nBBB\n\nCCC"]


def test_paragraphs_split_when_exceeding_chunk_size():
    chunker = TextChunker(chunk_size=10, chunk_overlap=0)
    assert chunker.split_text("AAAAAAA\n\nBBBBBBB\n\nCCCCCCC") == ["AAAAAAA", "BBBBBBB", "CCCCCCC"]


# ---------------------------------------------------------------------------
# 3. Boyut ve overlap garantileri
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size,overlap", [(30, 10), (100, 20), (1500, 200)])
def test_chunks_never_exceed_chunk_size(size, overlap):
    text = ("lorem ipsum dolor sit amet\n" * 300) + ("x" * (size * 3))
    chunker = TextChunker(chunk_size=size, chunk_overlap=overlap)
    chunks = chunker.split_text(text)
    assert chunks
    assert max(len(c) for c in chunks) <= size


def test_overlap_repeats_words_on_word_boundary():
    """Ardışık chunk'lar kelime sınırında örtüşmeli."""
    text = " ".join(f"w{i}" for i in range(100))
    chunker = TextChunker(chunk_size=40, chunk_overlap=15)
    chunks = chunker.split_text(text)
    assert len(chunks) > 1
    for prev, curr in zip(chunks, chunks[1:]):
        first_word = curr.split()[0]
        assert first_word in prev.split(), "Overlap önceki chunk'tan tam kelime içermeli"


def test_zero_overlap_has_no_repetition():
    text = " ".join(f"w{i}" for i in range(100))
    chunker = TextChunker(chunk_size=40, chunk_overlap=0)
    words = [w for c in chunker.split_text(text) for w in c.split()]
    assert words == text.split()


def test_unbreakable_text_is_hard_cut():
    """Hiç ayraç yoksa karakter bazında bölünmeli."""
    chunker = TextChunker(chunk_size=10, chunk_overlap=2, separators=[" "])
    chunks = chunker.split_text("x" * 35)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks).count("x") >= 35


# ---------------------------------------------------------------------------
# 4. End-to-end split
# ---------------------------------------------------------------------------

def test_split_end_to_end_produces_valid_chunks():
    """split() tam pipeline → Chunk nesneleri."""
    text = ("Python, Guido van Rossum tarafından geliştirilmiştir. "
            "İlk sürümü 1991 yılında yayınlandı.\n\n"
            "Python dinamik tipli bir dildir. "
            "Geniş bir standart kütüphaneye sahiptir.\n\n"
            "Python web geliştirme, veri bilimi ve yapay zeka alanlarında kullanılır.")
    chunker = TextChunker(chunk_size=80, chunk_overlap=15)
    chunks = chunker.split(text, source="python.pdf")

    assert len(chunks) >= 2
    for c in chunks:
        assert c.source == "python.pdf"
        assert c.id  # UUID mevcut
        assert 0 < len(c.content) <= 80
    assert "Guido van Rossum" in " ".join(_contents(chunks))
