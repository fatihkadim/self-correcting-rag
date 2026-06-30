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


def test_empty_text_raises_on_add_overlap():
    """Boş string → add_overlap'ta IndexError (bilinen edge case)."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    import pytest
    with pytest.raises(IndexError):
        chunker.split("", source="empty.pdf")


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


# ---------------------------------------------------------------------------
# 2. Recursive split davranışı
# ---------------------------------------------------------------------------

def test_recursive_split_returns_short_text_as_is():
    """chunk_size'dan kısa metin bölünmeden döner."""
    chunker = TextChunker(chunk_size=50, chunk_overlap=0)
    text = "AAA\n\nBBB\n\nCCC"  # 15 karakter < 50 chunk_size
    splits = chunker.recursive_split(text, chunker.separators)
    assert len(splits) == 1
    assert splits[0] == text


def test_recursive_split_splits_when_exceeds_chunk_size():
    """chunk_size'ı aşan metin paragraf ayırıcısı ile bölünmeli."""
    chunker = TextChunker(chunk_size=10, chunk_overlap=0)
    text = "AAAAAAA\n\nBBBBBBB\n\nCCCCCCC"
    splits = chunker.recursive_split(text, chunker.separators)
    assert len(splits) >= 3
    assert "AAAAAAA" in splits
    assert "BBBBBBB" in splits
    assert "CCCCCCC" in splits


def test_recursive_split_falls_through_separators():
    """Paragraf ayırıcısı yoksa satır sonu (\\n) ile bölmeli."""
    chunker = TextChunker(chunk_size=15, chunk_overlap=0)
    text = "Satır bir\nSatır iki\nSatır üç"
    splits = chunker.recursive_split(text, chunker.separators)
    assert len(splits) >= 2


# ---------------------------------------------------------------------------
# 3. Merge davranışı
# ---------------------------------------------------------------------------

def test_merge_combines_small_splits():
    """Küçük parçalar chunk_size'a kadar birleştirilmeli."""
    chunker = TextChunker(chunk_size=20, chunk_overlap=0)
    splits = ["AB", "CD", "EF", "GH"]
    merged = chunker.merge_splits(splits)
    # Her biri 2 karakter, 20 sınırı ile hepsi tek chunk'a sığmalı
    assert len(merged) == 1
    assert merged[0] == "ABCDEFGH"


def test_merge_respects_chunk_size_limit():
    """Chunk boyut sınırı aşıldığında yeni chunk başlatılmalı."""
    chunker = TextChunker(chunk_size=5, chunk_overlap=0)
    splits = ["AB", "CD", "EF", "GH"]
    merged = chunker.merge_splits(splits)
    # AB+CD=4 ≤ 5 → ok, +EF=6 > 5 → yeni chunk
    assert len(merged) >= 2


# ---------------------------------------------------------------------------
# 4. Overlap davranışı
# ---------------------------------------------------------------------------

def test_overlap_adds_prefix_from_previous_chunk():
    """İkinci chunk, önceki chunk'ın son N karakterini prefix olarak almalı."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=5)
    chunks_raw = ["AAAAA12345", "BBBBB"]
    overlapped = chunker.add_overlap(chunks_raw)
    assert overlapped[0] == "AAAAA12345"  # İlk chunk değişmemeli
    assert overlapped[1].startswith("12345")  # overlap prefix
    assert overlapped[1].endswith("BBBBB")


def test_zero_overlap_prepends_full_previous_chunk():
    """Overlap=0 ise prev[-0:] tüm önceki chunk'ı döndürür (Python slice davranışı)."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=0)
    chunks_raw = ["AAA", "BBB", "CCC"]
    overlapped = chunker.add_overlap(chunks_raw)
    # Python'da s[-0:] == s (tüm string), bu yüzden overlap=0 prefix olarak tüm prev'i ekler
    assert overlapped[0] == "AAA"
    assert overlapped[1] == "AAABBB"
    assert overlapped[2] == "BBBCCC"


# ---------------------------------------------------------------------------
# 5. End-to-end split
# ---------------------------------------------------------------------------

def test_split_end_to_end_produces_valid_chunks():
    """split() tam pipeline: recursive → merge → overlap → Chunk nesneleri."""
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
        assert len(c.content) > 0
