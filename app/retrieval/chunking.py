import uuid
from app.schemas.retrieval import Chunk


class TextChunker:
    """Recursive karakter tabanlı chunker.

    Metni en kaba ayraçtan (paragraf) en inceye (karakter) doğru böler,
    parçaları ayraçlarını koruyarak `chunk_size`'ı aşmayacak şekilde
    birleştirir. Overlap, bir önceki chunk'ın son parçalarından (kelime/satır
    sınırında) oluşur; böylece chunk'lar `chunk_size`'ı hiçbir zaman aşmaz.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None):
        if chunk_size <= 0:
            raise ValueError("chunk_size pozitif olmalı")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap 0 ile chunk_size arasında olmalı")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        chunks = self._split(text, self.separators)
        return [c.strip() for c in chunks if c.strip()]

    def split(self, text: str, source: str) -> list[Chunk]:
        return [
            Chunk(id=str(uuid.uuid4()), content=chunk, source=source)
            for chunk in self.split_text(text)
        ]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Metinde geçen ilk (en kaba) ayracı seç
        separator, next_separators = separators[-1], []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, next_separators = sep, separators[i + 1:]
                break

        splits = list(text) if separator == "" else text.split(separator)

        chunks: list[str] = []
        pending: list[str] = []
        for piece in splits:
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                pending.append(piece)
                continue
            # Sığmayan parça: bekleyenleri birleştir, parçayı daha ince ayraçla böl
            if pending:
                chunks.extend(self._merge(pending, separator))
                pending = []
            if next_separators:
                chunks.extend(self._split(piece, next_separators))
            else:
                step = self.chunk_size - self.chunk_overlap
                chunks.extend(piece[i:i + self.chunk_size] for i in range(0, len(piece), step))
        if pending:
            chunks.extend(self._merge(pending, separator))
        return chunks

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        sep_len = len(separator)
        chunks: list[str] = []
        window: list[str] = []
        total = 0  # separator.join(window) uzunluğu

        for piece in pieces:
            added = len(piece) + (sep_len if window else 0)
            if window and total + added > self.chunk_size:
                chunks.append(separator.join(window))
                # Overlap için sondaki parçaları tut; yeni parça sığana kadar baştan at
                while window and (
                    total > self.chunk_overlap
                    or total + len(piece) + sep_len > self.chunk_size
                ):
                    total -= len(window[0]) + (sep_len if len(window) > 1 else 0)
                    window.pop(0)
            window.append(piece)
            total += len(piece) + (sep_len if len(window) > 1 else 0)

        if window:
            chunks.append(separator.join(window))
        return chunks
