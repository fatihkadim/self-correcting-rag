import re 
import uuid
from app.schemas.retrieval import Chunk

class TextChunker: 
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def recursive_split(self, text, separators):
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            step = self.chunk_size - self.chunk_overlap
            return [text[i:i + self.chunk_size] for i in range(0, len(text), step)]        
        
        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = re.split(re.escape(separator), text)

        final_chunks = []
        for s in splits:
            if s == "":
                continue
            if len(s) <= self.chunk_size:
                final_chunks.append(s)
            else:
                recursive_result = self.recursive_split(s, next_separators)
                final_chunks.extend(recursive_result)
        return final_chunks
        
    def merge_splits(self,splits):
        merged = []
        current_chunk = ""
        for piece in splits: 
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk = current_chunk + piece
            else:
                merged.append(current_chunk)
                current_chunk = piece
        
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
        
    def add_overlap(self,chunks,):
        overlapped = [chunks[0]]  # İlk chunk olduğu gibi kalır
        for i in range(1,len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            curr_chunk = prev_chunk[-self.chunk_overlap:] + curr_chunk
            overlapped.append(curr_chunk)
        return overlapped

    def split(self, text: str, source: str) -> list[Chunk]:
        splits = self.recursive_split(text, self.separators)
        merged = self.merge_splits(splits)
        overlapped = self.add_overlap(merged)

        return [
            Chunk(id=str(uuid.uuid4()), content=chunk, source=source)
            for chunk in overlapped
        ]

    