import re 
from app.schemas.retrieval import Chunk

class TextChunker: 
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list[str] | None = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.seperators = seperators or ["\n\n", "\n", " ", ""]

    def recursive_split(text , seperators):
        if len(text) <= chunk_size:
            return [text]

    
        
        
    def merge_splits(splits):
        pass

    def add_overlap(chunks):
        pass

    def split(self,text:str, source: str) -> list[Chunk]:
        pass

    