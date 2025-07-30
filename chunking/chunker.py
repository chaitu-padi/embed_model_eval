import re
from typing import List, Dict, Any, Optional
from nltk.tokenize import sent_tokenize
import nltk
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logging.warning(f"Failed to download NLTK punkt: {e}")

class TextChunker:
    """
    A class to handle different text chunking strategies.
    Supports various chunking methods: none, sentence, fixed_length, sliding_window, paragraph
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Dictionary containing chunking configuration
                - strategy: Chunking strategy to use
                - chunk_size: Size of chunks for fixed_length and sliding_window
                - overlap: Overlap size for sliding_window
                - delimiter: Custom delimiter for paragraph chunking
        """
        self.config = config
        self.strategy = config.get('strategy', 'none')
        self.chunk_size = config.get('chunk_size', 1000)
        self.overlap = config.get('overlap', 0)
        self.delimiter = config.get('delimiter', '\n\n')
        
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text using the specified strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        if not text:
            return []
            
        strategy_map = {
            'none': self._no_chunking,
            'sentence': self._sentence_chunking,
            'fixed_length': self._fixed_length_chunking,
            'sliding_window': self._sliding_window_chunking,
            'paragraph': self._paragraph_chunking
        }
        
        if self.strategy not in strategy_map:
            logging.warning(f"Unknown chunking strategy: {self.strategy}. Falling back to 'none'")
            return self._no_chunking(text)
            
        return strategy_map[self.strategy](text)
        
    def _no_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Return text as a single chunk."""
        return [{
            'content': text,
            'chunk_index': 0,
            'total_chunks': 1,
            'strategy': 'none',
            'chunk_size': len(text)
        }]
        
    def _sentence_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentence-based chunks."""
        try:
            sentences = sent_tokenize(text)
            current_chunk = []
            chunks = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += sentence_length
                
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return [
                {
                    'content': chunk,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'strategy': 'sentence',
                    'chunk_size': len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            logging.error(f"Error in sentence chunking: {e}")
            return self._no_chunking(text)
            
    def _fixed_length_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text into fixed-length chunks."""
        chunks = [text[i:i + self.chunk_size] 
                 for i in range(0, len(text), self.chunk_size)]
        
        return [
            {
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'strategy': 'fixed_length',
                'chunk_size': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]
        
    def _sliding_window_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text using sliding window with overlap."""
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk_size")
            
        stride = self.chunk_size - self.overlap
        chunks = []
        
        for i in range(0, len(text), stride):
            chunk = text[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size * 0.5:  # Skip small final chunks
                break
            chunks.append(chunk)
            
        return [
            {
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'strategy': 'sliding_window',
                'chunk_size': len(chunk),
                'overlap': self.overlap
            }
            for i, chunk in enumerate(chunks)
        ]
        
    def _paragraph_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text into paragraphs based on delimiter."""
        paragraphs = text.split(self.delimiter)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Combine small paragraphs if needed
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append(self.delimiter.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(para)
            current_length += para_length
            
        if current_chunk:
            chunks.append(self.delimiter.join(current_chunk))
            
        return [
            {
                'content': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'strategy': 'paragraph',
                'chunk_size': len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

def create_chunker(config: Dict[str, Any]) -> TextChunker:
    """Factory function to create a TextChunker instance."""
    return TextChunker(config)
