"""
Curriculum-Aware Semantic Chunker for NCERT Content
====================================================

Implements intelligent chunking that respects educational boundaries,
preserves logical units, and optimizes for educational QA.
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import tiktoken

from .chunk_types import ChunkType, ChunkMetadata, CHUNK_TYPE_CHARACTERISTICS, get_chunk_type_from_structure

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a single semantic chunk with content and metadata."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/serialization."""
        return {
            'content': self.content,
            'metadata': self.metadata.to_dict()
        }


class TokenCounter:
    """
    Handles token counting for different models.
    
    Uses tiktoken for accurate OpenAI model token counting.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.
        
        Args:
            model: Model name for tokenization (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by gpt-3.5-turbo and gpt-4)
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def estimate_tokens(self, text: str) -> int:
        """Fast estimation: ~4 chars per token for English."""
        return len(text) // 4


class SemanticChunker:
    """
    Curriculum-aware semantic chunker for NCERT textbooks.
    
    Optimized for educational QA with:
    - Structure-aware chunking (respects definitions, examples, etc.)
    - Curriculum boundary preservation (never mix chapters/subjects)
    - Intelligent overlap (respects sentence boundaries)
    - Educational metadata enrichment
    """
    
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 500,
        overlap_size: int = 50,
        max_overlap: int = 80,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize semantic chunker.
        
        Args:
            min_chunk_size: Minimum tokens per chunk
            max_chunk_size: Maximum tokens per chunk
            overlap_size: Minimum overlap tokens
            max_overlap: Maximum overlap tokens
            model: Model for token counting
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_overlap = max_overlap
        
        self.token_counter = TokenCounter(model)
        
        # Statistics
        self.stats = {
            'total_chunks': 0,
            'atomic_chunks': 0,
            'split_chunks': 0,
            'avg_chunk_size': 0,
            'avg_overlap': 0
        }
    
    def chunk_document(
        self,
        cleaned_text: str,
        structures: List,  # List[EducationalBlock]
        base_metadata: Dict
    ) -> List[Chunk]:
        """
        Chunk a complete document with structure awareness.
        
        Args:
            cleaned_text: Cleaned text from text_cleaner
            structures: Educational structures from structure_recovery
            base_metadata: Base metadata (class, subject, chapter, etc.)
        
        Returns:
            List of Chunk objects with content and metadata
        """
        chunks = []
        
        # Group structures by type and priority
        atomic_structures = []
        splittable_structures = []
        
        for structure in structures:
            chunk_type = get_chunk_type_from_structure(structure.structure_type.value)
            characteristics = CHUNK_TYPE_CHARACTERISTICS.get(chunk_type, {})
            
            if characteristics.get('should_be_atomic', False):
                atomic_structures.append((structure, chunk_type))
            else:
                splittable_structures.append((structure, chunk_type))
        
        # Process atomic structures first (definitions, theorems, examples)
        for structure, chunk_type in atomic_structures:
            chunk = self._create_atomic_chunk(
                structure.content,
                chunk_type,
                base_metadata,
                structure
            )
            chunks.append(chunk)
            self.stats['atomic_chunks'] += 1
        
        # Process splittable structures (explanations, exercises)
        for structure, chunk_type in splittable_structures:
            structure_chunks = self._chunk_splittable_content(
                structure.content,
                chunk_type,
                base_metadata,
                structure
            )
            chunks.extend(structure_chunks)
            self.stats['split_chunks'] += len(structure_chunks)
        
        # Handle any text not covered by structures
        covered_lines = set()
        for structure in structures:
            covered_lines.update(range(structure.start_line, structure.end_line + 1))
        
        lines = cleaned_text.split('\n')
        uncovered_text = []
        for i, line in enumerate(lines):
            if i not in covered_lines and line.strip():
                uncovered_text.append(line)
        
        if uncovered_text:
            uncovered_chunks = self._chunk_splittable_content(
                '\n'.join(uncovered_text),
                ChunkType.CONTEXT,
                base_metadata,
                None
            )
            chunks.extend(uncovered_chunks)
        
        # Sort chunks by appearance order and assign IDs
        chunks = self._sort_and_assign_ids(chunks, base_metadata)
        
        # Update statistics
        self.stats['total_chunks'] = len(chunks)
        if chunks:
            self.stats['avg_chunk_size'] = sum(
                c.metadata.token_count for c in chunks
            ) / len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks: "
                   f"{self.stats['atomic_chunks']} atomic, "
                   f"{self.stats['split_chunks']} split")
        
        return chunks
    
    def _create_atomic_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        base_metadata: Dict,
        structure
    ) -> Chunk:
        """
        Create an atomic chunk that should not be split.
        
        Args:
            content: Text content
            chunk_type: Type of chunk
            base_metadata: Base curriculum metadata
            structure: EducationalBlock object
        
        Returns:
            Chunk object
        """
        token_count = self.token_counter.count_tokens(content)
        
        # If atomic chunk exceeds max size, log warning but keep intact
        if token_count > self.max_chunk_size:
            logger.warning(
                f"Atomic {chunk_type.value} chunk exceeds max size: "
                f"{token_count} tokens (keeping intact)"
            )
        
        metadata = self._create_metadata(
            content=content,
            chunk_type=chunk_type,
            base_metadata=base_metadata,
            structure=structure,
            token_count=token_count
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def _chunk_splittable_content(
        self,
        content: str,
        chunk_type: ChunkType,
        base_metadata: Dict,
        structure
    ) -> List[Chunk]:
        """
        Split content into chunks with overlap.
        
        Args:
            content: Text content to split
            chunk_type: Type of chunk
            base_metadata: Base curriculum metadata
            structure: EducationalBlock object (can be None)
        
        Returns:
            List of Chunk objects
        """
        # Split into sentences for intelligent chunking
        sentences = self._split_into_sentences(content)
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # If single sentence exceeds max, keep it as is
            if sentence_tokens > self.max_chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk_content = ' '.join(current_chunk)
                    chunks.append(self._create_chunk_from_content(
                        chunk_content, chunk_type, base_metadata, structure
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Add oversized sentence as its own chunk
                chunks.append(self._create_chunk_from_content(
                    sentence, chunk_type, base_metadata, structure
                ))
                i += 1
                continue
            
            # Check if adding sentence exceeds max
            if current_tokens + sentence_tokens > self.max_chunk_size:
                # Check if we have minimum chunk size
                if current_tokens >= self.min_chunk_size:
                    # Save current chunk
                    chunk_content = ' '.join(current_chunk)
                    chunks.append(self._create_chunk_from_content(
                        chunk_content, chunk_type, base_metadata, structure
                    ))
                    
                    # Create overlap
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.overlap_size, self.max_overlap
                    )
                    current_chunk = overlap_sentences
                    current_tokens = sum(
                        self.token_counter.count_tokens(s) for s in current_chunk
                    )
                else:
                    # Chunk too small, add sentence anyway
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    i += 1
                    continue
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            i += 1
        
        # Add remaining content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(self._create_chunk_from_content(
                chunk_content, chunk_type, base_metadata, structure
            ))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences intelligently.
        
        Handles abbreviations, decimals, and equation markers.
        """
        # Replace common abbreviations to protect them
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\be\.g\.', 'e<DOT>g<DOT>', text)
        text = re.sub(r'\bi\.e\.', 'i<DOT>e<DOT>', text)
        text = re.sub(r'\bvs\.', 'vs<DOT>', text)
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        min_overlap: int,
        max_overlap: int
    ) -> List[str]:
        """
        Get sentences for overlap from end of chunk.
        
        Args:
            sentences: List of sentences in current chunk
            min_overlap: Minimum overlap tokens
            max_overlap: Maximum overlap tokens
        
        Returns:
            List of sentences for overlap
        """
        overlap = []
        overlap_tokens = 0
        
        # Start from end and work backwards
        for sentence in reversed(sentences):
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            if overlap_tokens + sentence_tokens <= max_overlap:
                overlap.insert(0, sentence)
                overlap_tokens += sentence_tokens
                
                if overlap_tokens >= min_overlap:
                    break
            else:
                # Would exceed max overlap
                if overlap_tokens >= min_overlap:
                    break
                # If we haven't reached min yet, include sentence anyway
                overlap.insert(0, sentence)
                break
        
        return overlap
    
    def _create_chunk_from_content(
        self,
        content: str,
        chunk_type: ChunkType,
        base_metadata: Dict,
        structure
    ) -> Chunk:
        """Helper to create chunk from content string."""
        token_count = self.token_counter.count_tokens(content)
        
        metadata = self._create_metadata(
            content=content,
            chunk_type=chunk_type,
            base_metadata=base_metadata,
            structure=structure,
            token_count=token_count
        )
        
        return Chunk(content=content, metadata=metadata)
    
    def _create_metadata(
        self,
        content: str,
        chunk_type: ChunkType,
        base_metadata: Dict,
        structure,
        token_count: int
    ) -> ChunkMetadata:
        """
        Create comprehensive metadata for chunk.
        
        Args:
            content: Chunk content
            chunk_type: Type of chunk
            base_metadata: Base curriculum information
            structure: EducationalBlock (can be None)
            token_count: Token count
        
        Returns:
            ChunkMetadata object
        """
        # Detect content characteristics
        has_equations = self._has_equations(content)
        has_examples = chunk_type == ChunkType.EXAMPLE
        has_exercises = chunk_type in [ChunkType.EXERCISE, ChunkType.QUESTION]
        
        # Determine completeness
        if structure and hasattr(structure, 'confidence'):
            structure_confidence = structure.confidence
        else:
            structure_confidence = 1.0
        
        completeness = "complete"
        if token_count < self.min_chunk_size:
            completeness = "fragment"
        elif structure and chunk_type != ChunkType.CONTEXT:
            # Atomic structures are complete
            completeness = "complete"
        else:
            # Splittable content might be partial
            completeness = "partial"
        
        metadata = ChunkMetadata(
            class_number=base_metadata.get('class_number', 0),
            subject=base_metadata.get('subject', ''),
            chapter_number=base_metadata.get('chapter_number', 0),
            chapter_title=base_metadata.get('chapter_title', ''),
            section_number=base_metadata.get('section_number'),
            source_file=base_metadata.get('source_file', ''),
            page_numbers=base_metadata.get('page_numbers', []),
            language=base_metadata.get('language', 'eng'),
            chunk_type=chunk_type.value,
            token_count=token_count,
            char_count=len(content),
            has_equations=has_equations,
            has_examples=has_examples,
            has_exercises=has_exercises,
            structure_confidence=structure_confidence,
            completeness=completeness
        )
        
        return metadata
    
    def _has_equations(self, text: str) -> bool:
        """Detect if text contains mathematical equations."""
        equation_indicators = ['=', '≠', '≤', '≥', '∑', '∫', '√', '±', '×', '÷']
        return any(indicator in text for indicator in equation_indicators)
    
    def _sort_and_assign_ids(
        self,
        chunks: List[Chunk],
        base_metadata: Dict
    ) -> List[Chunk]:
        """
        Sort chunks and assign unique IDs.
        
        Args:
            chunks: List of chunks
            base_metadata: Base metadata for ID generation
        
        Returns:
            Sorted chunks with IDs
        """
        # ID format: class_subject_chapter_chunknum
        # Example: 10_mathematics_5_001
        
        class_num = base_metadata.get('class_number', 0)
        subject = base_metadata.get('subject', 'unknown')
        chapter = base_metadata.get('chapter_number', 0)
        
        for i, chunk in enumerate(chunks, 1):
            chunk_id = f"{class_num}_{subject}_{chapter}_{i:03d}"
            chunk.metadata.chunk_id = chunk_id
        
        return chunks
    
    def get_statistics(self) -> Dict:
        """Get chunking statistics."""
        return self.stats.copy()
