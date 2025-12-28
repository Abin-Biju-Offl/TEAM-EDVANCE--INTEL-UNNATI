"""
Multi-Stage Retrieval Pipeline
===============================

Three-stage retrieval optimized for educational QA:

Stage 1: Metadata Filtering
  - Filter by class, subject, chapter, language
  - Ensures grade-appropriate content
  - Reduces search space by 90-95%

Stage 2: Vector Similarity Search
  - Semantic search using FAISS
  - Top-k = 10-15 candidates
  - Fast approximate matches

Stage 3: Cross-Encoder Reranking
  - Deep semantic relevance scoring
  - Rerank top-k to top-5
  - High accuracy, slower than bi-encoder

Design Rationale:
- Stage 1 is fast O(n) metadata scan
- Stage 2 leverages FAISS for speed
- Stage 3 only processes 10-15 candidates (expensive but accurate)
- Combined: Fast + accurate + curriculum-aware
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from ..embeddings import EmbeddingGenerator, VectorStore
from ..embeddings.metadata_filter import MetadataFilter, FilterBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    
    # Stage 1: Metadata filtering
    enable_metadata_filtering: bool = True
    default_class_filter: Optional[int] = None
    default_subject_filter: Optional[str] = None
    default_language_filter: str = "eng"
    
    # Stage 2: Vector search
    initial_k: int = 15              # Retrieve more candidates for reranking
    min_similarity_threshold: float = 0.5  # Discard very low similarity
    
    # Stage 3: Reranking
    enable_reranking: bool = True
    final_k: int = 5                 # Return top-5 after reranking
    
    # Confidence scoring
    enable_confidence_scoring: bool = True
    low_confidence_threshold: float = 0.6
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    """
    A single retrieval result with metadata and scores.
    
    Attributes:
        chunk_id: Unique chunk identifier
        content: Chunk text content
        metadata: Full chunk metadata dict
        similarity_score: Cosine similarity (0-1, from FAISS)
        rerank_score: Cross-encoder score (0-1, if reranking enabled)
        final_score: Combined score used for final ranking
        confidence: Confidence estimate (0-1)
        rank: Final rank position (1-based)
    """
    
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Scores
    similarity_score: float
    rerank_score: Optional[float] = None
    final_score: float = 0.0
    confidence: float = 0.0
    
    # Position
    rank: int = 0
    
    # Flags
    is_low_confidence: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, chunk_id={self.chunk_id}, "
            f"final_score={self.final_score:.3f}, confidence={self.confidence:.3f})"
        )


class RetrievalPipeline:
    """
    Multi-stage retrieval pipeline for NCERT content.
    
    Pipeline:
        Query → Metadata Filter → Vector Search → Rerank → Confidence Score → Results
    
    Example:
        >>> pipeline = RetrievalPipeline(vector_store, embedding_generator)
        >>> results = pipeline.retrieve(
        ...     query="What is an arithmetic progression?",
        ...     class_number=10,
        ...     subject="mathematics"
        ... )
        >>> for result in results:
        ...     print(f"{result.rank}. {result.content[:100]}... (score: {result.final_score:.3f})")
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retrieval pipeline.
        
        Args:
            vector_store: Loaded VectorStore with embeddings
            embedding_generator: EmbeddingGenerator for query encoding
            config: Pipeline configuration
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or RetrievalConfig()
        
        # Initialize reranker and confidence scorer (lazy loading)
        self._reranker = None
        self._confidence_scorer = None
        
        logger.info("RetrievalPipeline initialized")
        logger.info(f"Config: {self.config}")
    
    @property
    def reranker(self):
        """Lazy load reranker (heavy model)."""
        if self._reranker is None and self.config.enable_reranking:
            from .reranker import CrossEncoderReranker
            logger.info("Loading cross-encoder reranker...")
            self._reranker = CrossEncoderReranker()
        return self._reranker
    
    @property
    def confidence_scorer(self):
        """Lazy load confidence scorer."""
        if self._confidence_scorer is None and self.config.enable_confidence_scoring:
            from .confidence_scorer import ConfidenceScorer
            self._confidence_scorer = ConfidenceScorer()
        return self._confidence_scorer
    
    def retrieve(
        self,
        query: str,
        class_number: Optional[int] = None,
        subject: Optional[str] = None,
        chapter_number: Optional[int] = None,
        language: Optional[str] = None,
        chunk_types: Optional[List[str]] = None,
        custom_filter: Optional[MetadataFilter] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for query using multi-stage pipeline.
        
        Args:
            query: User query text
            class_number: Filter by class (e.g., 10)
            subject: Filter by subject (e.g., "mathematics")
            chapter_number: Filter by chapter (optional)
            language: Filter by language (default: "eng")
            chunk_types: Filter by chunk types (e.g., ["definition", "example"])
            custom_filter: Custom MetadataFilter (overrides other filters)
        
        Returns:
            List of RetrievalResult objects, ranked by relevance
        """
        logger.info(f"Retrieving for query: '{query[:100]}...'")
        
        # Stage 1: Metadata Filtering
        filtered_ids = self._stage1_metadata_filtering(
            class_number, subject, chapter_number, language, chunk_types, custom_filter
        )
        
        if not filtered_ids:
            logger.warning("No chunks match metadata filters!")
            return []
        
        logger.info(f"Stage 1: Filtered to {len(filtered_ids)} chunks")
        
        # Stage 2: Vector Similarity Search
        candidates = self._stage2_vector_search(query, filtered_ids)
        
        if not candidates:
            logger.warning("No candidates found in vector search!")
            return []
        
        logger.info(f"Stage 2: Retrieved {len(candidates)} candidates")
        
        # Stage 3: Cross-Encoder Reranking
        if self.config.enable_reranking and len(candidates) > 1:
            reranked = self._stage3_reranking(query, candidates)
            logger.info(f"Stage 3: Reranked to top {len(reranked)}")
        else:
            reranked = candidates[:self.config.final_k]
        
        # Confidence Scoring
        if self.config.enable_confidence_scoring:
            results = self._compute_confidence(query, reranked)
        else:
            results = reranked
        
        # Assign final ranks
        for i, result in enumerate(results, 1):
            result.rank = i
        
        logger.info(f"Retrieval complete: {len(results)} results")
        
        return results
    
    def _stage1_metadata_filtering(
        self,
        class_number: Optional[int],
        subject: Optional[str],
        chapter_number: Optional[int],
        language: Optional[str],
        chunk_types: Optional[List[str]],
        custom_filter: Optional[MetadataFilter]
    ) -> List[int]:
        """
        Stage 1: Filter chunks by metadata.
        
        Returns:
            List of vector IDs that match filters
        """
        if not self.config.enable_metadata_filtering and custom_filter is None:
            # No filtering - return all IDs
            return list(self.vector_store.metadata_store.keys())
        
        if custom_filter:
            # Use custom filter
            return custom_filter.apply(self.vector_store.metadata_store)
        
        # Build filter
        filter_builder = FilterBuilder()
        
        # Apply defaults if not specified
        class_number = class_number or self.config.default_class_filter
        subject = subject or self.config.default_subject_filter
        language = language or self.config.default_language_filter
        
        if class_number:
            filter_builder.for_class(class_number)
        
        if subject:
            filter_builder.for_subject(subject)
        
        if chapter_number:
            filter_builder.for_chapter(chapter_number)
        
        if language:
            filter_builder.with_language(language)
        
        if chunk_types:
            filter_builder.with_chunk_type(chunk_types)
        
        metadata_filter = filter_builder.build()
        
        logger.info(f"Metadata filter: {metadata_filter.get_filter_summary()}")
        
        return metadata_filter.apply(self.vector_store.metadata_store)
    
    def _stage2_vector_search(
        self,
        query: str,
        filtered_ids: List[int]
    ) -> List[RetrievalResult]:
        """
        Stage 2: Vector similarity search on filtered subset.
        
        Returns:
            List of RetrievalResult objects with similarity scores
        """
        # Embed query
        query_embedding = self.embedding_generator.embed_text(query)
        
        # Search in filtered subset
        # Note: Current VectorStore.search() searches all vectors
        # For production, implement search_subset() that only searches filtered_ids
        # For now, search all and filter results
        
        distances, indices, metadata_list = self.vector_store.search(
            query_embedding,
            k=min(self.config.initial_k * 2, len(filtered_ids)),  # Get more for filtering
            return_metadata=True
        )
        
        # Filter to only include filtered_ids and above threshold
        candidates = []
        for dist, idx, metadata in zip(distances, indices, metadata_list):
            if idx not in filtered_ids:
                continue
            
            # Cosine similarity from inner product (normalized vectors)
            similarity = float(dist)
            
            if similarity < self.config.min_similarity_threshold:
                continue
            
            result = RetrievalResult(
                chunk_id=metadata['chunk_id'],
                content=metadata['content'],
                metadata=metadata,
                similarity_score=similarity,
                final_score=similarity
            )
            
            candidates.append(result)
            
            if len(candidates) >= self.config.initial_k:
                break
        
        return candidates
    
    def _stage3_reranking(
        self,
        query: str,
        candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Stage 3: Cross-encoder reranking for accuracy.
        
        Returns:
            Reranked and trimmed list of RetrievalResult objects
        """
        if self.reranker is None:
            logger.warning("Reranker not available, skipping reranking")
            return candidates[:self.config.final_k]
        
        # Extract texts for reranking
        query_doc_pairs = [(query, result.content) for result in candidates]
        
        # Get reranking scores
        rerank_scores = self.reranker.score(query_doc_pairs)
        
        # Update results with rerank scores
        for result, score in zip(candidates, rerank_scores):
            result.rerank_score = float(score)
            
            # Combine similarity and rerank scores
            # Weight: 40% similarity (fast), 60% rerank (accurate)
            result.final_score = 0.4 * result.similarity_score + 0.6 * result.rerank_score
        
        # Sort by final score
        reranked = sorted(candidates, key=lambda r: r.final_score, reverse=True)
        
        # Return top-k
        return reranked[:self.config.final_k]
    
    def _compute_confidence(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Compute confidence scores for results.
        
        Returns:
            Results with confidence scores updated
        """
        if self.confidence_scorer is None:
            logger.warning("Confidence scorer not available, skipping")
            return results
        
        # Compute confidence for each result
        for result in results:
            confidence = self.confidence_scorer.compute_confidence(
                query=query,
                retrieved_content=result.content,
                similarity_score=result.similarity_score,
                rerank_score=result.rerank_score,
                metadata=result.metadata
            )
            
            result.confidence = confidence
            result.is_low_confidence = confidence < self.config.low_confidence_threshold
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'vector_store_size': self.vector_store.vector_count,
            'config': self.config.to_dict(),
            'reranker_loaded': self._reranker is not None,
            'confidence_scorer_loaded': self._confidence_scorer is not None
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing RetrievalPipeline...")
    print("=" * 70)
    
    # This is a placeholder - in real use, load actual vector store
    print("\nNote: This is a skeleton test.")
    print("Run examples/retrieval_usage.py for complete examples with real data.")
