"""
Cross-Encoder Reranking
========================

Reranks initial retrieval results using a cross-encoder model for high accuracy.

Why Cross-Encoder?
------------------
Bi-encoder (used in Stage 2):
  - Encodes query and document separately
  - Fast: O(1) comparison via dot product
  - Good for initial retrieval (thousands of candidates)
  - Accuracy: ~75-85%

Cross-encoder (used in Stage 3):
  - Encodes query + document together
  - Slow: O(n) full model passes
  - Only for final reranking (10-15 candidates)
  - Accuracy: ~90-95%

Model Choice:
  ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking
  - 22M parameters, ~90MB
  - Fast enough for 10-15 candidates
  - Good balance: speed vs accuracy

Score Interpretation:
  - Output: Raw logits (can be negative)
  - Convert to 0-1 via sigmoid
  - Higher score = more relevant
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Install with:\n"
        "pip install sentence-transformers"
    )

logger = logging.getLogger(__name__)


@dataclass
class RerankingConfig:
    """Configuration for cross-encoder reranking."""
    
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    batch_size: int = 16
    show_progress: bool = False
    device: str = 'cpu'
    
    # Score normalization
    use_sigmoid: bool = True  # Convert logits to 0-1 probabilities


class CrossEncoderReranker:
    """
    Cross-encoder reranker for accurate relevance scoring.
    
    Usage:
        >>> reranker = CrossEncoderReranker()
        >>> 
        >>> query = "What is an arithmetic progression?"
        >>> docs = [
        ...     "Definition: An arithmetic progression is a sequence...",
        ...     "Example: For the AP 2, 7, 12, find the common difference...",
        ...     "Theorem: The sum of first n terms is..."
        ... ]
        >>> 
        >>> pairs = [(query, doc) for doc in docs]
        >>> scores = reranker.score(pairs)
        >>> 
        >>> # Rerank by score
        >>> ranked = sorted(zip(scores, docs), reverse=True)
        >>> for score, doc in ranked:
        ...     print(f"{score:.3f}: {doc[:50]}...")
    """
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            config: Reranking configuration
        """
        self.config = config or RerankingConfig()
        
        logger.info(f"Loading cross-encoder: {self.config.model_name}")
        
        try:
            self.model = CrossEncoder(
                self.config.model_name,
                device=self.config.device,
                default_activation_function=None  # We'll apply sigmoid manually
            )
            
            logger.info(f"Cross-encoder loaded on device: {self.config.device}")
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise
    
    def score(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Score query-document pairs.
        
        Args:
            query_doc_pairs: List of (query, document) tuples
            normalize: Apply sigmoid to normalize scores to 0-1 (default: from config)
        
        Returns:
            numpy array of scores, shape (num_pairs,)
        """
        if not query_doc_pairs:
            return np.array([])
        
        normalize = normalize if normalize is not None else self.config.use_sigmoid
        
        try:
            # Get raw scores (logits)
            scores = self.model.predict(
                query_doc_pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True
            )
            
            # Normalize with sigmoid if requested
            if normalize:
                scores = self._sigmoid(scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of document texts
            top_k: Return only top-k results (default: all)
        
        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        
        # Score
        scores = self.score(pairs)
        
        # Create (index, score) tuples
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        ranked = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k:
            ranked = ranked[:top_k]
        
        return ranked
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation: 1 / (1 + exp(-x))"""
        return 1.0 / (1.0 + np.exp(-x))
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.config.model_name,
            'device': self.config.device,
            'batch_size': self.config.batch_size,
            'use_sigmoid': self.config.use_sigmoid
        }


def compare_bi_encoder_vs_cross_encoder(
    query: str,
    documents: List[str],
    bi_encoder_scores: np.ndarray,
    cross_encoder: CrossEncoderReranker
) -> dict:
    """
    Compare bi-encoder (similarity) vs cross-encoder (rerank) scores.
    
    Useful for understanding how reranking changes result order.
    
    Args:
        query: Query text
        documents: List of documents
        bi_encoder_scores: Cosine similarities from bi-encoder
        cross_encoder: CrossEncoderReranker instance
    
    Returns:
        dict with comparison statistics
    """
    # Get cross-encoder scores
    ce_scores = cross_encoder.rerank(query, documents)
    ce_scores_array = np.array([score for _, score in ce_scores])
    
    # Compute rank correlation (Spearman's rho)
    from scipy.stats import spearmanr
    
    correlation, p_value = spearmanr(bi_encoder_scores, ce_scores_array)
    
    # Find rank changes
    bi_ranks = np.argsort(-bi_encoder_scores)  # Descending
    ce_ranks = np.argsort(-ce_scores_array)    # Descending
    
    rank_changes = np.abs(bi_ranks - ce_ranks)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'avg_rank_change': float(np.mean(rank_changes)),
        'max_rank_change': int(np.max(rank_changes)),
        'bi_encoder_order': bi_ranks.tolist(),
        'cross_encoder_order': ce_ranks.tolist()
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing CrossEncoderReranker...")
    print("=" * 70)
    
    reranker = CrossEncoderReranker()
    
    query = "What is an arithmetic progression?"
    
    documents = [
        "Definition: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number.",
        "Example: For the sequence 2, 5, 8, 11, the common difference is 3.",
        "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        "Theorem: The nth term of an AP is given by an = a + (n-1)d.",
        "Photosynthesis is the process by which plants convert sunlight into energy."
    ]
    
    print(f"\nQuery: {query}")
    print("\nDocuments:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:80]}...")
    
    # Score
    pairs = [(query, doc) for doc in documents]
    scores = reranker.score(pairs)
    
    print("\nCross-Encoder Scores:")
    for i, (doc, score) in enumerate(zip(documents, scores), 1):
        print(f"{i}. Score: {score:.4f} | {doc[:60]}...")
    
    # Rerank
    ranked = reranker.rerank(query, documents, top_k=3)
    
    print("\nTop 3 After Reranking:")
    for i, (orig_idx, score) in enumerate(ranked, 1):
        print(f"{i}. Score: {score:.4f} | {documents[orig_idx][:60]}...")
