"""
Multilingual Embedding Generation
==================================

Generates dense vector embeddings for NCERT chunks using multilingual models.

Design Decisions:
1. Model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
   - Supports 50+ languages (including English, Hindi, Sanskrit)
   - 768-dimensional embeddings
   - Good balance: quality vs speed
   - 420M parameters, ~1.6GB RAM

2. No Translation Required:
   - Multilingual models map all languages to shared semantic space
   - English and Hindi texts with same meaning have similar embeddings
   - Preserves original text for accurate retrieval

3. Batch Processing:
   - Process multiple chunks at once for efficiency
   - CPU-optimized with configurable batch size
   - Progress tracking for large datasets

4. Normalization:
   - L2 normalize embeddings for cosine similarity
   - Compatible with FAISS IndexFlatIP (inner product = cosine sim)
"""

import logging
from typing import List, Union, Optional
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers not installed. Install with:\n"
        "pip install sentence-transformers"
    )

from ..chunking.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True
    device: str = 'cpu'  # Force CPU for compatibility
    
    # Alternative models (for reference):
    # - 'paraphrase-multilingual-MiniLM-L12-v2' (lighter, 384-dim)
    # - 'distiluse-base-multilingual-cased-v2' (512-dim)
    # - 'intfloat/multilingual-e5-large' (heavier, better quality, 1024-dim)


class EmbeddingGenerator:
    """
    Generates multilingual embeddings for NCERT chunks.
    
    Supports:
    - English, Hindi, Sanskrit (and 50+ other languages)
    - Batch processing for efficiency
    - Progress tracking
    - CPU-optimized execution
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = generator.embed_chunks(chunks)
        >>> print(embeddings.shape)  # (num_chunks, 768)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator.
        
        Args:
            config: Configuration for embedding generation
        """
        self.config = config or EmbeddingConfig()
        
        logger.info(f"Loading embedding model: {self.config.model_name}")
        logger.info(f"Device: {self.config.device}")
        
        try:
            self.model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed_text(
        self,
        texts: Union[str, List[str]],
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize (default: from config)
        
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        normalize = normalize if normalize is not None else self.config.normalize_embeddings
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress and len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_chunks(
        self,
        chunks: List[Chunk],
        use_metadata: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for chunk contents.
        
        Args:
            chunks: List of Chunk objects from semantic_chunker
            use_metadata: If True, prepend metadata to content for embedding
                         (useful for incorporating chapter/topic context)
        
        Returns:
            numpy array of shape (num_chunks, embedding_dim)
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            if use_metadata:
                # Include curriculum context in embedding
                # Format: "Class X, Subject, Chapter Y: <content>"
                metadata_prefix = (
                    f"Class {chunk.metadata.class_number}, "
                    f"{chunk.metadata.subject.replace('_', ' ').title()}, "
                    f"Chapter {chunk.metadata.chapter_number}: "
                )
                text = metadata_prefix + chunk.content
            else:
                # Embed content only (recommended)
                text = chunk.content
            
            texts.append(text)
        
        # Generate embeddings
        embeddings = self.embed_text(texts)
        
        logger.info(f"Generated embeddings: shape={embeddings.shape}")
        
        # Verify no NaN/Inf values
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            logger.error("Embeddings contain NaN or Inf values!")
            raise ValueError("Invalid embeddings generated")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.config.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.config.device,
            'max_seq_length': self.model.max_seq_length,
            'normalize': self.config.normalize_embeddings
        }


class EmbeddingCache:
    """
    Cache for embeddings to avoid recomputation.
    
    Useful when processing large corpora in batches.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize embedding cache.
        
        Args:
            cache_file: Path to save/load cache (optional)
        """
        self.cache = {}
        self.cache_file = cache_file
        
        if cache_file:
            self.load()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        return self.cache.get(text)
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        self.cache[text] = embedding
    
    def contains(self, text: str) -> bool:
        """Check if text is in cache."""
        return text in self.cache
    
    def save(self):
        """Save cache to disk."""
        if not self.cache_file:
            logger.warning("No cache file specified, cannot save")
            return
        
        try:
            import pickle
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved embedding cache: {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load(self):
        """Load cache from disk."""
        if not self.cache_file:
            return
        
        try:
            import pickle
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)
            logger.info(f"Loaded embedding cache: {len(self.cache)} entries")
        except FileNotFoundError:
            logger.info("No existing cache file found")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def clear(self):
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get number of cached embeddings."""
        return len(self.cache)


def benchmark_embedding_speed(
    generator: EmbeddingGenerator,
    num_samples: int = 100,
    text_length: int = 400
) -> dict:
    """
    Benchmark embedding generation speed.
    
    Args:
        generator: EmbeddingGenerator instance
        num_samples: Number of texts to embed
        text_length: Average character length per text
    
    Returns:
        dict with timing statistics
    """
    import time
    
    # Generate sample texts
    sample_text = "This is a sample text for benchmarking. " * (text_length // 40)
    texts = [sample_text] * num_samples
    
    # Time embedding generation
    start_time = time.time()
    embeddings = generator.embed_text(texts)
    elapsed = time.time() - start_time
    
    return {
        'num_samples': num_samples,
        'text_length': text_length,
        'total_time_seconds': elapsed,
        'time_per_sample_ms': (elapsed / num_samples) * 1000,
        'samples_per_second': num_samples / elapsed,
        'embedding_shape': embeddings.shape
    }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing EmbeddingGenerator...")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    # Test multilingual
    texts = [
        "Definition: An arithmetic progression is a sequence.",
        "परिभाषा: अंकगणितीय प्रगति एक अनुक्रम है।",
        "The formula for the nth term is an = a + (n-1)d"
    ]
    
    print("\nTest texts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    embeddings = generator.embed_text(texts)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 5 dims): {embeddings[0, :5]}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    print("\nCosine similarity matrix:")
    print(sim_matrix)
    print("\n(Note: English and Hindi definitions should have high similarity)")
    
    # Benchmark
    print("\n" + "=" * 70)
    print("Benchmarking...")
    stats = benchmark_embedding_speed(generator, num_samples=100)
    for key, val in stats.items():
        print(f"  {key}: {val}")
