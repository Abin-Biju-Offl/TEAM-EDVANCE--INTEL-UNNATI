"""
FAISS Vector Store for NCERT Embeddings
========================================

Manages FAISS index creation, storage, and loading.

Design Decisions:
1. FAISS Index Type:
   - IndexFlatIP: Exact search using inner product (cosine similarity with normalized vectors)
   - CPU-optimized, no GPU required
   - For <100K vectors: Exact search is fast enough
   - For >100K vectors: Can upgrade to IndexIVFFlat (approximate search)

2. Metadata Storage:
   - Stored separately from FAISS (FAISS only stores vectors)
   - JSON format for human readability and debugging
   - Maps vector_id → full chunk metadata
   - Enables pre-retrieval filtering

3. Index Persistence:
   - Save FAISS index to .index file
   - Save metadata to .json file
   - Save config to .json file
   - All files stored together for easy management

4. Memory Optimization:
   - float32 precision (standard for sentence embeddings)
   - Batch addition to FAISS for large datasets
   - Memory-mapped index loading for very large datasets (future)
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu not installed. Install with:\n"
        "pip install faiss-cpu"
    )

from ..chunking.semantic_chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for FAISS vector store."""
    
    embedding_dim: int
    index_type: str = 'IndexFlatIP'  # Inner product for cosine similarity
    metric: str = 'cosine'           # Similarity metric
    normalize_vectors: bool = True   # Must be True for IndexFlatIP
    
    # For approximate search (future optimization)
    use_ivf: bool = False            # Use IndexIVFFlat for large datasets
    nlist: int = 100                 # Number of clusters for IVF
    nprobe: int = 10                 # Number of clusters to search
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'VectorStoreConfig':
        return cls(**data)


class VectorStore:
    """
    FAISS-based vector store for NCERT chunk embeddings.
    
    Features:
    - Exact or approximate nearest neighbor search
    - CPU-optimized (no GPU required)
    - Separate metadata storage
    - Persistence to disk
    - Batch operations for efficiency
    
    Example:
        >>> store = VectorStore(embedding_dim=768)
        >>> store.add_chunks(chunks, embeddings)
        >>> store.save('output/vector_store')
        >>> 
        >>> # Later...
        >>> store = VectorStore.load('output/vector_store')
        >>> results = store.search(query_embedding, k=5)
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize vector store.
        
        Args:
            config: Vector store configuration
        """
        if config is None:
            raise ValueError("config required (must specify embedding_dim)")
        
        self.config = config
        self.index = None
        self.metadata_store = {}  # vector_id → metadata dict
        self.chunk_id_to_vector_id = {}  # chunk_id → vector_id
        self.vector_count = 0
        
        # Create FAISS index
        self._create_index()
        
        logger.info(f"Initialized VectorStore: {self.config.index_type}, dim={self.config.embedding_dim}")
    
    def _create_index(self):
        """Create FAISS index based on configuration."""
        
        if self.config.use_ivf:
            # Approximate search with IVF (for large datasets)
            quantizer = faiss.IndexFlatIP(self.config.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.config.embedding_dim,
                self.config.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            logger.info(f"Created IndexIVFFlat: nlist={self.config.nlist}, nprobe={self.config.nprobe}")
        else:
            # Exact search with flat index
            self.index = faiss.IndexFlatIP(self.config.embedding_dim)
            logger.info("Created IndexFlatIP (exact search)")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        chunk_ids: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add embeddings with metadata to store.
        
        Args:
            embeddings: numpy array of shape (num_vectors, embedding_dim)
            metadata_list: List of metadata dicts (one per embedding)
            chunk_ids: Optional list of chunk IDs for lookup
        
        Returns:
            List of assigned vector IDs
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError("Number of embeddings must match metadata list length")
        
        if embeddings.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != config dim {self.config.embedding_dim}"
            )
        
        # Normalize if required
        if self.config.normalize_vectors:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # For IVF index, train if not already trained
        if self.config.use_ivf and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("Training complete")
        
        # Add to FAISS index
        start_id = self.vector_count
        self.index.add(embeddings)
        self.vector_count += embeddings.shape[0]
        
        # Store metadata
        vector_ids = list(range(start_id, self.vector_count))
        for vector_id, metadata in zip(vector_ids, metadata_list):
            self.metadata_store[vector_id] = metadata
        
        # Store chunk_id mapping if provided
        if chunk_ids:
            for vector_id, chunk_id in zip(vector_ids, chunk_ids):
                self.chunk_id_to_vector_id[chunk_id] = vector_id
        
        logger.info(f"Added {len(vector_ids)} vectors (total: {self.vector_count})")
        
        return vector_ids
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> List[int]:
        """
        Add chunks with embeddings to store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (num_chunks, embedding_dim)
        
        Returns:
            List of assigned vector IDs
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Extract metadata and chunk IDs
        metadata_list = []
        chunk_ids = []
        
        for chunk in chunks:
            # Store full chunk data as metadata
            metadata = {
                'chunk_id': chunk.metadata.chunk_id,
                'class_number': chunk.metadata.class_number,
                'subject': chunk.metadata.subject,
                'chapter_number': chunk.metadata.chapter_number,
                'chapter_title': chunk.metadata.chapter_title,
                'section_number': chunk.metadata.section_number,
                'source_file': chunk.metadata.source_file,
                'page_numbers': chunk.metadata.page_numbers,
                'language': chunk.metadata.language,
                'chunk_type': chunk.metadata.chunk_type,
                'token_count': chunk.metadata.token_count,
                'has_equations': chunk.metadata.has_equations,
                'has_examples': chunk.metadata.has_examples,
                'has_exercises': chunk.metadata.has_exercises,
                'structure_confidence': chunk.metadata.structure_confidence,
                'completeness': chunk.metadata.completeness,
                'content': chunk.content  # Store content in metadata for retrieval
            }
            
            metadata_list.append(metadata)
            chunk_ids.append(chunk.metadata.chunk_id)
        
        return self.add_embeddings(embeddings, metadata_list, chunk_ids)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_metadata: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,) or (1, embedding_dim)
            k: Number of nearest neighbors to return
            return_metadata: Whether to return metadata for results
        
        Returns:
            Tuple of (distances, indices, metadata_list)
            - distances: Similarity scores (higher = more similar for cosine)
            - indices: Vector IDs of nearest neighbors
            - metadata_list: List of metadata dicts (if return_metadata=True)
        """
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if required
        if self.config.normalize_vectors:
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / (norm + 1e-8)
        
        # Ensure float32
        query_embedding = query_embedding.astype(np.float32)
        
        # Set nprobe for IVF index
        if self.config.use_ivf:
            self.index.nprobe = self.config.nprobe
        
        # Search
        k = min(k, self.vector_count)  # Can't return more than we have
        distances, indices = self.index.search(query_embedding, k)
        
        # Get metadata if requested
        metadata_list = None
        if return_metadata:
            metadata_list = []
            for idx in indices[0]:
                if idx == -1:  # FAISS returns -1 for missing results
                    metadata_list.append(None)
                else:
                    metadata_list.append(self.metadata_store.get(int(idx)))
        
        return distances[0], indices[0], metadata_list
    
    def get_by_chunk_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata by chunk ID."""
        vector_id = self.chunk_id_to_vector_id.get(chunk_id)
        if vector_id is None:
            return None
        return self.metadata_store.get(vector_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            'total_vectors': self.vector_count,
            'embedding_dim': self.config.embedding_dim,
            'index_type': self.config.index_type,
            'is_trained': self.index.is_trained if self.config.use_ivf else True,
            'metadata_entries': len(self.metadata_store),
            'chunk_id_mappings': len(self.chunk_id_to_vector_id)
        }
    
    def save(self, output_dir: str):
        """
        Save vector store to disk.
        
        Args:
            output_dir: Directory to save files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = output_path / 'faiss.index'
        faiss.write_index(self.index, str(index_file))
        logger.info(f"Saved FAISS index: {index_file}")
        
        # Save metadata
        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata_store': {str(k): v for k, v in self.metadata_store.items()},
                'chunk_id_to_vector_id': self.chunk_id_to_vector_id,
                'vector_count': self.vector_count
            }, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata: {metadata_file}")
        
        # Save config
        config_file = output_path / 'config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved config: {config_file}")
        
        logger.info(f"Vector store saved to: {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'VectorStore':
        """
        Load vector store from disk.
        
        Args:
            input_dir: Directory containing saved files
        
        Returns:
            Loaded VectorStore instance
        """
        input_path = Path(input_dir)
        
        # Load config
        config_file = input_path / 'config.json'
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config = VectorStoreConfig.from_dict(config_data)
        
        # Create instance
        store = cls(config)
        
        # Load FAISS index
        index_file = input_path / 'faiss.index'
        store.index = faiss.read_index(str(index_file))
        logger.info(f"Loaded FAISS index: {index_file}")
        
        # Load metadata
        metadata_file = input_path / 'metadata.json'
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            store.metadata_store = {int(k): v for k, v in data['metadata_store'].items()}
            store.chunk_id_to_vector_id = data['chunk_id_to_vector_id']
            store.vector_count = data['vector_count']
        logger.info(f"Loaded metadata: {metadata_file}")
        
        logger.info(f"Vector store loaded from: {input_dir}")
        logger.info(f"Total vectors: {store.vector_count}")
        
        return store
    
    def reset(self):
        """Clear all data and recreate index."""
        self._create_index()
        self.metadata_store.clear()
        self.chunk_id_to_vector_id.clear()
        self.vector_count = 0
        logger.info("Vector store reset")


def create_vector_store_from_chunks(
    chunks: List[Chunk],
    embeddings: np.ndarray,
    save_path: Optional[str] = None
) -> VectorStore:
    """
    Convenience function to create and populate vector store.
    
    Args:
        chunks: List of Chunk objects
        embeddings: numpy array of embeddings
        save_path: Optional path to save store
    
    Returns:
        Populated VectorStore instance
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Number of chunks must match number of embeddings")
    
    # Create config
    config = VectorStoreConfig(
        embedding_dim=embeddings.shape[1],
        normalize_vectors=True
    )
    
    # Create store
    store = VectorStore(config)
    
    # Add chunks
    store.add_chunks(chunks, embeddings)
    
    # Save if path provided
    if save_path:
        store.save(save_path)
    
    return store


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VectorStore...")
    print("=" * 70)
    
    # Create dummy data
    embedding_dim = 768
    num_vectors = 100
    
    embeddings = np.random.randn(num_vectors, embedding_dim).astype(np.float32)
    metadata_list = [{'id': i, 'text': f'Sample {i}'} for i in range(num_vectors)]
    
    # Create store
    config = VectorStoreConfig(embedding_dim=embedding_dim)
    store = VectorStore(config)
    
    # Add vectors
    vector_ids = store.add_embeddings(embeddings, metadata_list)
    print(f"Added {len(vector_ids)} vectors")
    
    # Search
    query = np.random.randn(embedding_dim).astype(np.float32)
    distances, indices, metadata = store.search(query, k=5)
    
    print(f"\nTop 5 results:")
    for i, (dist, idx, meta) in enumerate(zip(distances, indices, metadata), 1):
        print(f"{i}. Vector {idx}: similarity={dist:.4f}, metadata={meta}")
    
    # Test save/load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store.save(tmpdir)
        loaded_store = VectorStore.load(tmpdir)
        print(f"\nLoaded store: {loaded_store.get_statistics()}")
