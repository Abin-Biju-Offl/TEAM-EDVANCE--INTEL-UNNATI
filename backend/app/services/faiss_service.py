"""
FAISS Service - Phase 5

Vector store using FAISS with Intel CPU optimizations.
Integrates OptimizedFAISSIndex from Phase 7.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger
import os

from app.core.config import settings


class FAISSService:
    """Service for FAISS vector store with Intel CPU optimizations"""
    
    def __init__(self):
        """Initialize FAISS service"""
        self.dimension = settings.embedding_dimension
        self.index_type = settings.faiss_index_type
        self.nlist = settings.faiss_nlist
        self.nprobe = settings.faiss_nprobe
        
        self.index = None
        self.chunks = []  # Store chunk metadata
        
        # Setup Intel MKL for better CPU performance
        self._setup_mkl()
    
    def _setup_mkl(self):
        """Configure Intel MKL for optimal CPU performance"""
        try:
            # Get number of physical cores
            import multiprocessing
            num_cores = multiprocessing.cpu_count() // 2  # Physical cores only
            
            # Set MKL threads
            os.environ['MKL_NUM_THREADS'] = str(num_cores)
            os.environ['OMP_NUM_THREADS'] = str(num_cores)
            os.environ['MKL_DYNAMIC'] = 'FALSE'
            
            logger.info(f"Intel MKL configured: {num_cores} threads")
        except Exception as e:
            logger.warning(f"Could not configure MKL: {e}")
    
    def create_index(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: Numpy array of embeddings (N x D)
            chunks: List of chunk dictionaries
        """
        logger.info(f"Creating FAISS index: {self.index_type}")
        
        n_vectors = embeddings.shape[0]
        
        # Check if embeddings array is empty
        if n_vectors == 0:
            logger.error("Cannot create FAISS index with 0 vectors")
            raise ValueError("No embeddings provided. Cannot create FAISS index.")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "IVF" and n_vectors > self.nlist:
            # IVF index (better for CPU)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                min(self.nlist, n_vectors // 10)
            )
            
            # Train index
            logger.info(f"Training IVF index with {n_vectors} vectors")
            self.index.train(embeddings.astype('float32'))
            
            # Set nprobe for search
            self.index.nprobe = min(self.nprobe, self.index.nlist)
            
        else:
            # Flat index (exact search)
            logger.info("Using Flat index (exact search)")
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks
        self.chunks = chunks
        
        logger.success(
            f"FAISS index created: {self.index.ntotal} vectors, "
            f"type={self.index_type}"
        )
    
    def search(self, query_embedding: np.ndarray, k: int = 20) -> List[Dict]:
        """
        Search FAISS index
        
        Args:
            query_embedding: Query vector (D,)
            k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(1 / (1 + distance))  # Convert distance to similarity
                chunk['distance'] = float(distance)
                results.append(chunk)
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 20) -> List[List[Dict]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Query vectors (N x D)
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        if self.index is None:
            raise ValueError("FAISS index not initialized")
        
        # Normalize queries
        queries = query_embeddings.astype('float32')
        faiss.normalize_L2(queries)
        
        # Batch search
        distances, indices = self.index.search(queries, k)
        
        # Convert to results
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for idx, distance in zip(query_indices, query_distances):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx].copy()
                    chunk['score'] = float(1 / (1 + distance))
                    chunk['distance'] = float(distance)
                    results.append(chunk)
            all_results.append(results)
        
        return all_results
    
    def save_index(self, index_path: str = None):
        """
        Save FAISS index and chunks to disk
        
        Args:
            index_path: Path to save index (optional)
        """
        if index_path is None:
            index_path = Path(settings.vector_store_dir) / "faiss_index"
        else:
            index_path = Path(index_path)
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path) + ".index")
        
        # Save chunks metadata
        with open(str(index_path) + ".chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.success(f"FAISS index saved to {index_path}")
    
    def load_index(self, index_path: str = None):
        """
        Load FAISS index and chunks from disk
        
        Args:
            index_path: Path to load index from (optional)
        """
        if index_path is None:
            index_path = Path(settings.vector_store_dir) / "faiss_index"
        else:
            index_path = Path(index_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path) + ".index")
        
        # Load chunks metadata
        with open(str(index_path) + ".chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Set nprobe if IVF index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = min(self.nprobe, self.index.nlist)
        
        logger.success(
            f"FAISS index loaded: {self.index.ntotal} vectors from {index_path}"
        )
    
    def index_exists(self, index_path: str = None) -> bool:
        """Check if index file exists"""
        if index_path is None:
            index_path = Path(settings.vector_store_dir) / "faiss_index"
        else:
            index_path = Path(index_path)
        
        return (Path(str(index_path) + ".index").exists() and
                Path(str(index_path) + ".chunks.pkl").exists())


# Global FAISS service instance
faiss_service = FAISSService()
