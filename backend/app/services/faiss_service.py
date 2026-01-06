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
        
        # Support for multiple language indexes
        self.indexes = {}  # {'en': index, 'hi': index}
        self.chunks_by_lang = {}  # {'en': chunks, 'hi': chunks}
        
        # Legacy support (defaults to English)
        self.index = None
        self.chunks = []
        
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
    
    def search(self, query_embedding: np.ndarray, k: int = 20, language: str = 'en') -> List[Dict]:
        """
        Search FAISS index
        
        Args:
            query_embedding: Query vector (D,)
            k: Number of results to return
            language: Language of the index to search ('en' or 'hi')
            
        Returns:
            List of retrieved chunks with scores
        """
        # Try to use language-specific index first
        if language in self.indexes:
            index = self.indexes[language]
            chunks = self.chunks_by_lang[language]
        elif self.index is not None:
            # Fallback to legacy index
            index = self.index
            chunks = self.chunks
        else:
            raise ValueError("FAISS index not initialized")
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = index.search(query, k)
        
        # Convert to results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(chunks):
                chunk = chunks[idx].copy()
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
    
    def load_index_for_language(self, language: str) -> bool:
        """Load FAISS index for specific language"""
        if language == 'en':
            index_path = Path(settings.vector_store_dir) / "faiss_index"
        elif language == 'hi':
            index_path = Path(settings.vector_store_dir) / "hindi_faiss_index"
        else:
            logger.warning(f"Unsupported language: {language}")
            return False
        
        if not self.index_exists(str(index_path)):
            logger.warning(f"Index not found for language '{language}' at {index_path}")
            return False
        
        try:
            # Load index files
            index = faiss.read_index(str(index_path) + ".index")
            
            with open(str(index_path) + ".chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            
            # Set nprobe if IVF index
            if isinstance(index, faiss.IndexIVFFlat):
                index.nprobe = min(self.nprobe, getattr(index, 'nlist', self.nlist))
            
            self.indexes[language] = index
            self.chunks_by_lang[language] = chunks
            
            # Update legacy attributes for backward compatibility (use English by default)
            if language == 'en' or not self.index:
                self.index = index
                self.chunks = chunks
            
            logger.success(f"Loaded '{language}' index: {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load '{language}' index: {e}")
            return False
    
    def load_index_for_class(self, class_num: str, subject: str = None, language: str = 'en') -> bool:
        """
        Load FAISS index for specific class/subject/language combination
        
        Args:
            class_num: Class number (e.g., '9', '10')
            subject: Subject name (e.g., 'science', 'maths', None for all)
            language: Language code ('en' or 'hi')
        
        Returns:
            True if index loaded successfully
        """
        # Build index path
        base_path = Path(settings.vector_store_dir) / f"class-{class_num}"
        
        if subject:
            # Check if subject already includes language suffix (e.g., "all-subjects-hindi")
            if subject.endswith('-english') or subject.endswith('-hindi'):
                # Subject already has language, use as-is
                index_path = base_path / subject / "faiss_index"
            else:
                # Append language code
                index_path = base_path / f"{subject}-{language}" / "faiss_index"
        else:
            # Try to find any subject for this class
            if not base_path.exists():
                logger.warning(f"No indices found for class {class_num}")
                return False
            
            # Get first available subject
            subject_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith(f"-{language}")]
            if not subject_dirs:
                logger.warning(f"No {language} indices found for class {class_num}")
                return False
            
            index_path = subject_dirs[0] / "faiss_index"
        
        if not self.index_exists(str(index_path)):
            logger.warning(f"Index not found at {index_path}")
            return False
        
        try:
            # Load index files
            index = faiss.read_index(str(index_path) + ".index")
            
            with open(str(index_path) + ".chunks.pkl", 'rb') as f:
                chunks = pickle.load(f)
            
            # Set nprobe if IVF index
            if isinstance(index, faiss.IndexIVFFlat):
                index.nprobe = min(self.nprobe, getattr(index, 'nlist', self.nlist))
            
            # Store with class-specific key
            key = f"class-{class_num}-{subject or 'all'}-{language}"
            self.indexes[key] = index
            self.chunks_by_lang[key] = chunks
            
            # Update legacy attributes
            self.index = index
            self.chunks = chunks
            
            logger.success(f"Loaded class {class_num} index: {len(chunks)} chunks from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load class {class_num} index: {e}")
            return False


# Global FAISS service instance
faiss_service = FAISSService()
