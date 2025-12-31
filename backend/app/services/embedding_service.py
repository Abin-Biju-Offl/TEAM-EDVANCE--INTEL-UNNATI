"""
Embedding Service - Phase 4

Generate embeddings using SentenceTransformers with Intel CPU optimizations.
Integrates batch processing and CPU optimizations from Phase 7.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from loguru import logger
import torch
from tqdm import tqdm

from app.core.config import settings


class EmbeddingService:
    """Service for generating embeddings with CPU optimizations"""
    
    def __init__(self):
        """Initialize embedding model"""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        
        # Load SentenceTransformer model with local_files_only to use cached model
        try:
            self.model = SentenceTransformer(settings.embedding_model, local_files_only=True)
            logger.info("Using cached model (offline mode)")
        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            logger.info("Downloading model from HuggingFace...")
            self.model = SentenceTransformer(settings.embedding_model)
        
        # Force CPU usage (Intel optimizations)
        self.model = self.model.to('cpu')
        
        # Set to evaluation mode
        self.model.eval()
        
        self.embedding_dim = settings.embedding_dimension
        self.batch_size = settings.batch_size
        
        logger.success(f"Embedding model loaded: {self.embedding_dim}D, batch_size={self.batch_size}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode single text
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        with torch.no_grad():
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        return embedding
    
    def encode_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode batch of texts with CPU optimizations
        
        Args:
            texts: List of texts to encode
            show_progress: Show progress bar
            
        Returns:
            Array of embedding vectors
        """
        logger.info(f"Encoding {len(texts)} texts in batches of {self.batch_size}")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
        
        logger.success(f"Encoded {len(texts)} texts -> {embeddings.shape}")
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Add embeddings to chunks
        
        Args:
            chunks: List of chunks with 'text' field
            
        Returns:
            List of chunks with added 'embedding' field
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.encode_batch(texts, show_progress=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
            chunk['embedding_dim'] = len(embedding)
        
        logger.success(f"Added embeddings to {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[List[Dict]]) -> List[Dict]:
        """
        Process multiple documents (each document is a list of chunks)
        
        Args:
            documents: List of document chunks
            
        Returns:
            Flattened list of all chunks with embeddings
        """
        # Flatten all chunks
        all_chunks = []
        for doc_chunks in documents:
            all_chunks.extend(doc_chunks)
        
        logger.info(f"Processing {len(all_chunks)} total chunks from {len(documents)} documents")
        
        # Add embeddings
        embedded_chunks = self.embed_chunks(all_chunks)
        
        return embedded_chunks


# Global embedding service instance
embedding_service = EmbeddingService()
