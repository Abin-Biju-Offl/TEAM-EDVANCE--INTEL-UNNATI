"""
Multilingual Embedding and Vector Storage Module
=================================================

This module provides embedding generation and vector storage for NCERT chunks.

Key components:
- EmbeddingGenerator: Multilingual sentence embeddings
- VectorStore: FAISS index management
- MetadataFilter: Pre-retrieval filtering
"""

from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore, VectorStoreConfig
from .metadata_filter import MetadataFilter, FilterCondition

__all__ = [
    'EmbeddingGenerator',
    'VectorStore',
    'VectorStoreConfig',
    'MetadataFilter',
    'FilterCondition'
]
