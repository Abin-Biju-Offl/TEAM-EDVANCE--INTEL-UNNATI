"""
Multi-Stage Retrieval Pipeline
===============================

Implements a production-grade retrieval system for NCERT educational content.

Key components:
- RetrievalPipeline: Multi-stage retrieval with filtering and ranking
- CrossEncoderReranker: Accuracy-focused reranking
- ConfidenceScorer: Confidence estimation and hallucination detection
"""

from .retrieval_pipeline import RetrievalPipeline, RetrievalConfig, RetrievalResult
from .reranker import CrossEncoderReranker, RerankingConfig
from .confidence_scorer import ConfidenceScorer, ConfidenceThresholds

__all__ = [
    'RetrievalPipeline',
    'RetrievalConfig',
    'RetrievalResult',
    'CrossEncoderReranker',
    'RerankingConfig',
    'ConfidenceScorer',
    'ConfidenceThresholds'
]
