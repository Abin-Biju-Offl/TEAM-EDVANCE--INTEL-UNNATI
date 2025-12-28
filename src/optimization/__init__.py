"""
Hardware Optimization Module
=============================

Intel CPU-optimized implementations for RAG system.

Key Features:
- FAISS CPU optimizations with Intel MKL
- Batch embedding processing
- Query caching layer
- INT8 quantization support
"""

from src.optimization.cpu_optimizations import (
    OptimizedFAISSIndex,
    BatchEmbedder,
    CPUConfig,
    get_optimal_cpu_config
)

from src.optimization.query_cache import (
    QueryCache,
    CacheConfig,
    CacheStats
)

from src.optimization.quantization import (
    INT8Quantizer,
    QuantizedEmbeddings,
    quantize_embeddings,
    dequantize_embeddings
)

__all__ = [
    # CPU optimizations
    'OptimizedFAISSIndex',
    'BatchEmbedder',
    'CPUConfig',
    'get_optimal_cpu_config',
    
    # Query caching
    'QueryCache',
    'CacheConfig',
    'CacheStats',
    
    # Quantization
    'INT8Quantizer',
    'QuantizedEmbeddings',
    'quantize_embeddings',
    'dequantize_embeddings',
]
