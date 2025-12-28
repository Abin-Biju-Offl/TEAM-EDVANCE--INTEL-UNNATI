"""
Query Caching Layer
===================

Cache frequent queries to reduce latency.

Features:
- LRU cache for queries
- Embedding cache
- Result cache
- Cache statistics
"""

import hashlib
import time
from typing import Any, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""
    # Query cache
    query_cache_size: int = 1000
    query_ttl: int = 3600  # seconds
    
    # Embedding cache
    embedding_cache_size: int = 5000
    embedding_ttl: int = 7200
    
    # Result cache
    result_cache_size: int = 500
    result_ttl: int = 1800
    
    # Behavior
    enable_query_cache: bool = True
    enable_embedding_cache: bool = True
    enable_result_cache: bool = True


@dataclass
class CacheStats:
    """Cache statistics."""
    query_hits: int = 0
    query_misses: int = 0
    embedding_hits: int = 0
    embedding_misses: int = 0
    result_hits: int = 0
    result_misses: int = 0
    
    # Timing
    total_cache_time: float = 0.0
    total_compute_time: float = 0.0
    
    def hit_rate(self, cache_type: str) -> float:
        """Calculate hit rate for cache type."""
        if cache_type == 'query':
            total = self.query_hits + self.query_misses
            return self.query_hits / total if total > 0 else 0.0
        elif cache_type == 'embedding':
            total = self.embedding_hits + self.embedding_misses
            return self.embedding_hits / total if total > 0 else 0.0
        elif cache_type == 'result':
            total = self.result_hits + self.result_misses
            return self.result_hits / total if total > 0 else 0.0
        return 0.0
    
    def print_report(self):
        """Print cache statistics."""
        print("\n" + "=" * 70)
        print("CACHE STATISTICS")
        print("=" * 70)
        
        print(f"\nQuery Cache:")
        print(f"  Hits:     {self.query_hits}")
        print(f"  Misses:   {self.query_misses}")
        print(f"  Hit Rate: {self.hit_rate('query'):.1%}")
        
        print(f"\nEmbedding Cache:")
        print(f"  Hits:     {self.embedding_hits}")
        print(f"  Misses:   {self.embedding_misses}")
        print(f"  Hit Rate: {self.hit_rate('embedding'):.1%}")
        
        print(f"\nResult Cache:")
        print(f"  Hits:     {self.result_hits}")
        print(f"  Misses:   {self.result_misses}")
        print(f"  Hit Rate: {self.hit_rate('result'):.1%}")
        
        print(f"\nTiming:")
        print(f"  Cache Time:   {self.total_cache_time*1000:.2f}ms")
        print(f"  Compute Time: {self.total_compute_time*1000:.2f}ms")
        
        if self.total_cache_time + self.total_compute_time > 0:
            speedup = self.total_compute_time / (self.total_cache_time + self.total_compute_time)
            print(f"  Speedup:      {speedup:.2f}x")


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    key: str
    value: Any
    timestamp: float
    ttl: int
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > self.ttl


class LRUCache:
    """
    LRU (Least Recently Used) cache.
    
    Thread-safe cache with expiration.
    """
    
    def __init__(self, max_size: int, ttl: int):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        return entry.value
    
    def put(self, key: str, value: Any):
        """Put value in cache."""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Evict oldest if full
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first item)
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            ttl=self.ttl
        )
        self.cache[key] = entry
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)


class QueryCache:
    """
    Multi-level cache for RAG queries.
    
    Caches:
    1. Query → Embeddings
    2. Query → Search Results
    3. Query → Final Answer
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize query cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # Initialize caches
        self.query_cache = LRUCache(
            self.config.query_cache_size,
            self.config.query_ttl
        )
        
        self.embedding_cache = LRUCache(
            self.config.embedding_cache_size,
            self.config.embedding_ttl
        )
        
        self.result_cache = LRUCache(
            self.config.result_cache_size,
            self.config.result_ttl
        )
        
        logger.info(f"Query cache initialized: "
                   f"query_size={self.config.query_cache_size}, "
                   f"embedding_size={self.config.embedding_cache_size}, "
                   f"result_size={self.config.result_cache_size}")
    
    def _hash_query(self, query: str) -> str:
        """Hash query for cache key."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def _hash_params(self, **params) -> str:
        """Hash parameters for cache key."""
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for query.
        
        Args:
            query: Query text
            
        Returns:
            Cached embedding or None
        """
        if not self.config.enable_embedding_cache:
            return None
        
        start_time = time.time()
        
        key = self._hash_query(query)
        embedding = self.embedding_cache.get(key)
        
        self.stats.total_cache_time += time.time() - start_time
        
        if embedding is not None:
            self.stats.embedding_hits += 1
            logger.debug(f"Embedding cache hit: {query[:50]}")
            return embedding
        else:
            self.stats.embedding_misses += 1
            return None
    
    def put_embedding(self, query: str, embedding: np.ndarray):
        """Cache embedding for query."""
        if not self.config.enable_embedding_cache:
            return
        
        key = self._hash_query(query)
        self.embedding_cache.put(key, embedding)
        logger.debug(f"Cached embedding: {query[:50]}")
    
    def get_results(
        self,
        query: str,
        top_k: int = 5,
        **params
    ) -> Optional[List[Any]]:
        """
        Get cached search results.
        
        Args:
            query: Query text
            top_k: Number of results
            **params: Additional parameters
            
        Returns:
            Cached results or None
        """
        if not self.config.enable_result_cache:
            return None
        
        start_time = time.time()
        
        query_hash = self._hash_query(query)
        param_hash = self._hash_params(top_k=top_k, **params)
        key = f"{query_hash}_{param_hash}"
        
        results = self.result_cache.get(key)
        
        self.stats.total_cache_time += time.time() - start_time
        
        if results is not None:
            self.stats.result_hits += 1
            logger.debug(f"Result cache hit: {query[:50]}")
            return results
        else:
            self.stats.result_misses += 1
            return None
    
    def put_results(
        self,
        query: str,
        results: List[Any],
        top_k: int = 5,
        **params
    ):
        """Cache search results."""
        if not self.config.enable_result_cache:
            return
        
        query_hash = self._hash_query(query)
        param_hash = self._hash_params(top_k=top_k, **params)
        key = f"{query_hash}_{param_hash}"
        
        self.result_cache.put(key, results)
        logger.debug(f"Cached results: {query[:50]}")
    
    def get_answer(
        self,
        query: str,
        **params
    ) -> Optional[str]:
        """
        Get cached final answer.
        
        Args:
            query: Query text
            **params: Additional parameters
            
        Returns:
            Cached answer or None
        """
        if not self.config.enable_query_cache:
            return None
        
        start_time = time.time()
        
        query_hash = self._hash_query(query)
        param_hash = self._hash_params(**params)
        key = f"{query_hash}_{param_hash}_answer"
        
        answer = self.query_cache.get(key)
        
        self.stats.total_cache_time += time.time() - start_time
        
        if answer is not None:
            self.stats.query_hits += 1
            logger.debug(f"Answer cache hit: {query[:50]}")
            return answer
        else:
            self.stats.query_misses += 1
            return None
    
    def put_answer(
        self,
        query: str,
        answer: str,
        **params
    ):
        """Cache final answer."""
        if not self.config.enable_query_cache:
            return
        
        query_hash = self._hash_query(query)
        param_hash = self._hash_params(**params)
        key = f"{query_hash}_{param_hash}_answer"
        
        self.query_cache.put(key, answer)
        logger.debug(f"Cached answer: {query[:50]}")
    
    def clear_all(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.result_cache.clear()
        logger.info("All caches cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = CacheStats()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def benchmark_cache_performance(
    n_queries: int = 1000,
    cache_hit_rate: float = 0.5
) -> dict:
    """
    Benchmark cache performance.
    
    Args:
        n_queries: Number of queries to test
        cache_hit_rate: Expected cache hit rate
        
    Returns:
        Benchmark results
    """
    cache = QueryCache()
    
    # Generate queries
    queries = [f"query_{i % int(n_queries * cache_hit_rate)}" for i in range(n_queries)]
    
    # Simulate workload
    start_time = time.time()
    
    for query in queries:
        # Try cache
        result = cache.get_answer(query)
        
        if result is None:
            # Simulate computation (50ms)
            time.sleep(0.05)
            cache.put_answer(query, f"answer for {query}")
    
    total_time = time.time() - start_time
    
    stats = cache.get_stats()
    
    return {
        'n_queries': n_queries,
        'total_time': total_time,
        'avg_latency': total_time / n_queries,
        'query_hit_rate': stats.hit_rate('query'),
        'stats': stats
    }
