"""
Intel CPU Optimizations for FAISS and Embeddings
=================================================

Hardware-aware optimizations for Intel CPUs:
- FAISS with Intel MKL (Math Kernel Library)
- Batch embedding processing
- Thread pool optimization
- Cache-friendly data structures
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CPUConfig:
    """CPU optimization configuration."""
    # Threading
    num_threads: int = -1  # -1 = auto-detect
    use_mkl: bool = True   # Intel MKL for FAISS
    
    # Batch processing
    batch_size: int = 32
    max_batch_size: int = 128
    
    # FAISS index
    index_type: str = "IVF"  # IVF, Flat, HNSW
    nprobe: int = 16         # IVF probes
    efSearch: int = 64       # HNSW search
    
    # Memory
    use_mmap: bool = False   # Memory-map large indices
    preload_index: bool = True


def get_optimal_cpu_config() -> CPUConfig:
    """
    Detect optimal CPU configuration for Intel hardware.
    
    Returns:
        CPUConfig optimized for current CPU
    """
    import multiprocessing
    
    num_cores = multiprocessing.cpu_count()
    
    # Intel-specific optimizations
    config = CPUConfig()
    
    # Threading: Use physical cores (not hyperthreads)
    config.num_threads = max(1, num_cores // 2)
    
    # Batch size based on core count
    if num_cores >= 16:
        config.batch_size = 64
        config.max_batch_size = 256
    elif num_cores >= 8:
        config.batch_size = 32
        config.max_batch_size = 128
    else:
        config.batch_size = 16
        config.max_batch_size = 64
    
    # FAISS index type
    config.index_type = "IVF"  # Good CPU performance
    config.nprobe = min(32, num_cores)
    
    logger.info(f"Detected {num_cores} CPU cores")
    logger.info(f"Optimal config: threads={config.num_threads}, batch_size={config.batch_size}")
    
    return config


class OptimizedFAISSIndex:
    """
    FAISS index optimized for Intel CPUs.
    
    Features:
    - Intel MKL acceleration
    - Optimized threading
    - Batch search support
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        dimension: int,
        config: Optional[CPUConfig] = None
    ):
        """
        Initialize optimized FAISS index.
        
        Args:
            dimension: Embedding dimension
            config: CPU configuration (auto-detect if None)
        """
        self.dimension = dimension
        self.config = config or get_optimal_cpu_config()
        self.index = None
        self._setup_mkl()
        
    def _setup_mkl(self):
        """Configure Intel MKL for FAISS."""
        if not self.config.use_mkl:
            return
        
        try:
            import faiss
            
            # Set number of threads
            if self.config.num_threads > 0:
                faiss.omp_set_num_threads(self.config.num_threads)
                logger.info(f"FAISS using {self.config.num_threads} threads")
            
            # Enable Intel MKL optimizations
            os.environ['MKL_NUM_THREADS'] = str(self.config.num_threads)
            os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
            os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic thread adjustment
            
            logger.info("Intel MKL enabled for FAISS")
            
        except ImportError:
            logger.warning("FAISS not available - skipping MKL setup")
    
    def build_index(
        self,
        embeddings: np.ndarray,
        use_ivf: bool = True,
        nlist: int = 100
    ):
        """
        Build optimized FAISS index.
        
        Args:
            embeddings: Embedding vectors (N x D)
            use_ivf: Use IVF index for speed
            nlist: Number of IVF clusters
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS required: pip install faiss-cpu")
        
        n_embeddings = len(embeddings)
        
        # Ensure float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Choose index type based on size
        if use_ivf and n_embeddings > 1000:
            # IVF index for large datasets
            nlist = min(nlist, n_embeddings // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
            # Train index
            logger.info(f"Training IVF index with {nlist} clusters...")
            self.index.train(embeddings)
            self.index.nprobe = self.config.nprobe
            
        else:
            # Flat index for small datasets
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors
        logger.info(f"Adding {n_embeddings} vectors to index...")
        self.index.add(embeddings)
        
        logger.info(f"Index built: {self.index.ntotal} vectors")
    
    def search(
        self,
        queries: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index (optimized for CPU).
        
        Args:
            queries: Query vectors (N x D)
            k: Number of neighbors
            
        Returns:
            distances, indices (both N x k)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure float32
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        
        # Normalize
        import faiss
        faiss.normalize_L2(queries)
        
        # Search
        distances, indices = self.index.search(queries, k)
        
        return distances, indices
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 5,
        batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for efficiency.
        
        Args:
            queries: Query vectors (N x D)
            k: Number of neighbors
            batch_size: Batch size (uses config if None)
            
        Returns:
            distances, indices (both N x k)
        """
        batch_size = batch_size or self.config.batch_size
        n_queries = len(queries)
        
        all_distances = []
        all_indices = []
        
        for i in range(0, n_queries, batch_size):
            batch = queries[i:i+batch_size]
            distances, indices = self.search(batch, k)
            all_distances.append(distances)
            all_indices.append(indices)
        
        return np.vstack(all_distances), np.vstack(all_indices)
    
    def save(self, filepath: str):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        try:
            import faiss
            faiss.write_index(self.index, filepath)
            logger.info(f"Index saved to {filepath}")
        except ImportError:
            raise ImportError("FAISS required")
    
    def load(self, filepath: str):
        """Load index from disk."""
        try:
            import faiss
            self.index = faiss.read_index(filepath)
            logger.info(f"Index loaded from {filepath}")
        except ImportError:
            raise ImportError("FAISS required")


class BatchEmbedder:
    """
    Batch embedding processor for CPU efficiency.
    
    Features:
    - Automatic batching
    - Progress tracking
    - Memory-efficient processing
    """
    
    def __init__(
        self,
        embedding_model: Any,
        config: Optional[CPUConfig] = None
    ):
        """
        Initialize batch embedder.
        
        Args:
            embedding_model: Embedding model with encode() method
            config: CPU configuration
        """
        self.model = embedding_model
        self.config = config or get_optimal_cpu_config()
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed texts in batches.
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Embeddings array (N x D)
        """
        n_texts = len(texts)
        batch_size = self.config.batch_size
        
        embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(
                range(0, n_texts, batch_size),
                desc="Embedding batches",
                total=(n_texts + batch_size - 1) // batch_size
            )
        else:
            iterator = range(0, n_texts, batch_size)
        
        for i in iterator:
            batch = texts[i:i+batch_size]
            
            # Embed batch
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        logger.info(f"Embedded {n_texts} texts in {len(embeddings)} batches")
        
        return all_embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.model.encode([text], convert_to_numpy=True)[0]


class CPUProfiler:
    """
    Profile CPU performance for optimization.
    
    Tracks:
    - Latency per operation
    - Throughput (queries/sec)
    - CPU utilization
    """
    
    def __init__(self):
        self.timings = {}
    
    def profile(self, operation: str):
        """Context manager for profiling."""
        return _ProfileContext(self, operation)
    
    def record(self, operation: str, duration: float):
        """Record timing."""
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
    
    def get_stats(self, operation: str) -> dict:
        """Get statistics for operation."""
        if operation not in self.timings:
            return {}
        
        timings = self.timings[operation]
        return {
            'count': len(timings),
            'mean': np.mean(timings),
            'median': np.median(timings),
            'p95': np.percentile(timings, 95),
            'p99': np.percentile(timings, 99),
            'min': np.min(timings),
            'max': np.max(timings)
        }
    
    def print_report(self):
        """Print profiling report."""
        print("\n" + "=" * 70)
        print("CPU PROFILING REPORT")
        print("=" * 70)
        
        for operation, timings in self.timings.items():
            stats = self.get_stats(operation)
            
            print(f"\n{operation}:")
            print(f"  Count:   {stats['count']}")
            print(f"  Mean:    {stats['mean']*1000:.2f}ms")
            print(f"  Median:  {stats['median']*1000:.2f}ms")
            print(f"  P95:     {stats['p95']*1000:.2f}ms")
            print(f"  P99:     {stats['p99']*1000:.2f}ms")
            print(f"  Min:     {stats['min']*1000:.2f}ms")
            print(f"  Max:     {stats['max']*1000:.2f}ms")
            
            # Throughput
            if stats['mean'] > 0:
                throughput = 1.0 / stats['mean']
                print(f"  Throughput: {throughput:.1f} ops/sec")


class _ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: CPUProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        self.profiler.record(self.operation, duration)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def optimize_numpy_for_intel():
    """Configure NumPy for Intel CPUs."""
    # Set thread counts
    config = get_optimal_cpu_config()
    
    os.environ['OPENBLAS_NUM_THREADS'] = str(config.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(config.num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(config.num_threads)
    
    logger.info(f"NumPy configured for {config.num_threads} threads")


def benchmark_faiss_operations(
    dimension: int = 384,
    n_vectors: int = 10000,
    n_queries: int = 100
) -> dict:
    """
    Benchmark FAISS operations on CPU.
    
    Args:
        dimension: Embedding dimension
        n_vectors: Number of vectors in index
        n_queries: Number of queries
        
    Returns:
        Benchmark results dict
    """
    # Generate random data
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Build index
    index = OptimizedFAISSIndex(dimension)
    
    profiler = CPUProfiler()
    
    with profiler.profile("build_index"):
        index.build_index(vectors)
    
    # Search
    with profiler.profile("search_single"):
        for i in range(10):
            index.search(queries[:1], k=5)
    
    with profiler.profile("search_batch"):
        index.batch_search(queries, k=5)
    
    # Get results
    results = {
        'dimension': dimension,
        'n_vectors': n_vectors,
        'n_queries': n_queries,
        'build_time': profiler.get_stats("build_index")['mean'],
        'search_single_mean': profiler.get_stats("search_single")['mean'],
        'search_batch_mean': profiler.get_stats("search_batch")['mean']
    }
    
    return results
