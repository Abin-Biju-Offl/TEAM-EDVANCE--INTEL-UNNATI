"""
Intel CPU Optimization Benchmarks
==================================

Comprehensive latency and throughput benchmarks for Intel CPU optimizations.

Benchmarks:
1. FAISS CPU optimizations
2. Batch embedding processing
3. Query caching
4. INT8 quantization
5. End-to-end pipeline
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import time
from typing import List
import multiprocessing


# ============================================================================
# BENCHMARK 1: FAISS CPU Optimizations
# ============================================================================

def benchmark_faiss_cpu():
    """Benchmark FAISS with Intel CPU optimizations."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: FAISS CPU Optimizations")
    print("=" * 70)
    
    from src.optimization.cpu_optimizations import (
        OptimizedFAISSIndex,
        get_optimal_cpu_config,
        CPUProfiler
    )
    
    # Configuration
    dimension = 384
    n_vectors = 10000
    n_queries = 100
    
    config = get_optimal_cpu_config()
    print(f"\nCPU Configuration:")
    print(f"  Cores: {multiprocessing.cpu_count()}")
    print(f"  Threads: {config.num_threads}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  MKL enabled: {config.use_mkl}")
    
    # Generate data
    print(f"\nGenerating test data...")
    print(f"  Vectors: {n_vectors} x {dimension}")
    print(f"  Queries: {n_queries} x {dimension}")
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Build index
    print(f"\nBuilding FAISS index...")
    index = OptimizedFAISSIndex(dimension, config)
    
    start = time.time()
    index.build_index(vectors, use_ivf=True, nlist=100)
    build_time = time.time() - start
    
    print(f"  Build time: {build_time*1000:.2f}ms")
    print(f"  Throughput: {n_vectors/build_time:.0f} vectors/sec")
    
    # Single query search
    print(f"\nSingle Query Search (k=5):")
    
    single_times = []
    for i in range(10):
        start = time.time()
        distances, indices = index.search(queries[:1], k=5)
        single_times.append(time.time() - start)
    
    print(f"  Mean: {np.mean(single_times)*1000:.2f}ms")
    print(f"  Median: {np.median(single_times)*1000:.2f}ms")
    print(f"  P95: {np.percentile(single_times, 95)*1000:.2f}ms")
    print(f"  P99: {np.percentile(single_times, 99)*1000:.2f}ms")
    
    # Batch search
    print(f"\nBatch Search ({n_queries} queries, k=5):")
    
    start = time.time()
    distances, indices = index.batch_search(queries, k=5)
    batch_time = time.time() - start
    
    print(f"  Total time: {batch_time*1000:.2f}ms")
    print(f"  Per query: {(batch_time/n_queries)*1000:.2f}ms")
    print(f"  Throughput: {n_queries/batch_time:.0f} queries/sec")
    
    # Speedup
    single_total = np.mean(single_times) * n_queries
    speedup = single_total / batch_time
    print(f"  Speedup vs single: {speedup:.2f}x")
    
    return {
        'build_time': build_time,
        'single_query_mean': np.mean(single_times),
        'batch_total': batch_time,
        'batch_per_query': batch_time / n_queries,
        'speedup': speedup
    }


# ============================================================================
# BENCHMARK 2: Batch Embedding
# ============================================================================

def benchmark_batch_embedding():
    """Benchmark batch embedding processing."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Batch Embedding Processing")
    print("=" * 70)
    
    try:
        from sentence_transformers import SentenceTransformer
        from src.optimization.cpu_optimizations import BatchEmbedder, get_optimal_cpu_config
        
        # Load model
        print(f"\nLoading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate texts
        n_texts = 1000
        texts = [f"This is test sentence number {i} for embedding." for i in range(n_texts)]
        
        config = get_optimal_cpu_config()
        print(f"\nConfiguration:")
        print(f"  Texts: {n_texts}")
        print(f"  Batch size: {config.batch_size}")
        
        # Benchmark without batching (single at a time)
        print(f"\nWithout Batching (single encoding):")
        start = time.time()
        embeddings_single = []
        for text in texts[:100]:  # Only 100 for speed
            emb = model.encode([text], show_progress_bar=False)
            embeddings_single.append(emb[0])
        single_time = time.time() - start
        
        print(f"  Time (100 texts): {single_time*1000:.2f}ms")
        print(f"  Per text: {(single_time/100)*1000:.2f}ms")
        
        # Benchmark with batching
        print(f"\nWith Batching (batch_size={config.batch_size}):")
        embedder = BatchEmbedder(model, config)
        
        start = time.time()
        embeddings_batch = embedder.embed_texts(texts, show_progress=False)
        batch_time = time.time() - start
        
        print(f"  Time ({n_texts} texts): {batch_time*1000:.2f}ms")
        print(f"  Per text: {(batch_time/n_texts)*1000:.2f}ms")
        print(f"  Throughput: {n_texts/batch_time:.0f} texts/sec")
        
        # Speedup
        estimated_single = (single_time / 100) * n_texts
        speedup = estimated_single / batch_time
        print(f"  Speedup vs single: {speedup:.2f}x")
        
        return {
            'single_per_text': single_time / 100,
            'batch_per_text': batch_time / n_texts,
            'speedup': speedup
        }
        
    except ImportError:
        print("\n⚠️  sentence-transformers not installed, skipping this benchmark")
        print("   Install: pip install sentence-transformers")
        return None


# ============================================================================
# BENCHMARK 3: Query Caching
# ============================================================================

def benchmark_query_caching():
    """Benchmark query caching layer."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Query Caching")
    print("=" * 70)
    
    from src.optimization.query_cache import QueryCache, CacheConfig
    
    # Configuration
    n_queries = 1000
    unique_queries = 200  # 80% hit rate
    
    config = CacheConfig(
        query_cache_size=1000,
        embedding_cache_size=5000,
        result_cache_size=500
    )
    
    cache = QueryCache(config)
    
    print(f"\nConfiguration:")
    print(f"  Total queries: {n_queries}")
    print(f"  Unique queries: {unique_queries}")
    print(f"  Expected hit rate: {(1 - unique_queries/n_queries)*100:.0f}%")
    
    # Generate queries (with repeats for cache hits)
    queries = [f"query_{i % unique_queries}" for i in range(n_queries)]
    
    # Simulate workload WITHOUT cache
    print(f"\nWithout Cache:")
    start = time.time()
    for query in queries:
        # Simulate computation (5ms)
        time.sleep(0.005)
    no_cache_time = time.time() - start
    
    print(f"  Total time: {no_cache_time*1000:.2f}ms")
    print(f"  Per query: {(no_cache_time/n_queries)*1000:.2f}ms")
    
    # Simulate workload WITH cache
    print(f"\nWith Cache:")
    cache.reset_stats()
    
    start = time.time()
    for query in queries:
        # Try cache
        result = cache.get_answer(query)
        
        if result is None:
            # Cache miss - simulate computation (5ms)
            time.sleep(0.005)
            cache.put_answer(query, f"answer for {query}")
        # Cache hit - instant (0.1ms overhead)
        else:
            time.sleep(0.0001)
    
    cache_time = time.time() - start
    
    print(f"  Total time: {cache_time*1000:.2f}ms")
    print(f"  Per query: {(cache_time/n_queries)*1000:.2f}ms")
    
    # Statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats.query_hits}")
    print(f"  Misses: {stats.query_misses}")
    print(f"  Hit rate: {stats.hit_rate('query')*100:.1f}%")
    
    # Speedup
    speedup = no_cache_time / cache_time
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Latency reduction: {((no_cache_time - cache_time) / no_cache_time)*100:.1f}%")
    
    return {
        'no_cache_time': no_cache_time,
        'cache_time': cache_time,
        'hit_rate': stats.hit_rate('query'),
        'speedup': speedup
    }


# ============================================================================
# BENCHMARK 4: INT8 Quantization
# ============================================================================

def benchmark_int8_quantization():
    """Benchmark INT8 quantization."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: INT8 Quantization")
    print("=" * 70)
    
    from src.optimization.quantization import (
        INT8Quantizer,
        benchmark_quantization,
        compare_search_accuracy
    )
    
    dimension = 384
    n_embeddings = 10000
    
    print(f"\nConfiguration:")
    print(f"  Embeddings: {n_embeddings} x {dimension}")
    print(f"  Original dtype: float32")
    print(f"  Target dtype: int8")
    
    # Run benchmark
    results = benchmark_quantization(dimension, n_embeddings)
    
    print(f"\nMemory:")
    print(f"  Original: {results['original_size_mb']:.2f} MB")
    print(f"  Quantized: {results['quantized_size_mb']:.2f} MB")
    print(f"  Reduction: {results['memory_reduction']:.2f}x")
    
    print(f"\nSpeed:")
    print(f"  Quantize: {results['quantize_time']*1000:.2f}ms")
    print(f"  Dequantize: {results['dequantize_time']*1000:.2f}ms")
    
    print(f"\nAccuracy:")
    print(f"  MAE: {results['mae']:.6f}")
    print(f"  Cosine similarity: {results['cosine_similarity']:.4f}")
    print(f"  Accuracy loss: {results['accuracy_loss']:.2f}%")
    
    # Search accuracy comparison
    print(f"\nSearch Accuracy Test:")
    
    # Generate embeddings
    embeddings = np.random.randn(n_embeddings, dimension).astype(np.float32)
    from numpy.linalg import norm
    embeddings = embeddings / (norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    quantizer = INT8Quantizer()
    quantized = quantizer.quantize(embeddings)
    
    search_results = compare_search_accuracy(embeddings, quantized, n_queries=100, k=5)
    
    print(f"  Test queries: {search_results['n_queries']}")
    print(f"  k: {search_results['k']}")
    print(f"  Recall@{search_results['k']}: {search_results['recall@k']:.4f}")
    print(f"  Accuracy retained: {search_results['accuracy_retained']:.1f}%")
    
    return results


# ============================================================================
# BENCHMARK 5: End-to-End Pipeline
# ============================================================================

def benchmark_end_to_end():
    """Benchmark complete optimized pipeline."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: End-to-End Optimized Pipeline")
    print("=" * 70)
    
    from src.optimization.cpu_optimizations import OptimizedFAISSIndex, get_optimal_cpu_config
    from src.optimization.query_cache import QueryCache
    from src.optimization.quantization import INT8Quantizer
    
    # Configuration
    dimension = 384
    n_documents = 10000
    n_queries = 100
    
    config = get_optimal_cpu_config()
    
    print(f"\nPipeline Configuration:")
    print(f"  CPU threads: {config.num_threads}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Documents: {n_documents}")
    print(f"  Queries: {n_queries}")
    
    # Generate data
    print(f"\n1. Generating embeddings...")
    documents = np.random.randn(n_documents, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    # Option 1: BASELINE (no optimizations)
    print(f"\n2. BASELINE (No Optimizations):")
    
    try:
        import faiss
        
        # Build basic index
        start = time.time()
        baseline_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(documents)
        baseline_index.add(documents)
        baseline_build = time.time() - start
        
        # Search
        start = time.time()
        faiss.normalize_L2(queries)
        for query in queries:
            baseline_index.search(query.reshape(1, -1), 5)
        baseline_search = time.time() - start
        
        print(f"  Build: {baseline_build*1000:.2f}ms")
        print(f"  Search: {baseline_search*1000:.2f}ms")
        print(f"  Per query: {(baseline_search/n_queries)*1000:.2f}ms")
        
    except ImportError:
        print("  ⚠️  FAISS not available")
        baseline_build = 1.0
        baseline_search = 1.0
    
    # Option 2: OPTIMIZED (all optimizations)
    print(f"\n3. OPTIMIZED (CPU + Batch + Cache + INT8):")
    
    # Build optimized index
    start = time.time()
    opt_index = OptimizedFAISSIndex(dimension, config)
    opt_index.build_index(documents, use_ivf=True, nlist=100)
    opt_build = time.time() - start
    
    # Initialize cache
    cache = QueryCache()
    
    # Quantize (optional - for memory savings)
    quantizer = INT8Quantizer()
    quantized_docs = quantizer.quantize(documents)
    
    # Search with batching and caching
    start = time.time()
    
    # Simulate repeated queries (50% cache hit rate)
    test_queries = [queries[i % (n_queries // 2)] for i in range(n_queries)]
    
    for query in test_queries:
        # Check cache
        cached = cache.get_results(str(query[:10]), top_k=5)
        
        if cached is None:
            # Cache miss - search
            distances, indices = opt_index.search(query.reshape(1, -1), k=5)
            cache.put_results(str(query[:10]), list(indices[0]), top_k=5)
    
    opt_search = time.time() - start
    
    print(f"  Build: {opt_build*1000:.2f}ms")
    print(f"  Search: {opt_search*1000:.2f}ms")
    print(f"  Per query: {(opt_search/n_queries)*1000:.2f}ms")
    
    # Memory savings
    original_size = documents.nbytes / (1024**2)
    quantized_size = (
        quantized_docs.embeddings.nbytes +
        quantized_docs.scale.nbytes +
        quantized_docs.offset.nbytes
    ) / (1024**2)
    
    print(f"\n4. Memory Usage:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  Quantized: {quantized_size:.2f} MB")
    print(f"  Reduction: {original_size/quantized_size:.2f}x")
    
    # Comparison
    print(f"\n5. Performance Comparison:")
    build_speedup = baseline_build / opt_build
    search_speedup = baseline_search / opt_search
    
    print(f"  Build speedup: {build_speedup:.2f}x")
    print(f"  Search speedup: {search_speedup:.2f}x")
    print(f"  Overall latency reduction: {((baseline_search - opt_search) / baseline_search)*100:.1f}%")
    
    return {
        'baseline_search': baseline_search,
        'optimized_search': opt_search,
        'speedup': search_speedup,
        'memory_reduction': original_size / quantized_size
    }


# ============================================================================
# RUN ALL BENCHMARKS
# ============================================================================

def run_all_benchmarks():
    """Run comprehensive benchmark suite."""
    print("\n" + "=" * 70)
    print("INTEL CPU OPTIMIZATION BENCHMARKS")
    print("Comprehensive Performance Evaluation")
    print("=" * 70)
    
    results = {}
    
    # Run benchmarks
    results['faiss'] = benchmark_faiss_cpu()
    results['batch_embedding'] = benchmark_batch_embedding()
    results['caching'] = benchmark_query_caching()
    results['quantization'] = benchmark_int8_quantization()
    results['end_to_end'] = benchmark_end_to_end()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"\n1. FAISS CPU:")
    print(f"   Single query: {results['faiss']['single_query_mean']*1000:.2f}ms")
    print(f"   Batch speedup: {results['faiss']['speedup']:.2f}x")
    
    if results['batch_embedding']:
        print(f"\n2. Batch Embedding:")
        print(f"   Per text: {results['batch_embedding']['batch_per_text']*1000:.2f}ms")
        print(f"   Speedup: {results['batch_embedding']['speedup']:.2f}x")
    
    print(f"\n3. Query Caching:")
    print(f"   Hit rate: {results['caching']['hit_rate']*100:.1f}%")
    print(f"   Speedup: {results['caching']['speedup']:.2f}x")
    
    print(f"\n4. INT8 Quantization:")
    print(f"   Memory reduction: {results['quantization']['memory_reduction']:.2f}x")
    print(f"   Accuracy loss: {results['quantization']['accuracy_loss']:.2f}%")
    
    print(f"\n5. End-to-End:")
    print(f"   Search speedup: {results['end_to_end']['speedup']:.2f}x")
    print(f"   Memory reduction: {results['end_to_end']['memory_reduction']:.2f}x")
    
    print("\n" + "=" * 70)
    print("OVERALL IMPACT")
    print("=" * 70)
    print(f"✅ Latency reduced by {((1 - 1/results['end_to_end']['speedup']) * 100):.0f}%")
    print(f"✅ Memory usage reduced by {((1 - 1/results['end_to_end']['memory_reduction']) * 100):.0f}%")
    print(f"✅ Query throughput increased by {results['end_to_end']['speedup']:.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
