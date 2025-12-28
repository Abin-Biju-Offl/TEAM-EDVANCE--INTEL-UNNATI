"""
Embedding and Vector Storage Usage Examples
============================================

Complete examples for:
1. Generating embeddings for chunks
2. Creating FAISS vector store
3. Filtering by metadata
4. Complete pipeline (chunks → embeddings → storage)
"""

import logging
from pathlib import Path
import json
import numpy as np

from src.embeddings import EmbeddingGenerator, VectorStore, VectorStoreConfig, MetadataFilter
from src.embeddings.metadata_filter import FilterBuilder, create_chapter_filter
from src.chunking import Chunk
from src.chunking.chunk_types import ChunkMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_1_basic_embedding():
    """Example 1: Generate embeddings for text."""
    print("=" * 70)
    print("Example 1: Basic Embedding Generation")
    print("=" * 70)
    
    generator = EmbeddingGenerator()
    
    # Multilingual texts
    texts = [
        "Definition: An arithmetic progression is a sequence of numbers.",
        "परिभाषा: अंकगणितीय प्रगति संख्याओं का एक अनुक्रम है।",
        "The common difference is constant between consecutive terms.",
        "Example: For the AP 2, 5, 8, 11, the common difference is 3."
    ]
    
    print("\nTexts to embed:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    # Generate embeddings
    embeddings = generator.embed_text(texts)
    
    print(f"\nEmbeddings generated:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    print(f"  Sample (first 5 dims): {embeddings[0, :5]}")
    
    # Check similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    # English and Hindi definitions should be similar
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\nSimilarity between English and Hindi definitions: {sim:.4f}")
    print("(Should be high ~0.7-0.9, showing multilingual understanding)")
    
    return generator, embeddings


def example_2_embed_chunks():
    """Example 2: Generate embeddings for chunks."""
    print("\n" + "=" * 70)
    print("Example 2: Embed NCERT Chunks")
    print("=" * 70)
    
    # Create sample chunks
    chunks = [
        Chunk(
            content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference.",
            metadata=ChunkMetadata(
                class_number=10,
                subject="mathematics",
                chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[95],
                language="eng",
                chunk_type="definition",
                chunk_id="10_mathematics_5_001",
                token_count=45,
                char_count=200,
                has_equations=False,
                has_examples=False,
                has_exercises=False,
                structure_confidence=1.0,
                completeness="complete"
            )
        ),
        Chunk(
            content="Example 5.3: For the AP: 3, 7, 11, 15, find the common difference.\n\nSolution: The common difference d = 7 - 3 = 4. We can verify: 11 - 7 = 4 and 15 - 11 = 4. Therefore, d = 4.",
            metadata=ChunkMetadata(
                class_number=10,
                subject="mathematics",
                chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[96],
                language="eng",
                chunk_type="example",
                chunk_id="10_mathematics_5_002",
                token_count=52,
                char_count=180,
                has_equations=True,
                has_examples=True,
                has_exercises=False,
                structure_confidence=0.95,
                completeness="complete"
            )
        ),
        Chunk(
            content="Theorem 5.1: The nth term of an AP with first term a and common difference d is given by the formula: an = a + (n-1)d",
            metadata=ChunkMetadata(
                class_number=10,
                subject="mathematics",
                chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[97],
                language="eng",
                chunk_type="theorem",
                chunk_id="10_mathematics_5_003",
                token_count=35,
                char_count=150,
                has_equations=True,
                has_examples=False,
                has_exercises=False,
                structure_confidence=1.0,
                completeness="complete"
            )
        )
    ]
    
    print(f"\nChunks to embed: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk.metadata.chunk_type.upper()}: {chunk.content[:60]}...")
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_chunks(chunks)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    
    return chunks, embeddings


def example_3_create_vector_store():
    """Example 3: Create and populate FAISS vector store."""
    print("\n" + "=" * 70)
    print("Example 3: Create FAISS Vector Store")
    print("=" * 70)
    
    # Get chunks and embeddings from previous example
    chunks, embeddings = example_2_embed_chunks()
    
    # Create vector store config
    config = VectorStoreConfig(
        embedding_dim=embeddings.shape[1],
        normalize_vectors=True
    )
    
    # Create store
    store = VectorStore(config)
    print(f"\nCreated vector store: {config.index_type}, dim={config.embedding_dim}")
    
    # Add chunks
    vector_ids = store.add_chunks(chunks, embeddings)
    print(f"Added {len(vector_ids)} vectors to store")
    
    # Get statistics
    stats = store.get_statistics()
    print("\nVector store statistics:")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    
    return store, chunks, embeddings


def example_4_metadata_filtering():
    """Example 4: Filter chunks by metadata before search."""
    print("\n" + "=" * 70)
    print("Example 4: Metadata Filtering")
    print("=" * 70)
    
    # Create store from previous example
    store, chunks, embeddings = example_3_create_vector_store()
    
    # Add more diverse chunks for filtering demo
    additional_chunks = [
        Chunk(
            content="Class 11 content: Trigonometric functions...",
            metadata=ChunkMetadata(
                class_number=11,  # Different class
                subject="mathematics",
                chapter_number=3,
                chapter_title="Trigonometric Functions",
                source_file="NCERT_Class11_Mathematics_English.pdf",
                page_numbers=[45],
                language="eng",
                chunk_type="definition",
                chunk_id="11_mathematics_3_001",
                token_count=40,
                char_count=100,
                has_equations=True,
                has_examples=False,
                has_exercises=False,
                structure_confidence=1.0,
                completeness="complete"
            )
        )
    ]
    
    # Embed and add
    generator = EmbeddingGenerator()
    additional_embeddings = generator.embed_chunks(additional_chunks)
    store.add_chunks(additional_chunks, additional_embeddings)
    
    print(f"Total vectors in store: {store.get_statistics()['total_vectors']}")
    
    # Test filters
    print("\n" + "-" * 70)
    print("Filter 1: Class 10 only")
    filter1 = MetadataFilter()
    filter1.add_condition("class_number", "==", 10)
    
    matching_ids = filter1.apply(store.metadata_store)
    print(f"Matched {len(matching_ids)} vectors: {matching_ids}")
    
    print("\n" + "-" * 70)
    print("Filter 2: Class 10, Math, Chapter 5")
    filter2 = create_chapter_filter(10, "mathematics", 5)
    matching_ids = filter2.apply(store.metadata_store)
    print(f"Matched {len(matching_ids)} vectors: {matching_ids}")
    
    print("\n" + "-" * 70)
    print("Filter 3: Definitions only (any class)")
    filter3 = MetadataFilter()
    filter3.add_condition("chunk_type", "==", "definition")
    matching_ids = filter3.apply(store.metadata_store)
    print(f"Matched {len(matching_ids)} vectors")
    for vid in matching_ids:
        meta = store.metadata_store[vid]
        print(f"  Vector {vid}: Class {meta['class_number']}, {meta['chunk_type']}")
    
    print("\n" + "-" * 70)
    print("Filter 4: FilterBuilder - Examples with equations")
    filter4 = (FilterBuilder()
        .for_class(10)
        .with_chunk_type("example")
        .with_equations(True)
        .build())
    
    matching_ids = filter4.apply(store.metadata_store)
    print(f"Filter: {filter4.get_filter_summary()}")
    print(f"Matched {len(matching_ids)} vectors: {matching_ids}")
    
    return store


def example_5_save_and_load():
    """Example 5: Save and load vector store."""
    print("\n" + "=" * 70)
    print("Example 5: Save and Load Vector Store")
    print("=" * 70)
    
    # Create store
    store = example_4_metadata_filtering()
    
    # Save
    output_dir = "output/vector_store/class10_math_ch5"
    store.save(output_dir)
    
    print(f"\nVector store saved to: {output_dir}")
    print("Files created:")
    for file in Path(output_dir).iterdir():
        print(f"  - {file.name} ({file.stat().st_size} bytes)")
    
    # Load
    print("\nLoading vector store...")
    loaded_store = VectorStore.load(output_dir)
    
    print("Loaded successfully!")
    print(f"Statistics: {loaded_store.get_statistics()}")
    
    return loaded_store


def example_6_complete_pipeline():
    """Example 6: Complete pipeline from chunks to searchable store."""
    print("\n" + "=" * 70)
    print("Example 6: Complete Pipeline")
    print("=" * 70)
    
    # Simulate loading chunks from Phase 3
    print("\n1. Load chunks from Phase 3 output...")
    
    # In real scenario, load from JSON:
    # with open('output/chunks/class10_math_ch5_chunks.json', 'r') as f:
    #     data = json.load(f)
    #     chunks = [Chunk.from_dict(c) for c in data['chunks']]
    
    # For demo, use sample chunks
    chunks = [
        Chunk(
            content="Definition: An arithmetic progression is a sequence...",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[95], language="eng", chunk_type="definition",
                chunk_id="10_mathematics_5_001", token_count=45, char_count=200,
                has_equations=False, has_examples=False, has_exercises=False,
                structure_confidence=1.0, completeness="complete"
            )
        ),
        Chunk(
            content="Example: For the AP 2, 7, 12, find the 10th term...",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[96], language="eng", chunk_type="example",
                chunk_id="10_mathematics_5_002", token_count=50, char_count=180,
                has_equations=True, has_examples=True, has_exercises=False,
                structure_confidence=0.95, completeness="complete"
            )
        )
    ]
    
    print(f"Loaded {len(chunks)} chunks")
    
    # 2. Generate embeddings
    print("\n2. Generate embeddings...")
    generator = EmbeddingGenerator()
    embeddings = generator.embed_chunks(chunks)
    print(f"Generated embeddings: {embeddings.shape}")
    
    # 3. Create vector store
    print("\n3. Create FAISS vector store...")
    config = VectorStoreConfig(
        embedding_dim=embeddings.shape[1],
        normalize_vectors=True
    )
    store = VectorStore(config)
    store.add_chunks(chunks, embeddings)
    print(f"Added {len(chunks)} vectors to store")
    
    # 4. Save
    print("\n4. Save vector store...")
    output_dir = "output/vector_store/complete_pipeline"
    store.save(output_dir)
    print(f"Saved to: {output_dir}")
    
    # 5. Demonstrate filtering
    print("\n5. Test metadata filtering...")
    filter_obj = (FilterBuilder()
        .for_class(10)
        .for_subject("mathematics")
        .for_chapter(5)
        .build())
    
    matching_ids = filter_obj.apply(store.metadata_store)
    print(f"Filter: {filter_obj.get_filter_summary()}")
    print(f"Matched: {len(matching_ids)}/{len(chunks)} vectors")
    
    # 6. Ready for retrieval
    print("\n6. Vector store ready for retrieval!")
    print("   Next step: Implement query processing and retrieval")
    
    return store


def example_7_benchmark_performance():
    """Example 7: Benchmark embedding and indexing performance."""
    print("\n" + "=" * 70)
    print("Example 7: Performance Benchmarking")
    print("=" * 70)
    
    import time
    
    generator = EmbeddingGenerator()
    
    # Test different batch sizes
    print("\nEmbedding Speed Test:")
    print("-" * 70)
    
    test_sizes = [10, 50, 100, 500]
    sample_text = "This is a sample text for performance testing. " * 10
    
    for size in test_sizes:
        texts = [sample_text] * size
        
        start = time.time()
        embeddings = generator.embed_text(texts)
        elapsed = time.time() - start
        
        print(f"Batch size {size:4d}: {elapsed:.3f}s ({size/elapsed:.1f} texts/sec)")
    
    # Test FAISS indexing speed
    print("\nFAISS Indexing Speed Test:")
    print("-" * 70)
    
    for size in [100, 1000, 10000]:
        # Generate random embeddings
        embeddings = np.random.randn(size, 768).astype(np.float32)
        metadata = [{'id': i} for i in range(size)]
        
        config = VectorStoreConfig(embedding_dim=768)
        store = VectorStore(config)
        
        start = time.time()
        store.add_embeddings(embeddings, metadata)
        elapsed = time.time() - start
        
        print(f"Index size {size:5d}: {elapsed:.3f}s ({size/elapsed:.0f} vectors/sec)")
    
    print("\n✓ Performance benchmarking complete")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MULTILINGUAL EMBEDDING & VECTOR STORAGE - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        # Run examples
        example_1_basic_embedding()
        chunks, embeddings = example_2_embed_chunks()
        store = example_3_create_vector_store()
        example_4_metadata_filtering()
        example_5_save_and_load()
        example_6_complete_pipeline()
        example_7_benchmark_performance()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  ✓ Multilingual embeddings work without translation")
        print("  ✓ FAISS provides fast, CPU-optimized vector search")
        print("  ✓ Metadata filtering enables grade-appropriate retrieval")
        print("  ✓ Complete pipeline: chunks → embeddings → searchable store")
        print("\nNext Steps:")
        print("  → Implement query processing")
        print("  → Build retrieval pipeline with re-ranking")
        print("  → Integrate with LLM for answer generation")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
