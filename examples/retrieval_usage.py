"""
Multi-Stage Retrieval Pipeline Usage Examples
==============================================

Complete examples demonstrating:
1. Basic retrieval with metadata filtering
2. Retrieval with reranking
3. Confidence scoring and hallucination detection
4. Complete pipeline with all stages
5. Batch retrieval for multiple queries
6. Low-confidence handling strategies
"""

import logging
from pathlib import Path
import json
import numpy as np

from src.embeddings import EmbeddingGenerator, VectorStore
from src.retrieval import RetrievalPipeline, RetrievalConfig
from src.retrieval.confidence_scorer import ConfidenceThresholds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_store():
    """Create a sample vector store for testing."""
    from src.chunking import Chunk
    from src.chunking.chunk_types import ChunkMetadata
    
    # Sample chunks
    chunks = [
        Chunk(
            content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference and is denoted by d.",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions", source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[95], language="eng", chunk_type="definition",
                chunk_id="10_mathematics_5_001", token_count=45, char_count=250,
                has_equations=False, has_examples=False, has_exercises=False,
                structure_confidence=1.0, completeness="complete"
            )
        ),
        Chunk(
            content="Example 5.3: For the AP: 3, 7, 11, 15, find the common difference.\n\nSolution: The common difference d = 7 - 3 = 4. We can verify: 11 - 7 = 4 and 15 - 11 = 4. Therefore, the common difference d = 4.",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions", source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[96], language="eng", chunk_type="example",
                chunk_id="10_mathematics_5_002", token_count=52, char_count=200,
                has_equations=True, has_examples=True, has_exercises=False,
                structure_confidence=0.95, completeness="complete"
            )
        ),
        Chunk(
            content="Theorem 5.1: The nth term of an arithmetic progression with first term a and common difference d is given by the formula: an = a + (n-1)d",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions", source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[97], language="eng", chunk_type="theorem",
                chunk_id="10_mathematics_5_003", token_count=35, char_count=150,
                has_equations=True, has_examples=False, has_exercises=False,
                structure_confidence=1.0, completeness="complete"
            )
        ),
        Chunk(
            content="Note: In an arithmetic progression, if the common difference is positive, the terms increase. If it is negative, the terms decrease. When d = 0, all terms are equal.",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions", source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[95], language="eng", chunk_type="note",
                chunk_id="10_mathematics_5_004", token_count=38, char_count=170,
                has_equations=False, has_examples=False, has_exercises=False,
                structure_confidence=0.9, completeness="complete"
            )
        ),
        Chunk(
            content="Exercise 5.2: 1. Find the 10th term of the AP: 2, 7, 12, ... 2. Which term of the AP: 21, 18, 15, ... is -81? 3. Determine the AP whose 3rd term is 16 and 7th term is 36.",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions", source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[98], language="eng", chunk_type="exercise",
                chunk_id="10_mathematics_5_005", token_count=48, char_count=180,
                has_equations=True, has_examples=False, has_exercises=True,
                structure_confidence=0.95, completeness="complete"
            )
        )
    ]
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_chunks(chunks)
    
    # Create vector store
    from src.embeddings.vector_store import VectorStoreConfig
    config = VectorStoreConfig(embedding_dim=embeddings.shape[1])
    store = VectorStore(config)
    store.add_chunks(chunks, embeddings)
    
    return store, generator


def example_1_basic_retrieval():
    """Example 1: Basic retrieval with metadata filtering."""
    print("=" * 70)
    print("Example 1: Basic Retrieval with Metadata Filtering")
    print("=" * 70)
    
    # Create sample data
    store, generator = create_sample_store()
    
    # Create retrieval pipeline (without reranking for speed)
    config = RetrievalConfig(
        enable_reranking=False,  # Disable for this example
        enable_confidence_scoring=False,
        initial_k=5
    )
    
    pipeline = RetrievalPipeline(store, generator, config)
    
    # Query
    query = "What is an arithmetic progression?"
    
    print(f"\nQuery: '{query}'")
    print(f"Filters: Class 10, Mathematics")
    
    # Retrieve
    results = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    
    print(f"\nResults: {len(results)} chunks retrieved")
    print("-" * 70)
    
    for result in results:
        print(f"\nRank {result.rank}: {result.chunk_id}")
        print(f"  Type: {result.metadata['chunk_type']}")
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Content: {result.content[:100]}...")
    
    return pipeline, results


def example_2_with_reranking():
    """Example 2: Retrieval with cross-encoder reranking."""
    print("\n" + "=" * 70)
    print("Example 2: Retrieval with Cross-Encoder Reranking")
    print("=" * 70)
    
    # Create sample data
    store, generator = create_sample_store()
    
    # Create pipeline WITH reranking
    config = RetrievalConfig(
        enable_reranking=True,
        enable_confidence_scoring=False,
        initial_k=5,
        final_k=3
    )
    
    pipeline = RetrievalPipeline(store, generator, config)
    
    query = "Show me an example of finding the common difference in an AP"
    
    print(f"\nQuery: '{query}'")
    print(f"Filters: Class 10, Mathematics")
    print(f"Pipeline: Metadata Filter → Vector Search (k={config.initial_k}) → Rerank (top {config.final_k})")
    
    # Retrieve
    results = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    
    print(f"\nResults: {len(results)} chunks after reranking")
    print("-" * 70)
    
    for result in results:
        print(f"\nRank {result.rank}: {result.chunk_id}")
        print(f"  Type: {result.metadata['chunk_type']}")
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Rerank Score: {result.rerank_score:.3f}")
        print(f"  Final Score: {result.final_score:.3f} (40% sim + 60% rerank)")
        print(f"  Content: {result.content[:100]}...")


def example_3_confidence_scoring():
    """Example 3: Confidence scoring and hallucination detection."""
    print("\n" + "=" * 70)
    print("Example 3: Confidence Scoring & Hallucination Detection")
    print("=" * 70)
    
    store, generator = create_sample_store()
    
    # Enable all stages
    config = RetrievalConfig(
        enable_reranking=True,
        enable_confidence_scoring=True,
        initial_k=5,
        final_k=3,
        low_confidence_threshold=0.6
    )
    
    pipeline = RetrievalPipeline(store, generator, config)
    
    # Test Case 1: Good query (should have high confidence)
    print("\n" + "-" * 70)
    print("Test Case 1: Good Query (Expected: High Confidence)")
    print("-" * 70)
    
    query1 = "What is the definition of arithmetic progression?"
    results1 = pipeline.retrieve(query=query1, class_number=10, subject="mathematics")
    
    print(f"Query: '{query1}'")
    print(f"\nTop Result:")
    top = results1[0]
    print(f"  Chunk: {top.chunk_id} ({top.metadata['chunk_type']})")
    print(f"  Similarity: {top.similarity_score:.3f}")
    print(f"  Rerank: {top.rerank_score:.3f}")
    print(f"  Final Score: {top.final_score:.3f}")
    print(f"  Confidence: {top.confidence:.3f} {'✓ HIGH' if top.confidence >= 0.8 else '⚠ LOW' if top.confidence < 0.6 else '~ MEDIUM'}")
    print(f"  Low Confidence? {top.is_low_confidence}")
    
    # Test Case 2: Ambiguous query (should have lower confidence)
    print("\n" + "-" * 70)
    print("Test Case 2: Off-Topic Query (Expected: Low Confidence)")
    print("-" * 70)
    
    query2 = "Explain photosynthesis process"  # Not in our sample data
    results2 = pipeline.retrieve(query=query2, class_number=10, subject="mathematics")
    
    print(f"Query: '{query2}'")
    if results2:
        top2 = results2[0]
        print(f"\nTop Result:")
        print(f"  Chunk: {top2.chunk_id} ({top2.metadata['chunk_type']})")
        print(f"  Similarity: {top2.similarity_score:.3f}")
        print(f"  Rerank: {top2.rerank_score:.3f}")
        print(f"  Confidence: {top2.confidence:.3f} {'✓ HIGH' if top2.confidence >= 0.8 else '⚠ LOW' if top2.confidence < 0.6 else '~ MEDIUM'}")
        print(f"  Low Confidence? {top2.is_low_confidence}")
        
        if top2.is_low_confidence:
            print("\n⚠ WARNING: Low confidence detected!")
            print("  → System should respond: 'I don't have enough information about this topic.'")
    else:
        print("  No results found (filtered by threshold)")


def example_4_complete_pipeline():
    """Example 4: Complete pipeline demonstration."""
    print("\n" + "=" * 70)
    print("Example 4: Complete Multi-Stage Pipeline")
    print("=" * 70)
    
    store, generator = create_sample_store()
    
    config = RetrievalConfig(
        enable_metadata_filtering=True,
        enable_reranking=True,
        enable_confidence_scoring=True,
        initial_k=10,
        final_k=3
    )
    
    pipeline = RetrievalPipeline(store, generator, config)
    
    query = "How do I find the nth term of an arithmetic progression?"
    
    print(f"\nQuery: '{query}'")
    print(f"Student Context: Class 10, Mathematics")
    print(f"\nPipeline Stages:")
    print(f"  1. Metadata Filtering: class=10, subject=mathematics")
    print(f"  2. Vector Search: Retrieve top-{config.initial_k} candidates")
    print(f"  3. Cross-Encoder Rerank: Rerank to top-{config.final_k}")
    print(f"  4. Confidence Scoring: Compute confidence for each result")
    
    results = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS: {len(results)} chunks")
    print(f"{'='*70}")
    
    for result in results:
        print(f"\n📄 Rank {result.rank}: {result.chunk_id}")
        print(f"   Type: {result.metadata['chunk_type'].upper()}")
        print(f"   Scores:")
        print(f"     • Similarity:  {result.similarity_score:.3f}")
        print(f"     • Rerank:      {result.rerank_score:.3f}")
        print(f"     • Final:       {result.final_score:.3f}")
        print(f"     • Confidence:  {result.confidence:.3f} {'✓' if result.confidence >= 0.8 else '⚠' if result.confidence < 0.6 else '~'}")
        print(f"   Content: {result.content[:120]}...")
        
        if result.is_low_confidence:
            print(f"   ⚠ WARNING: Low confidence - may not be relevant")


def example_5_hallucination_detection():
    """Example 5: Explicit hallucination detection."""
    print("\n" + "=" * 70)
    print("Example 5: Hallucination Detection Signals")
    print("=" * 70)
    
    from src.retrieval.confidence_scorer import ConfidenceScorer
    
    scorer = ConfidenceScorer()
    
    test_cases = [
        {
            'name': 'High Confidence (Safe)',
            'query': 'What is an arithmetic progression?',
            'content': 'Definition: An arithmetic progression is a sequence...',
            'similarity': 0.88,
            'rerank': 0.92,
            'metadata': {'chunk_type': 'definition', 'completeness': 'complete'}
        },
        {
            'name': 'Medium Confidence (Caution)',
            'query': 'What is an arithmetic progression?',
            'content': 'Exercise: Solve problems on arithmetic progressions...',
            'similarity': 0.68,
            'rerank': 0.71,
            'metadata': {'chunk_type': 'exercise', 'completeness': 'complete'}
        },
        {
            'name': 'Low Confidence (REJECT)',
            'query': 'What is an arithmetic progression?',
            'content': 'Note: Additional practice problems are available...',
            'similarity': 0.42,
            'rerank': 0.38,
            'metadata': {'chunk_type': 'note', 'completeness': 'partial'}
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'-'*70}")
        print(f"Test Case {i}: {case['name']}")
        print(f"{'-'*70}")
        
        confidence = scorer.compute_confidence(
            query=case['query'],
            retrieved_content=case['content'],
            similarity_score=case['similarity'],
            rerank_score=case['rerank'],
            metadata=case['metadata']
        )
        
        risk = scorer.detect_hallucination_risk(
            confidence=confidence,
            similarity_score=case['similarity'],
            rerank_score=case['rerank']
        )
        
        print(f"Query: '{case['query'][:60]}...'")
        print(f"Retrieved: '{case['content'][:60]}...'")
        print(f"\nScores:")
        print(f"  Similarity: {case['similarity']:.3f}")
        print(f"  Rerank:     {case['rerank']:.3f}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"\nHallucination Risk Assessment:")
        print(f"  Risk Level:    {risk['risk_level'].upper()}")
        print(f"  Should Reject: {risk['should_reject']}")
        print(f"  Reason:        {risk['reason']}")
        
        if risk['should_reject']:
            print(f"\n⚠ SYSTEM ACTION: Reject retrieval")
            print(f"  → Response: 'I don't have enough information to answer this question accurately.'")


def example_6_batch_queries():
    """Example 6: Batch retrieval for multiple queries."""
    print("\n" + "=" * 70)
    print("Example 6: Batch Retrieval for Multiple Queries")
    print("=" * 70)
    
    store, generator = create_sample_store()
    
    config = RetrievalConfig(
        enable_reranking=True,
        enable_confidence_scoring=True,
        initial_k=5,
        final_k=2
    )
    
    pipeline = RetrievalPipeline(store, generator, config)
    
    queries = [
        "What is an arithmetic progression?",
        "Give me an example of finding common difference",
        "What is the formula for nth term?",
    ]
    
    print(f"\nProcessing {len(queries)} queries...")
    print(f"Context: Class 10, Mathematics\n")
    
    for i, query in enumerate(queries, 1):
        print(f"{'-'*70}")
        print(f"Query {i}: '{query}'")
        print(f"{'-'*70}")
        
        results = pipeline.retrieve(
            query=query,
            class_number=10,
            subject="mathematics"
        )
        
        print(f"Top result:")
        if results:
            top = results[0]
            print(f"  • {top.metadata['chunk_type'].upper()}: {top.content[:80]}...")
            print(f"  • Score: {top.final_score:.3f}, Confidence: {top.confidence:.3f}")
            
            if top.is_low_confidence:
                print(f"  ⚠ Low confidence - handle carefully")
        else:
            print(f"  No results found")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("MULTI-STAGE RETRIEVAL PIPELINE - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        example_1_basic_retrieval()
        example_2_with_reranking()
        example_3_confidence_scoring()
        example_4_complete_pipeline()
        example_5_hallucination_detection()
        example_6_batch_queries()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
        print("\nKey Takeaways:")
        print("  ✓ Stage 1 (Metadata filtering) ensures grade-appropriate content")
        print("  ✓ Stage 2 (Vector search) finds semantic matches quickly")
        print("  ✓ Stage 3 (Reranking) improves accuracy with cross-encoder")
        print("  ✓ Confidence scoring detects hallucination risks")
        print("  ✓ Low-confidence results can be rejected or handled specially")
        
        print("\nNext Steps:")
        print("  → Integrate with LLM for answer generation")
        print("  → Add query expansion and intent classification")
        print("  → Implement conversation history tracking")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
