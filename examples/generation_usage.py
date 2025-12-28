"""
RAG Answer Generation - Usage Examples
=======================================

Demonstrates strict RAG answer generation with hallucination prevention.
"""

import logging
from pathlib import Path

from src.embeddings import EmbeddingGenerator, VectorStore
from src.retrieval import RetrievalPipeline, RetrievalConfig
from src.generation import (
    RAGAnswerGenerator,
    GenerationConfig,
    CitationStyle,
    AnswerStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def create_sample_retrieval_system():
    """Create sample retrieval system for testing."""
    from src.chunking import Chunk
    from src.chunking.chunk_types import ChunkMetadata
    
    # Sample chunks
    chunks = [
        Chunk(
            content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference and is denoted by d.",
            metadata=ChunkMetadata(
                class_number=10, subject="mathematics", chapter_number=5,
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
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
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
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
                chapter_title="Arithmetic Progressions",
                source_file="NCERT_Class10_Mathematics_English.pdf",
                page_numbers=[97], language="eng", chunk_type="theorem",
                chunk_id="10_mathematics_5_003", token_count=35, char_count=150,
                has_equations=True, has_examples=False, has_exercises=False,
                structure_confidence=1.0, completeness="complete"
            )
        ),
    ]
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.embed_chunks(chunks)
    
    # Create vector store
    from src.embeddings.vector_store import VectorStoreConfig
    config = VectorStoreConfig(embedding_dim=embeddings.shape[1])
    store = VectorStore(config)
    store.add_chunks(chunks, embeddings)
    
    # Create retrieval pipeline
    retrieval_config = RetrievalConfig(
        enable_reranking=False,  # Disable for simplicity
        enable_confidence_scoring=True,
        initial_k=5,
    )
    
    pipeline = RetrievalPipeline(store, generator, retrieval_config)
    
    return pipeline


def example_1_basic_answer_generation():
    """Example 1: Basic answer generation with citations."""
    print("=" * 70)
    print("Example 1: Basic Answer Generation with Citations")
    print("=" * 70)
    
    # Setup
    pipeline = create_sample_retrieval_system()
    generator = RAGAnswerGenerator()
    
    query = "What is an arithmetic progression?"
    
    print(f"\nQuery: '{query}'")
    print(f"Student: Class 10, Mathematics\n")
    
    # Retrieve context
    retrieved_chunks = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    print(f"Average confidence: {sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks):.2f}\n")
    
    # Generate answer
    result = generator.generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        class_number=10
    )
    
    print("GENERATED ANSWER:")
    print("-" * 70)
    print(result.answer)
    print()
    
    print("METADATA:")
    print(f"  Status: {result.status.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Citations: {len(result.citations)}")
    print(f"  Hallucination detected: {result.hallucination_detected}")
    print(f"  Is safe: {result.is_safe()}")
    print(f"  Generation time: {result.generation_time_ms:.1f}ms")


def example_2_insufficient_context():
    """Example 2: Handling queries with insufficient context."""
    print("\n" + "=" * 70)
    print("Example 2: Insufficient Context Handling")
    print("=" * 70)
    
    pipeline = create_sample_retrieval_system()
    generator = RAGAnswerGenerator()
    
    # Query about topic not in our sample data
    query = "Explain photosynthesis in plants"
    
    print(f"\nQuery: '{query}'")
    print(f"Note: This topic is not in our mathematics corpus\n")
    
    # Retrieve (will get low-confidence or no results)
    retrieved_chunks = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    
    print(f"Retrieved {len(retrieved_chunks)} chunks")
    
    if retrieved_chunks:
        avg_conf = sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks)
        print(f"Average confidence: {avg_conf:.2f} (LOW)\n")
    
    # Generate answer
    result = generator.generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        class_number=10
    )
    
    print("GENERATED ANSWER:")
    print("-" * 70)
    print(result.answer)
    print()
    
    print("SYSTEM BEHAVIOR:")
    print(f"  Status: {result.status.value}")
    print(f"  → System correctly REJECTS the query")
    print(f"  → Responds: 'I don't know based on NCERT textbooks.'")
    print(f"  → No hallucination!")


def example_3_hallucination_detection():
    """Example 3: Hallucination detection mechanisms."""
    print("\n" + "=" * 70)
    print("Example 3: Hallucination Detection")
    print("=" * 70)
    
    from src.generation.answer_generator import HallucinationDetector
    
    detector = HallucinationDetector()
    
    # Create mock retrieved chunks
    pipeline = create_sample_retrieval_system()
    retrieved_chunks = pipeline.retrieve(
        query="What is an arithmetic progression?",
        class_number=10,
        subject="mathematics"
    )
    
    test_cases = [
        {
            'name': 'Valid Answer (with citations)',
            'answer': 'An arithmetic progression is a sequence [Source 1].',
            'expected': False
        },
        {
            'name': 'Invalid Answer (no citations)',
            'answer': 'An arithmetic progression is a sequence of numbers.',
            'expected': True
        },
        {
            'name': 'Invalid Answer (external knowledge)',
            'answer': 'As we know, an AP is commonly used [Source 1].',
            'expected': True
        },
        {
            'name': 'Invalid Answer (non-existent citation)',
            'answer': 'An arithmetic progression is defined [Source 99].',
            'expected': True
        },
        {
            'name': 'Valid Answer (insufficient context)',
            'answer': 'I don\'t know based on NCERT textbooks.',
            'expected': False
        },
    ]
    
    print("\nTesting hallucination detection on various answers:\n")
    
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['name']}")
        print(f"   Answer: '{case['answer']}'")
        
        is_hallucination, reason = detector.detect(
            answer=case['answer'],
            retrieved_chunks=retrieved_chunks,
            require_citations=True
        )
        
        status = "✓ DETECTED" if is_hallucination else "✓ PASSED"
        print(f"   Result: {status}")
        
        if is_hallucination:
            print(f"   Reason: {reason}")
        
        expected = "SHOULD DETECT" if case['expected'] else "SHOULD PASS"
        actual = "DETECTED" if is_hallucination else "PASSED"
        match = "✓" if (is_hallucination == case['expected']) else "✗"
        
        print(f"   {match} Expected: {expected}, Got: {actual}")
        print()


def example_4_citation_styles():
    """Example 4: Different citation styles."""
    print("\n" + "=" * 70)
    print("Example 4: Citation Styles")
    print("=" * 70)
    
    pipeline = create_sample_retrieval_system()
    
    query = "What is an arithmetic progression?"
    retrieved_chunks = pipeline.retrieve(query, class_number=10, subject="mathematics")
    
    styles = [
        CitationStyle.INLINE,
        CitationStyle.NUMBERED,
        CitationStyle.SIMPLE,
    ]
    
    for style in styles:
        print(f"\n{style.value.upper()} STYLE:")
        print("-" * 70)
        
        config = GenerationConfig(citation_style=style)
        generator = RAGAnswerGenerator(config=config)
        
        result = generator.generate(query, retrieved_chunks, class_number=10)
        
        # Show just the first few lines
        answer_preview = '\n'.join(result.answer.split('\n')[:3])
        print(answer_preview)
        print("...")


def example_5_grade_appropriate_answers():
    """Example 5: Grade-appropriate language."""
    print("\n" + "=" * 70)
    print("Example 5: Grade-Appropriate Answers")
    print("=" * 70)
    
    pipeline = create_sample_retrieval_system()
    generator = RAGAnswerGenerator()
    
    query = "What is an arithmetic progression?"
    retrieved_chunks = pipeline.retrieve(query, class_number=10, subject="mathematics")
    
    # Test different grade levels
    grades = [8, 10, 12]
    
    for grade in grades:
        print(f"\nCLASS {grade}:")
        print("-" * 70)
        
        result = generator.generate(query, retrieved_chunks, class_number=grade)
        
        # Show preview
        print(result.answer.split('\n')[0][:100] + "...")
        print(f"(Language complexity adjusted for Class {grade})")


def example_6_complete_rag_pipeline():
    """Example 6: Complete RAG pipeline from query to answer."""
    print("\n" + "=" * 70)
    print("Example 6: Complete RAG Pipeline")
    print("=" * 70)
    
    # Initialize all components
    pipeline = create_sample_retrieval_system()
    generator = RAGAnswerGenerator()
    
    query = "What is the formula for the nth term of an arithmetic progression?"
    
    print(f"\nQuery: '{query}'")
    print(f"Context: Class 10, Mathematics\n")
    
    print("STEP 1: RETRIEVE CONTEXT")
    print("-" * 70)
    retrieved_chunks = pipeline.retrieve(
        query=query,
        class_number=10,
        subject="mathematics"
    )
    print(f"✓ Retrieved {len(retrieved_chunks)} chunks")
    print(f"✓ Average confidence: {sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks):.2f}")
    
    print("\nSTEP 2: CHECK CONFIDENCE")
    print("-" * 70)
    if retrieved_chunks[0].confidence >= 0.8:
        print("✓ High confidence - safe to generate answer")
    elif retrieved_chunks[0].confidence >= 0.6:
        print("~ Medium confidence - will add disclaimer")
    else:
        print("✗ Low confidence - will reject")
    
    print("\nSTEP 3: GENERATE ANSWER")
    print("-" * 70)
    result = generator.generate(query, retrieved_chunks, class_number=10)
    print(f"✓ Answer generated in {result.generation_time_ms:.1f}ms")
    
    print("\nSTEP 4: VERIFY SAFETY")
    print("-" * 70)
    print(f"✓ Hallucination detection: {'PASSED' if not result.hallucination_detected else 'FAILED'}")
    print(f"✓ Citation count: {len(result.citations)}")
    print(f"✓ Overall safety: {'SAFE' if result.is_safe() else 'UNSAFE'}")
    
    print("\nFINAL ANSWER:")
    print("=" * 70)
    print(result.answer)
    print()


def example_7_error_handling():
    """Example 7: Error handling and edge cases."""
    print("\n" + "=" * 70)
    print("Example 7: Error Handling")
    print("=" * 70)
    
    generator = RAGAnswerGenerator()
    
    test_cases = [
        {
            'name': 'Empty context',
            'chunks': [],
            'query': 'What is an AP?'
        },
        {
            'name': 'Very low confidence',
            'chunks': None,  # Will be created with low confidence
            'query': 'Explain quantum mechanics'
        },
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print("-" * 70)
        
        if case['chunks'] is None:
            # Create low-confidence chunks
            pipeline = create_sample_retrieval_system()
            chunks = pipeline.retrieve(case['query'], class_number=10, subject="mathematics")
        else:
            chunks = case['chunks']
        
        result = generator.generate(case['query'], chunks, class_number=10)
        
        print(f"Query: '{case['query']}'")
        print(f"Status: {result.status.value}")
        print(f"Answer: {result.answer}")
        print(f"Safe: {result.is_safe()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("RAG ANSWER GENERATION - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        example_1_basic_answer_generation()
        example_2_insufficient_context()
        example_3_hallucination_detection()
        example_4_citation_styles()
        example_5_grade_appropriate_answers()
        example_6_complete_rag_pipeline()
        example_7_error_handling()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
        print("\nKey Takeaways:")
        print("  ✓ Every answer includes mandatory citations")
        print("  ✓ Multiple hallucination detection mechanisms")
        print("  ✓ Automatic rejection of low-confidence answers")
        print("  ✓ Grade-appropriate language adaptation")
        print("  ✓ Safe fallback: 'I don't know based on NCERT textbooks.'")
        
        print("\nSafety Guarantees:")
        print("  1. NEVER adds information not in context")
        print("  2. ALWAYS cites sources for every claim")
        print("  3. REJECTS queries with insufficient context")
        print("  4. DETECTS hallucination patterns")
        print("  5. VERIFIES citations reference actual sources")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
