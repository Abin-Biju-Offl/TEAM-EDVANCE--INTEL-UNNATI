"""
Safety Mechanism Examples
=========================

Demonstrates "I don't know" safety mechanism with various rejection scenarios.
"""

import sys
sys.path.insert(0, '.')

from dataclasses import dataclass
from typing import List
from src.generation.safety_mechanism import (
    SafetyMechanism,
    SafetyThresholds,
    check_safety,
    get_safe_answer
)


@dataclass
class MockChunk:
    """Mock chunk for testing."""
    content: str
    confidence: float
    similarity_score: float
    metadata: dict


def create_high_quality_chunks() -> List[MockChunk]:
    """Create high-quality chunks."""
    return [
        MockChunk(
            content="An arithmetic progression (AP) is a sequence where each term after the first is obtained by adding a fixed number called the common difference.",
            confidence=0.92,
            similarity_score=0.88,
            metadata={'class_number': 10, 'subject': 'Mathematics', 'chapter_number': 5, 'page_numbers': [95]}
        ),
        MockChunk(
            content="The common difference d can be calculated as d = a₂ - a₁ where a₁ and a₂ are consecutive terms.",
            confidence=0.85,
            similarity_score=0.82,
            metadata={'class_number': 10, 'subject': 'Mathematics', 'chapter_number': 5, 'page_numbers': [95]}
        ),
        MockChunk(
            content="For the AP: 3, 7, 11, 15, ..., the common difference d = 7 - 3 = 4.",
            confidence=0.90,
            similarity_score=0.85,
            metadata={'class_number': 10, 'subject': 'Mathematics', 'chapter_number': 5, 'page_numbers': [96]}
        )
    ]


def create_low_quality_chunks() -> List[MockChunk]:
    """Create low-quality chunks with low confidence."""
    return [
        MockChunk(
            content="Some unrelated text about numbers.",
            confidence=0.35,
            similarity_score=0.32,
            metadata={'class_number': 10, 'subject': 'Mathematics', 'chapter_number': 3, 'page_numbers': [50]}
        )
    ]


def create_off_topic_chunks() -> List[MockChunk]:
    """Create chunks that indicate off-topic query."""
    return [
        MockChunk(
            content="The water cycle involves evaporation and precipitation.",
            confidence=0.28,
            similarity_score=0.25,
            metadata={'class_number': 10, 'subject': 'Science', 'chapter_number': 2, 'page_numbers': [30]}
        )
    ]


# ============================================================================
# EXAMPLE 1: PASSED - High Quality Answer
# ============================================================================

def example_1_passed_all_checks():
    """Example: All safety checks pass."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: PASSED - High Quality Answer")
    print("=" * 70)
    
    query = "What is an arithmetic progression?"
    chunks = create_high_quality_chunks()
    answer = """An arithmetic progression (AP) is a sequence where each term after the first is obtained by adding a fixed number called the common difference (NCERT Class 10, Mathematics, Chapter 5, Page 95).

For example, in the AP: 3, 7, 11, 15, ..., the common difference d = 7 - 3 = 4 (NCERT Class 10, Mathematics, Chapter 5, Page 96)."""
    
    result = check_safety(query, chunks, answer, class_number=10, subject="Mathematics")
    
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Average confidence: {sum(c.confidence for c in chunks) / len(chunks):.2f}")
    
    print(f"\n✅ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Overall confidence: {result.confidence_score:.2f}")
    
    print(f"\nPassed checks ({len(result.passed_checks)}):")
    for check in result.passed_checks:
        print(f"  ✓ {check}")
    
    if result.failed_checks:
        print(f"\nFailed checks ({len(result.failed_checks)}):")
        for check in result.failed_checks:
            print(f"  ✗ {check}")
    
    print(f"\nExplanation: {result.explanation}")
    
    # Use convenience function
    final_answer = get_safe_answer(query, chunks, answer, class_number=10, subject="Mathematics")
    print(f"\nFinal answer preview: {final_answer[:150]}...")


# ============================================================================
# EXAMPLE 2: REJECTED - Low Retrieval Confidence
# ============================================================================

def example_2_rejected_low_confidence():
    """Example: Rejection due to low retrieval confidence."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: REJECTED - Low Retrieval Confidence")
    print("=" * 70)
    
    query = "Explain quantum entanglement in simple terms"
    chunks = create_low_quality_chunks()
    answer = "Quantum entanglement is a phenomenon where particles are connected..."
    
    result = check_safety(query, chunks, answer)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Average confidence: {sum(c.confidence for c in chunks) / len(chunks):.2f}")
    
    print(f"\n❌ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Rejection reason: {result.rejection_reason.value if result.rejection_reason else 'N/A'}")
    
    print(f"\nPassed checks ({len(result.passed_checks)}):")
    for check in result.passed_checks:
        print(f"  ✓ {check}")
    
    print(f"\nFailed checks ({len(result.failed_checks)}):")
    for check in result.failed_checks:
        print(f"  ✗ {check}")
    
    print(f"\nExplanation: {result.explanation}")
    
    # Diagnostic details
    conf_details = result.diagnostic_info.get('confidence', {}).get('details', {})
    print(f"\nDiagnostic: Average confidence {conf_details.get('avg_confidence', 0):.2f} < threshold {conf_details.get('threshold', 0):.2f}")
    
    final_answer = get_safe_answer(query, chunks, answer)
    print(f"\nFinal answer: {final_answer}")


# ============================================================================
# EXAMPLE 3: REJECTED - Insufficient Context
# ============================================================================

def example_3_rejected_insufficient_context():
    """Example: Rejection due to no retrieved chunks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: REJECTED - Insufficient Context")
    print("=" * 70)
    
    query = "What is the capital of Mars?"
    chunks = []  # No chunks retrieved!
    answer = "The capital of Mars is..."
    
    result = check_safety(query, chunks, answer)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    
    print(f"\n❌ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Rejection reason: {result.rejection_reason.value if result.rejection_reason else 'N/A'}")
    
    print(f"\nFailed checks ({len(result.failed_checks)}):")
    for check in result.failed_checks:
        print(f"  ✗ {check}")
    
    print(f"\nExplanation: {result.explanation}")
    
    final_answer = get_safe_answer(query, chunks, answer)
    print(f"\nFinal answer: {final_answer}")


# ============================================================================
# EXAMPLE 4: REJECTED - Off-Topic Query
# ============================================================================

def example_4_rejected_off_topic():
    """Example: Rejection due to off-topic query."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: REJECTED - Off-Topic Query")
    print("=" * 70)
    
    query = "How do I make chocolate chip cookies?"
    chunks = create_off_topic_chunks()
    answer = "To make chocolate chip cookies, first mix flour and sugar..."
    
    result = check_safety(query, chunks, answer)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Best similarity score: {max(c.similarity_score for c in chunks):.2f}")
    
    print(f"\n❌ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Rejection reason: {result.rejection_reason.value if result.rejection_reason else 'N/A'}")
    
    print(f"\nFailed checks ({len(result.failed_checks)}):")
    for check in result.failed_checks:
        print(f"  ✗ {check}")
    
    print(f"\nExplanation: {result.explanation}")
    
    # Diagnostic details
    topic_details = result.diagnostic_info.get('topic', {}).get('details', {})
    print(f"\nDiagnostic: Best score {topic_details.get('best_score', 0):.2f} < threshold {topic_details.get('threshold', 0.3):.2f}")
    
    final_answer = get_safe_answer(query, chunks, answer)
    print(f"\nFinal answer: {final_answer}")


# ============================================================================
# EXAMPLE 5: REJECTED - Missing Citations
# ============================================================================

def example_5_rejected_missing_citations():
    """Example: Rejection due to missing citations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: REJECTED - Missing Citations")
    print("=" * 70)
    
    query = "What is an arithmetic progression?"
    chunks = create_high_quality_chunks()
    answer = "An arithmetic progression is a sequence where each term is obtained by adding a fixed number."
    # Note: No citations!
    
    result = check_safety(query, chunks, answer, class_number=10, subject="Mathematics")
    
    print(f"\nQuery: {query}")
    print(f"Answer has citations: NO")
    
    print(f"\n❌ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Rejection reason: {result.rejection_reason.value if result.rejection_reason else 'N/A'}")
    
    print(f"\nFailed checks ({len(result.failed_checks)}):")
    for check in result.failed_checks:
        print(f"  ✗ {check}")
    
    print(f"\nExplanation: {result.explanation}")
    
    final_answer = get_safe_answer(query, chunks, answer, class_number=10, subject="Mathematics")
    print(f"\nFinal answer: {final_answer}")


# ============================================================================
# EXAMPLE 6: Custom Thresholds
# ============================================================================

def example_6_custom_thresholds():
    """Example: Using custom safety thresholds."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Custom Thresholds (More Strict)")
    print("=" * 70)
    
    # Create stricter thresholds
    strict_thresholds = SafetyThresholds(
        min_retrieval_confidence=0.85,  # Default: 0.6
        min_chunks_required=3,           # Default: 1
        min_grounding_score=0.9          # Default: 0.7
    )
    
    query = "What is an arithmetic progression?"
    chunks = create_high_quality_chunks()[:2]  # Only 2 chunks (less than required 3)
    answer = """An arithmetic progression (AP) is a sequence where each term after the first is obtained by adding a fixed number (NCERT Class 10, Mathematics, Chapter 5, Page 95)."""
    
    mechanism = SafetyMechanism(strict_thresholds)
    result = mechanism.should_reject_query(query, chunks, answer)
    
    print(f"\nQuery: {query}")
    print(f"Retrieved chunks: {len(chunks)}")
    print(f"Minimum required: {strict_thresholds.min_chunks_required}")
    
    print(f"\n❌ DECISION: {'REJECT' if result.should_reject else 'ACCEPT'}")
    print(f"Rejection reason: {result.rejection_reason.value if result.rejection_reason else 'N/A'}")
    
    print(f"\nFailed checks ({len(result.failed_checks)}):")
    for check in result.failed_checks:
        print(f"  ✗ {check}")
    
    print(f"\nWith stricter thresholds, system is more conservative.")


# ============================================================================
# EXAMPLE 7: Complete Workflow with Safety
# ============================================================================

def example_7_complete_workflow():
    """Example: Complete RAG workflow with safety checks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Complete Workflow with Safety Mechanism")
    print("=" * 70)
    
    test_cases = [
        {
            'name': 'Valid query with good context',
            'query': 'What is an arithmetic progression?',
            'chunks': create_high_quality_chunks(),
            'answer': 'An arithmetic progression is a sequence... (NCERT Class 10, Mathematics, Chapter 5, Page 95).'
        },
        {
            'name': 'Valid query but low confidence',
            'query': 'Explain advanced calculus',
            'chunks': create_low_quality_chunks(),
            'answer': 'Calculus is...'
        },
        {
            'name': 'Off-topic query',
            'query': 'Who won the 2020 Olympics?',
            'chunks': create_off_topic_chunks(),
            'answer': 'The Olympics...'
        },
        {
            'name': 'No context available',
            'query': 'Unknown topic',
            'chunks': [],
            'answer': 'This is...'
        }
    ]
    
    print("\nTesting multiple scenarios:")
    print("-" * 70)
    
    for i, case in enumerate(test_cases, 1):
        result = check_safety(case['query'], case['chunks'], case['answer'])
        
        status = "✅ ACCEPT" if not result.should_reject else "❌ REJECT"
        reason = f" ({result.rejection_reason.value})" if result.rejection_reason else ""
        
        print(f"\n{i}. {case['name']}")
        print(f"   Query: {case['query'][:50]}...")
        print(f"   Chunks: {len(case['chunks'])}, Avg conf: {sum(c.confidence for c in case['chunks']) / len(case['chunks']) if case['chunks'] else 0:.2f}")
        print(f"   {status}{reason}")
    
    print("\n" + "=" * 70)
    print("Summary: System rejected 3/4 queries with safety issues")
    print("=" * 70)


# ============================================================================
# EXAMPLE 8: Diagnostic Information
# ============================================================================

def example_8_diagnostic_information():
    """Example: Detailed diagnostic information."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Detailed Diagnostic Information")
    print("=" * 70)
    
    query = "What is an arithmetic progression?"
    chunks = create_high_quality_chunks()
    answer = """An arithmetic progression is a sequence (NCERT Class 10, Mathematics, Chapter 5, Page 95)."""
    
    result = check_safety(query, chunks, answer, class_number=10, subject="Mathematics")
    
    print(f"\nQuery: {query}")
    print(f"\n{'ACCEPTED' if not result.should_reject else 'REJECTED'}")
    print(f"Overall Confidence: {result.confidence_score:.2%}")
    
    print("\n" + "-" * 70)
    print("DIAGNOSTIC BREAKDOWN")
    print("-" * 70)
    
    # Confidence check
    conf = result.diagnostic_info.get('confidence', {})
    print(f"\n1. Retrieval Confidence: {'✓ PASS' if conf.get('passed') else '✗ FAIL'}")
    conf_details = conf.get('details', {})
    print(f"   - Average: {conf_details.get('avg_confidence', 0):.2f}")
    print(f"   - Top score: {conf_details.get('top_confidence', 0):.2f}")
    print(f"   - Threshold: {conf_details.get('threshold', 0):.2f}")
    
    # Context check
    ctx = result.diagnostic_info.get('context', {})
    print(f"\n2. Context Sufficiency: {'✓ PASS' if ctx.get('passed') else '✗ FAIL'}")
    ctx_details = ctx.get('details', {})
    print(f"   - Chunks: {ctx_details.get('num_chunks', 0)}")
    print(f"   - Total length: {ctx_details.get('total_length', 0)} chars")
    
    # Topic check
    topic = result.diagnostic_info.get('topic', {})
    print(f"\n3. Topic Relevance: {'✓ PASS' if topic.get('passed') else '✗ FAIL'}")
    topic_details = topic.get('details', {})
    print(f"   - Best score: {topic_details.get('best_score', 0):.2f}")
    
    # Citation check
    if 'citations' in result.diagnostic_info:
        cit = result.diagnostic_info.get('citations', {})
        print(f"\n4. Citations: {'✓ PASS' if cit.get('passed') else '✗ FAIL'}")
        cit_details = cit.get('details', {})
        print(f"   - Cited sentences: {cit_details.get('cited_sentences', 0)}")
        print(f"   - Valid citations: {cit_details.get('valid_citations', 0)}")
    
    # Grounding check
    if 'grounding' in result.diagnostic_info:
        grd = result.diagnostic_info.get('grounding', {})
        print(f"\n5. Answer Grounding: {'✓ PASS' if grd.get('passed') else '✗ FAIL'}")
        grd_details = grd.get('details', {})
        print(f"   - Grounding score: {grd_details.get('grounding_score', 0):.2f}")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("SAFETY MECHANISM DEMONSTRATION")
    print("'I Don't Know' Decision Logic")
    print("=" * 70)
    
    example_1_passed_all_checks()
    example_2_rejected_low_confidence()
    example_3_rejected_insufficient_context()
    example_4_rejected_off_topic()
    example_5_rejected_missing_citations()
    example_6_custom_thresholds()
    example_7_complete_workflow()
    example_8_diagnostic_information()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Multiple safety checks protect against bad answers")
    print("2. 'I don't know' is returned when any check fails")
    print("3. Thresholds are configurable for different use cases")
    print("4. Detailed diagnostics help debug rejections")
    print("5. Conservative rejection improves system reliability")
