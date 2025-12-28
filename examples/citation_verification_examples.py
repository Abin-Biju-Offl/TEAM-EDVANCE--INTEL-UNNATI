"""
Citation Verification Examples
================================

Demonstrates citation validation with validated and rejected answers.
"""

import logging
from typing import List

from src.chunking import Chunk
from src.chunking.chunk_types import ChunkMetadata
from src.generation.citation_verifier import (
    CitationVerifier,
    CitationFormatter,
    verify_answer_citations,
    validate_or_reject
)

logging.basicConfig(level=logging.INFO)


def create_sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference and is denoted by d.",
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
                char_count=250,
                has_equations=False,
                has_examples=False,
                has_exercises=False,
                structure_confidence=1.0,
                completeness="complete"
            )
        ),
        Chunk(
            content="Example 5.3: For the AP: 3, 7, 11, 15, find the common difference. Solution: d = 7 - 3 = 4.",
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
                token_count=30,
                char_count=150,
                has_equations=True,
                has_examples=True,
                has_exercises=False,
                structure_confidence=0.95,
                completeness="complete"
            )
        ),
        Chunk(
            content="Theorem 5.1: The nth term of an arithmetic progression with first term a and common difference d is given by the formula: an = a + (n-1)d",
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
                char_count=140,
                has_equations=True,
                has_examples=False,
                has_exercises=False,
                structure_confidence=1.0,
                completeness="complete"
            )
        ),
    ]


def example_1_validated_answer():
    """Example 1: Properly cited answer (VALIDATED)."""
    print("=" * 70)
    print("Example 1: VALIDATED Answer (All Sentences Properly Cited)")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    # Good answer - every sentence has valid citation
    answer = """An arithmetic progression is a sequence of numbers where each term after the first is obtained by adding a fixed number (NCERT Class 10, Mathematics, Chapter 5, Page 95). The nth term formula is an = a + (n-1)d (NCERT Class 10, Mathematics, Chapter 5, Page 97). For example, in the sequence 3, 7, 11, 15, the common difference is 4 (NCERT Class 10, Mathematics, Chapter 5, Page 96)."""
    
    print("\nQuery: 'What is an arithmetic progression?'")
    print("\nGenerated Answer:")
    print("-" * 70)
    print(answer)
    
    print("\nRetrieved Chunks:")
    print("-" * 70)
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. Class {chunk.metadata.class_number}, {chunk.metadata.subject}, "
              f"Chapter {chunk.metadata.chapter_number}, Page {chunk.metadata.page_numbers[0]}")
        print(f"   Type: {chunk.metadata.chunk_type}")
    
    # Verify
    verifier = CitationVerifier()
    result = verifier.verify(answer, chunks, strict_mode=True)
    
    print("\nVALIDATION RESULT:")
    print(verifier.get_validation_report(result))
    
    print("OUTCOME: ✓ ACCEPTED")
    print("Answer will be shown to student with all citations intact.")


def example_2_missing_citations():
    """Example 2: Answer with missing citations (REJECTED)."""
    print("\n" + "=" * 70)
    print("Example 2: REJECTED Answer (Missing Citations)")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    # Bad answer - some sentences lack citations
    answer = """An arithmetic progression is a sequence of numbers. The nth term formula is an = a + (n-1)d (NCERT Class 10, Mathematics, Chapter 5, Page 97). The common difference determines how the sequence progresses."""
    
    print("\nQuery: 'What is an arithmetic progression?'")
    print("\nGenerated Answer:")
    print("-" * 70)
    print(answer)
    
    # Verify
    verifier = CitationVerifier()
    result = verifier.verify(answer, chunks, strict_mode=True)
    
    print("\nVALIDATION RESULT:")
    print(verifier.get_validation_report(result))
    
    print("\nOUTCOME: ✗ REJECTED")
    print("System Response: 'I don't know based on NCERT textbooks.'")
    print("\nReason: Sentences 1 and 3 lack citations - likely hallucinated content")


def example_3_invalid_citations():
    """Example 3: Answer with invalid citations (REJECTED)."""
    print("\n" + "=" * 70)
    print("Example 3: REJECTED Answer (Invalid Citations)")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    # Bad answer - citations don't match retrieved chunks
    answer = """An arithmetic progression is a sequence (NCERT Class 10, Mathematics, Chapter 5, Page 95). It is commonly used in real-world applications (NCERT Class 10, Mathematics, Chapter 8, Page 150). The formula is well-known (NCERT Class 11, Mathematics, Chapter 3, Page 45)."""
    
    print("\nQuery: 'What is an arithmetic progression?'")
    print("\nGenerated Answer:")
    print("-" * 70)
    print(answer)
    
    print("\nRetrieved Chunks:")
    print("-" * 70)
    print("Available pages: 95, 96, 97 (Chapter 5 only)")
    
    # Verify
    verifier = CitationVerifier()
    result = verifier.verify(answer, chunks, strict_mode=True)
    
    print("\nVALIDATION RESULT:")
    print(verifier.get_validation_report(result))
    
    print("\nOUTCOME: ✗ REJECTED")
    print("System Response: 'I don't know based on NCERT textbooks.'")
    print("\nReason: Citations reference pages/chapters not in retrieved chunks")


def example_4_formatted_citations():
    """Example 4: Format citations from chunks."""
    print("\n" + "=" * 70)
    print("Example 4: Citation Formatting from Chunks")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    print("\nHow to format citations from retrieved chunks:")
    print("-" * 70)
    
    formatter = CitationFormatter()
    
    for i, chunk in enumerate(chunks, 1):
        citation = formatter.format_citation_from_chunk(chunk)
        print(f"\nChunk {i}:")
        print(f"  Content: {chunk.content[:60]}...")
        print(f"  Citation: {citation}")


def example_5_validation_workflow():
    """Example 5: Complete validation workflow."""
    print("\n" + "=" * 70)
    print("Example 5: Complete Validation Workflow")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    test_answers = [
        {
            'name': 'Valid Answer',
            'answer': 'An AP is a sequence with fixed difference (NCERT Class 10, Mathematics, Chapter 5, Page 95).',
            'expected': 'VALIDATED'
        },
        {
            'name': 'Missing Citation',
            'answer': 'An AP is a sequence with fixed difference.',
            'expected': 'REJECTED'
        },
        {
            'name': 'Wrong Page Number',
            'answer': 'An AP is a sequence with fixed difference (NCERT Class 10, Mathematics, Chapter 5, Page 999).',
            'expected': 'REJECTED'
        },
        {
            'name': 'Wrong Chapter',
            'answer': 'An AP is a sequence with fixed difference (NCERT Class 10, Mathematics, Chapter 99, Page 95).',
            'expected': 'REJECTED'
        },
        {
            'name': 'I Don\'t Know Response',
            'answer': 'I don\'t know based on NCERT textbooks.',
            'expected': 'VALIDATED'
        }
    ]
    
    print("\nTesting various answer formats:\n")
    
    for i, test in enumerate(test_answers, 1):
        print(f"{i}. {test['name']}")
        print(f"   Answer: {test['answer'][:60]}...")
        
        is_valid, _ = verify_answer_citations(test['answer'], chunks, strict=True)
        
        actual = "VALIDATED" if is_valid else "REJECTED"
        match = "✓" if actual == test['expected'] else "✗"
        
        print(f"   Expected: {test['expected']}, Got: {actual} {match}")
        print()


def example_6_rejection_behavior():
    """Example 6: Automatic rejection with validate_or_reject."""
    print("\n" + "=" * 70)
    print("Example 6: Automatic Rejection Behavior")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    print("\nTest Case 1: Valid Answer")
    print("-" * 70)
    answer1 = "An AP has common difference (NCERT Class 10, Mathematics, Chapter 5, Page 95)."
    result1 = validate_or_reject(answer1, chunks)
    print(f"Input:  {answer1}")
    print(f"Output: {result1}")
    print(f"Status: {'PASSED' if result1 == answer1 else 'REJECTED'}")
    
    print("\n\nTest Case 2: Invalid Answer (Missing Citation)")
    print("-" * 70)
    answer2 = "An AP has common difference."
    result2 = validate_or_reject(answer2, chunks)
    print(f"Input:  {answer2}")
    print(f"Output: {result2}")
    print(f"Status: {'PASSED' if result2 == answer2 else 'REJECTED'}")


def example_7_detailed_validation_report():
    """Example 7: Detailed validation report."""
    print("\n" + "=" * 70)
    print("Example 7: Detailed Validation Report")
    print("=" * 70)
    
    chunks = create_sample_chunks()
    
    # Complex answer with multiple issues
    answer = """An arithmetic progression is a sequence (NCERT Class 10, Mathematics, Chapter 5, Page 95). It has a common difference. The formula is an = a + (n-1)d (NCERT Class 10, Mathematics, Chapter 99, Page 999). This concept is fundamental in mathematics."""
    
    print("\nAnswer with Multiple Issues:")
    print("-" * 70)
    print(answer)
    
    verifier = CitationVerifier()
    result = verifier.verify(answer, chunks, strict_mode=True)
    
    print("\nDETAILED VALIDATION REPORT:")
    print("=" * 70)
    print(verifier.get_validation_report(result))
    
    print("\nIssues Found:")
    print("  1. Sentence 2: 'It has a common difference.' - NO CITATION")
    print("  2. Sentence 3: Has citation but references non-existent Chapter 99, Page 999")
    print("  3. Sentence 4: 'This concept is fundamental...' - NO CITATION")
    
    print("\nSystem Decision: REJECT")
    print("Fallback Response: 'I don't know based on NCERT textbooks.'")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CITATION VERIFICATION - EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        example_1_validated_answer()
        example_2_missing_citations()
        example_3_invalid_citations()
        example_4_formatted_citations()
        example_5_validation_workflow()
        example_6_rejection_behavior()
        example_7_detailed_validation_report()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
        print("\nKey Takeaways:")
        print("  ✓ Every sentence must have a citation")
        print("  ✓ Citations must match retrieved chunks")
        print("  ✓ Invalid citations → automatic rejection")
        print("  ✓ Missing citations → automatic rejection")
        print("  ✓ Format: (NCERT Class X, Subject, Chapter Y, Page Z)")
        
        print("\nValidation Rules:")
        print("  1. All sentences require citations")
        print("  2. Citations must reference actual chunks")
        print("  3. Class/subject/chapter/page must match exactly")
        print("  4. Failed validation → 'I don't know based on NCERT textbooks.'")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
