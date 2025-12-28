"""
Complete Chunking Pipeline Usage Examples
==========================================

Demonstrates curriculum-aware semantic chunking for NCERT content.
"""

import logging
from pathlib import Path
import json

from src.cleaning.text_cleaner import TextCleaner
from src.cleaning.structure_recovery import StructureRecovery
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.chunk_types import ChunkType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_basic_chunking():
    """Example: Basic chunking workflow."""
    print("=" * 70)
    print("Example 1: Basic Chunking Workflow")
    print("=" * 70)
    
    # Sample cleaned text with structures
    cleaned_text = """
Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference.

Example 5.3: For the AP: 3, 7, 11, 15, ..., find the common difference.

Solution: The common difference d = 7 - 3 = 4. We can verify: 11 - 7 = 4 and 15 - 11 = 4. Therefore, d = 4.

Theorem 5.1: The nth term of an AP with first term a and common difference d is given by the formula an = a + (n-1)d.

Exercise 5.2
1. Find the 10th term of the AP: 2, 7, 12, ...
2. Which term of the AP: 21, 18, 15, ... is -81?
3. Determine the AP whose 3rd term is 16 and 7th term is 36.
"""
    
    # Simulate structures (in real use, from StructureRecovery)
    from src.cleaning.structure_recovery import EducationalBlock, StructureType
    
    structures = [
        EducationalBlock(
            structure_type=StructureType.DEFINITION,
            content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term. This fixed number is called the common difference.",
            start_line=0,
            end_line=0,
            number="5.1",
            confidence=1.0
        ),
        EducationalBlock(
            structure_type=StructureType.EXAMPLE,
            content="Example 5.3: For the AP: 3, 7, 11, 15, ..., find the common difference.",
            start_line=2,
            end_line=2,
            number="5.3",
            confidence=1.0
        ),
        EducationalBlock(
            structure_type=StructureType.SOLUTION,
            content="Solution: The common difference d = 7 - 3 = 4. We can verify: 11 - 7 = 4 and 15 - 11 = 4. Therefore, d = 4.",
            start_line=4,
            end_line=4,
            confidence=1.0
        ),
        EducationalBlock(
            structure_type=StructureType.THEOREM,
            content="Theorem 5.1: The nth term of an AP with first term a and common difference d is given by the formula an = a + (n-1)d.",
            start_line=6,
            end_line=6,
            number="5.1",
            confidence=1.0
        ),
    ]
    
    # Base metadata
    base_metadata = {
        'class_number': 10,
        'subject': 'mathematics',
        'chapter_number': 5,
        'chapter_title': 'Arithmetic Progressions',
        'source_file': 'NCERT_Class10_Mathematics_English.pdf',
        'page_numbers': [95, 96],
        'language': 'eng'
    }
    
    # Create chunker
    chunker = SemanticChunker(
        min_chunk_size=300,
        max_chunk_size=500,
        overlap_size=50
    )
    
    # Chunk the document
    chunks = chunker.chunk_document(cleaned_text, structures, base_metadata)
    
    print(f"\nCreated {len(chunks)} chunks:")
    print("-" * 70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  ID: {chunk.metadata.chunk_id}")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Has equations: {chunk.metadata.has_equations}")
        print(f"  Completeness: {chunk.metadata.completeness}")
        print(f"  Content preview: {chunk.content[:100]}...")
    
    # Statistics
    stats = chunker.get_statistics()
    print(f"\n\nChunking Statistics:")
    print("-" * 70)
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_complete_pipeline():
    """Example: Complete pipeline from OCR to chunks."""
    print("\n" + "=" * 70)
    print("Example 2: Complete Pipeline (OCR → Clean → Structure → Chunk)")
    print("=" * 70)
    
    # Sample OCR output
    ocr_text = """
Page  95

Definition  5.1  :  An  arithmetic  progression  is  a
sequence  where  the  difference  between  consecutive
terms  is  constant .  This  constant  is  called  the
common  difference  and  is  denoted  by  d .

Example  5.3  :  Consider  the  sequence  3,  7,  11,  15 .
Is  this  an  AP ?  If  yes,  find  the  common  difference .

Solution  :  Check  differences :  7-3=4,  11-7=4,  15-11=4 .
Since  all  differences  are  equal ,  it  is  an  AP  with  d=4 .

Note  :  The  common  difference  can  be  positive,  negative,
or  zero .  When  d=0,  all  terms  are  equal .
"""
    
    print("\n1. Original OCR Text:")
    print("-" * 70)
    print(ocr_text[:200] + "...")
    
    # Step 1: Clean
    cleaner = TextCleaner()
    cleaned_text, cleaning_stats = cleaner.clean(ocr_text)
    
    print("\n2. Cleaned Text:")
    print("-" * 70)
    print(cleaned_text[:300] + "...")
    print(f"\nCleaning reduced text by {cleaning_stats.to_dict()['reduction_percent']}%")
    
    # Step 2: Identify structures
    structure_recovery = StructureRecovery()
    structures = structure_recovery.identify_structures(cleaned_text)
    
    print(f"\n3. Structures Identified: {len(structures)}")
    print("-" * 70)
    for struct in structures:
        print(f"  - {struct.structure_type.value.upper()}: {struct.number or ''}")
    
    # Step 3: Chunk
    base_metadata = {
        'class_number': 10,
        'subject': 'mathematics',
        'chapter_number': 5,
        'chapter_title': 'Arithmetic Progressions',
        'source_file': 'NCERT_Class10_Mathematics_English.pdf',
        'page_numbers': [95],
        'language': 'eng'
    }
    
    chunker = SemanticChunker(
        min_chunk_size=300,
        max_chunk_size=500,
        overlap_size=50
    )
    
    chunks = chunker.chunk_document(cleaned_text, structures, base_metadata)
    
    print(f"\n4. Chunks Created: {len(chunks)}")
    print("-" * 70)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}: {chunk.metadata.chunk_id}")
        print(f"  Type: {chunk.metadata.chunk_type}")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Content: {chunk.content[:150]}...")
    
    return chunks


def example_boundary_preservation():
    """Example: Demonstrate curriculum boundary preservation."""
    print("\n" + "=" * 70)
    print("Example 3: Curriculum Boundary Preservation")
    print("=" * 70)
    
    print("\n✓ RULE: Never mix chapters, subjects, or classes")
    print("-" * 70)
    
    # Metadata for Chapter 5
    metadata_ch5 = {
        'class_number': 10,
        'subject': 'mathematics',
        'chapter_number': 5,
        'chapter_title': 'Arithmetic Progressions',
        'source_file': 'NCERT_Class10_Mathematics_English.pdf',
        'page_numbers': [95],
        'language': 'eng'
    }
    
    # Metadata for Chapter 6 (different chapter)
    metadata_ch6 = {
        'class_number': 10,
        'subject': 'mathematics',
        'chapter_number': 6,
        'chapter_title': 'Triangles',
        'source_file': 'NCERT_Class10_Mathematics_English.pdf',
        'page_numbers': [105],
        'language': 'eng'
    }
    
    print("\nChapter 5 chunks will have IDs: 10_mathematics_5_001, 10_mathematics_5_002, ...")
    print("Chapter 6 chunks will have IDs: 10_mathematics_6_001, 10_mathematics_6_002, ...")
    print("\n⚠ These will NEVER be mixed in retrieval due to metadata filtering")
    
    print("\n✓ RULE: Preserve logical units")
    print("-" * 70)
    print("- Definitions: Always atomic (never split)")
    print("- Theorems: Always atomic (never split)")
    print("- Examples: Kept with solutions")
    print("- Proofs: Complete logical unit")
    print("- Explanations: Can be split with overlap")


def example_chunk_types_and_retrieval():
    """Example: How chunk types improve retrieval."""
    print("\n" + "=" * 70)
    print("Example 4: Chunk Types and Retrieval Optimization")
    print("=" * 70)
    
    from src.chunking.chunk_types import CHUNK_TYPE_CHARACTERISTICS
    
    print("\nChunk Type Characteristics for Retrieval:")
    print("-" * 70)
    
    for chunk_type, chars in CHUNK_TYPE_CHARACTERISTICS.items():
        print(f"\n{chunk_type.value.upper()}:")
        print(f"  Priority: {chars['priority']}")
        print(f"  Retrieval Weight: {chars['retrieval_weight']}x")
        print(f"  Atomic: {chars['should_be_atomic']}")
        print(f"  Requires Context: {chars['requires_context']}")
    
    print("\n\nRetrieval Strategy:")
    print("-" * 70)
    print("When user asks: 'What is an arithmetic progression?'")
    print("  → Prioritize DEFINITION chunks (1.5x weight)")
    print("  → DEFINITION chunks are atomic and self-contained")
    print("  → Can be directly returned without additional context")
    
    print("\nWhen user asks: 'How do I solve this AP problem?'")
    print("  → Prioritize EXAMPLE chunks (1.3x weight)")
    print("  → Include complete example with solution")
    print("  → May include related FORMULA chunks")


def example_save_chunks():
    """Example: Save chunks for vector database."""
    print("\n" + "=" * 70)
    print("Example 5: Save Chunks for Vector Database")
    print("=" * 70)
    
    # Create sample chunks
    from src.chunking.chunk_types import ChunkMetadata
    from src.chunking.semantic_chunker import Chunk
    
    chunk = Chunk(
        content="Definition 5.1: An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term.",
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
            char_count=175,
            has_equations=False,
            has_examples=False,
            has_exercises=False,
            structure_confidence=1.0,
            completeness="complete"
        )
    )
    
    # Convert to dict for storage
    chunk_dict = chunk.to_dict()
    
    # Save to file
    output_path = Path('output/chunks/class10_math_ch5_chunks.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    chunks_data = {
        'document_info': {
            'class': 10,
            'subject': 'mathematics',
            'chapter': 5,
            'total_chunks': 1
        },
        'chunks': [chunk_dict]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved chunks to: {output_path}")
    print("\nChunk format for vector database:")
    print("-" * 70)
    print(json.dumps(chunk_dict, indent=2, ensure_ascii=False)[:500] + "...")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CURRICULUM-AWARE SEMANTIC CHUNKING - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        example_basic_chunking()
        chunks = example_complete_pipeline()
        example_boundary_preservation()
        example_chunk_types_and_retrieval()
        example_save_chunks()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
