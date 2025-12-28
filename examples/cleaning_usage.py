"""
Complete Cleaning Pipeline Usage Examples
==========================================

Demonstrates how to use the text cleaning and structure recovery system
for NCERT textbook content.
"""

import logging
from pathlib import Path
import json

from src.cleaning.text_cleaner import TextCleaner, CleaningStats
from src.cleaning.structure_recovery import StructureRecovery, EducationalBlock
from src.ingestion.pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_basic_cleaning():
    """Example: Basic text cleaning."""
    print("=" * 70)
    print("Example 1: Basic Text Cleaning")
    print("=" * 70)
    
    # Sample OCR output with common errors
    dirty_text = """
M a t h e m a t i c s
Page  95
Chapter  5  :  Arithmetic  Progressions

In  this  chapter ,  we  will  1earn  about  arith-
metic  progressions .  An  arithmetic  progression  is
a  sequence  where  each  term  differs  from  the
previous  0ne  by  a  constant  amount .

--------------------------------------------
95
"""
    
    # Initialize cleaner
    cleaner = TextCleaner(
        preserve_equations=True,
        preserve_special_formatting=True,
        aggressive_deduplication=False
    )
    
    # Clean the text
    cleaned_text, stats = cleaner.clean(dirty_text)
    
    print("\nOriginal text:")
    print("-" * 70)
    print(dirty_text)
    
    print("\nCleaned text:")
    print("-" * 70)
    print(cleaned_text)
    
    print("\nCleaning Statistics:")
    print("-" * 70)
    for key, value in stats.to_dict().items():
        print(f"  {key}: {value}")


def example_structure_recovery():
    """Example: Identifying educational structures."""
    print("\n" + "=" * 70)
    print("Example 2: Educational Structure Recovery")
    print("=" * 70)
    
    # Sample cleaned text with educational structures
    text = """
Chapter 5: Arithmetic Progressions

Definition 1: A sequence is called an arithmetic progression if the difference between consecutive terms is constant.

Example 5.2: Check whether 2, 5, 8, 11, 14 is an AP.

Solution: Here a1 = 2, a2 = 5, a3 = 8
a2 - a1 = 5 - 2 = 3
a3 - a2 = 8 - 5 = 3
Since the difference is constant, it is an AP.

Note: The common difference can be positive, negative, or zero.

Exercise 5.1

1. In which of the following situations, do the lists of numbers form an AP?
(a) The cost of digging a well for first meter is Rs 150
(b) The amount of money in the account every year
"""
    
    # Initialize structure recovery
    structure_recovery = StructureRecovery(
        preserve_numbering=True,
        detect_implicit_structures=True
    )
    
    # Identify structures
    blocks = structure_recovery.identify_structures(text)
    
    print(f"\nFound {len(blocks)} educational structures:")
    print("-" * 70)
    
    for i, block in enumerate(blocks, 1):
        print(f"\n{i}. {block.structure_type.value.upper()}")
        if block.number:
            print(f"   Number: {block.number}")
        if block.title:
            print(f"   Title: {block.title}")
        print(f"   Lines: {block.start_line} - {block.end_line}")
        print(f"   Content preview: {block.content[:100]}...")
    
    # Get summary
    summary = structure_recovery.get_structure_summary(blocks)
    print("\nStructure Summary:")
    print("-" * 70)
    print(f"Total blocks: {summary['total_blocks']}")
    print("By type:")
    for struct_type, count in summary['by_type'].items():
        print(f"  {struct_type}: {count}")


def example_complete_pipeline():
    """Example: Complete pipeline from OCR to cleaned structured text."""
    print("\n" + "=" * 70)
    print("Example 3: Complete Pipeline (OCR → Cleaning → Structure)")
    print("=" * 70)
    
    # Simulate OCR output
    ocr_output = """
Page  95
NCERT

Definition  5.1  :  An  arithmetic  progression  (AP)  is  a
sequence  of  numbers  in  which  each  term  after  the
first  is  obtained  by  adding  a  fixed  number  to  the
preceding  term .  This  fixed  number  is  called  the
common  difference .

Example  5.3  :  For  the  AP  :  3,  7,  11,  15,  ...,  find
the  common  difference .
Solution  :  d  =  7  -  3  =  4

Theorem  5.1  :  The  nth  term  of  an  AP  with  first  term
a  and  common  difference  d  is  given  by  an  =  a+(n-1)d

Proof  :  We  know  that  a1  =  a
a2  =  a  +  d
a3  =  a  +  2d
Therefore  an  =  a  +  (n-1)d        Q.E.D.

Exercise  5.2
1.  Find  the  10th  term  of  the  AP  :  2,  7,  12,  ...
"""
    
    # Step 1: Clean the text
    cleaner = TextCleaner(preserve_equations=True)
    cleaned_text, cleaning_stats = cleaner.clean(ocr_output)
    
    print("\nStep 1: Text Cleaning")
    print("-" * 70)
    print("Cleaning statistics:", cleaning_stats.to_dict())
    
    # Step 2: Identify structures
    structure_recovery = StructureRecovery()
    blocks = structure_recovery.identify_structures(cleaned_text)
    
    print("\nStep 2: Structure Identification")
    print("-" * 70)
    print(f"Identified {len(blocks)} educational structures")
    
    # Step 3: Annotate text with structures
    annotated_text = structure_recovery.annotate_text_with_structures(
        cleaned_text, blocks
    )
    
    print("\nStep 3: Annotated Text (Ready for Chunking)")
    print("-" * 70)
    print(annotated_text)
    
    # Step 4: Save results
    output = {
        'original_length': len(ocr_output),
        'cleaned_length': len(cleaned_text),
        'cleaning_stats': cleaning_stats.to_dict(),
        'structures_found': len(blocks),
        'structure_types': structure_recovery.get_structure_summary(blocks),
        'cleaned_text': cleaned_text,
        'annotated_text': annotated_text,
        'blocks': [block.to_dict() for block in blocks]
    }
    
    return output


def example_batch_processing():
    """Example: Batch process multiple pages."""
    print("\n" + "=" * 70)
    print("Example 4: Batch Processing Multiple Pages")
    print("=" * 70)
    
    # Simulate multiple pages
    pages = [
        {
            'page_number': 95,
            'text': 'Definition 1: An AP is a sequence with constant difference.',
            'metadata': {'content_type': 'text', 'chapter': 5}
        },
        {
            'page_number': 96,
            'text': 'Example 5.1: Find the 10th term of 2, 5, 8, 11, ...',
            'metadata': {'content_type': 'text', 'chapter': 5}
        },
        {
            'page_number': 97,
            'text': 'Exercise 5.1\n1. Find sum of first 20 terms\n2. Calculate nth term',
            'metadata': {'content_type': 'exercise', 'chapter': 5}
        }
    ]
    
    cleaner = TextCleaner()
    structure_recovery = StructureRecovery()
    
    results = []
    total_structures = 0
    
    for page in pages:
        # Clean
        cleaned_text, stats = cleaner.clean_page(
            page['text'],
            page['metadata']
        )
        
        # Identify structures
        blocks = structure_recovery.identify_structures(cleaned_text)
        total_structures += len(blocks)
        
        results.append({
            'page_number': page['page_number'],
            'cleaned_text': cleaned_text,
            'structures': [block.to_dict() for block in blocks],
            'stats': stats.to_dict()
        })
        
        print(f"\nPage {page['page_number']}:")
        print(f"  Structures found: {len(blocks)}")
        print(f"  Size reduction: {stats.to_dict()['reduction_percent']}%")
    
    print(f"\n\nBatch Summary:")
    print(f"  Total pages processed: {len(pages)}")
    print(f"  Total structures found: {total_structures}")
    
    return results


def example_quality_control():
    """Example: Quality control and validation."""
    print("\n" + "=" * 70)
    print("Example 5: Quality Control and Validation")
    print("=" * 70)
    
    # Sample text with quality issues
    text = """
Definition: A sequence is AP if difference is constant.

Example: Check if 2, 5, 8 is AP.
Sol: 5-2=3, 8-5=3. Yes, it's AP.

Xx  # OCR artifact
a  # Single character (noise)

Theorem 1: nth term is a+(n-1)d
"""
    
    cleaner = TextCleaner(aggressive_deduplication=True)
    cleaned_text, stats = cleaner.clean(text)
    
    structure_recovery = StructureRecovery()
    blocks = structure_recovery.identify_structures(cleaned_text)
    
    # Quality checks
    print("\nQuality Checks:")
    print("-" * 70)
    
    # Check 1: Adequate content remaining
    reduction = stats.to_dict()['reduction_percent']
    if reduction > 50:
        print("⚠ WARNING: Over 50% content removed. Review needed.")
    else:
        print(f"✓ Content reduction acceptable: {reduction}%")
    
    # Check 2: Structures identified
    if len(blocks) == 0:
        print("⚠ WARNING: No structures identified. Check input quality.")
    else:
        print(f"✓ Structures identified: {len(blocks)}")
    
    # Check 3: Average block confidence
    avg_confidence = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0
    if avg_confidence < 0.7:
        print(f"⚠ WARNING: Low average structure confidence: {avg_confidence:.2f}")
    else:
        print(f"✓ Structure confidence good: {avg_confidence:.2f}")
    
    # Check 4: Minimum content length
    if len(cleaned_text) < 50:
        print("⚠ WARNING: Very short cleaned text. Possible over-cleaning.")
    else:
        print(f"✓ Content length adequate: {len(cleaned_text)} characters")
    
    print("\nCleaned text:")
    print(cleaned_text)


def example_save_cleaned_document():
    """Example: Save cleaned document with metadata."""
    print("\n" + "=" * 70)
    print("Example 6: Save Cleaned Document")
    print("=" * 70)
    
    # Complete workflow
    text = """
Chapter 5: Arithmetic Progressions

Definition 1: An AP is a sequence with constant difference.
Example 5.1: Check if 2, 4, 6, 8 is an AP.
Exercise 5.1
1. Find the 20th term of 3, 7, 11, ...
"""
    
    # Process
    cleaner = TextCleaner()
    cleaned_text, stats = cleaner.clean(text)
    
    structure_recovery = StructureRecovery()
    blocks = structure_recovery.identify_structures(cleaned_text)
    
    # Create output document
    output_doc = {
        'metadata': {
            'class': 10,
            'subject': 'mathematics',
            'chapter': 5,
            'chapter_title': 'Arithmetic Progressions',
            'language': 'eng'
        },
        'cleaning_info': {
            'timestamp': '2025-12-28T10:00:00Z',
            'statistics': stats.to_dict(),
            'structures_identified': len(blocks)
        },
        'content': {
            'cleaned_text': cleaned_text,
            'structures': [block.to_dict() for block in blocks]
        }
    }
    
    # Save to file
    output_path = Path('output/cleaned/class10_math_ch5_cleaned.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_doc, f, indent=2, ensure_ascii=False)
    
    print(f"Saved cleaned document to: {output_path}")
    print("\nDocument summary:")
    print(f"  Original length: {stats.original_length}")
    print(f"  Cleaned length: {stats.cleaned_length}")
    print(f"  Structures: {len(blocks)}")
    print(f"  Quality: {'High' if len(blocks) > 0 else 'Low'}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NCERT TEXT CLEANING AND STRUCTURE RECOVERY - USAGE EXAMPLES")
    print("=" * 70 + "\n")
    
    try:
        # Run examples
        example_basic_cleaning()
        example_structure_recovery()
        output = example_complete_pipeline()
        batch_results = example_batch_processing()
        example_quality_control()
        example_save_cleaned_document()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
