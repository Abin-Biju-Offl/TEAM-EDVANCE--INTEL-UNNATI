"""
Example usage of NCERT Ingestion Pipeline
==========================================

Demonstrates how to use the OCR and ingestion system for NCERT textbooks.
"""

import logging
from pathlib import Path

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.ocr_engine import Language
from src.ingestion.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_process_single_pdf():
    """Example: Process a single NCERT PDF file."""
    print("=" * 60)
    print("Example 1: Processing Single PDF")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        output_dir=Path("output/extracted"),
        tesseract_path=None,  # Auto-detect
        default_language=Language.COMBINED_ENG_HIN,
        dpi=300,
        enable_preprocessing=True,
        remove_margins=True,
        confidence_threshold=50.0
    )
    
    # Path to NCERT PDF (example filename)
    pdf_path = Path("data/raw/NCERT_Class10_Mathematics_English.pdf")
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Please place NCERT PDFs in data/raw/ directory")
        return
    
    # Process PDF
    print(f"Processing: {pdf_path.name}")
    extracted_pages = pipeline.process_pdf(
        pdf_path=pdf_path,
        start_page=0,  # Start from first page
        end_page=10    # Process first 10 pages (for testing)
    )
    
    # Save results
    output_file = pipeline.save_results(
        extracted_pages=extracted_pages,
        output_filename=f"{pdf_path.stem}_extracted.json"
    )
    
    # Display statistics
    stats = pipeline.get_statistics()
    print(f"\nProcessing Statistics:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Successful: {stats['successful_pages']}")
    print(f"  Failed: {stats['failed_pages']}")
    print(f"  Low confidence: {stats['low_confidence_pages']}")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"\nOutput saved to: {output_file}")


def example_process_single_image():
    """Example: Process a single scanned image."""
    print("=" * 60)
    print("Example 2: Processing Single Image")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        output_dir=Path("output/extracted"),
        default_language=Language.HINDI
    )
    
    # Path to scanned image
    image_path = Path("data/raw/page_scan.png")
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please place scanned images in data/raw/ directory")
        return
    
    # Process image with metadata
    extracted_page = pipeline.process_image(
        image_path=image_path,
        class_number=10,
        subject="mathematics",
        language="hin",
        page_number=45,
        chapter_number=5,
        chapter_title="Arithmetic Progressions"
    )
    
    # Save result
    output_file = pipeline.save_results(
        extracted_pages=[extracted_page],
        output_filename=f"{image_path.stem}_extracted.json"
    )
    
    print(f"Extracted text preview:")
    print("-" * 60)
    print(extracted_page.text[:500])  # First 500 characters
    print("-" * 60)
    print(f"\nOCR Confidence: {extracted_page.metadata.ocr_confidence:.2f}%")
    print(f"Content Type: {extracted_page.metadata.content_type}")
    print(f"Output saved to: {output_file}")


def example_batch_processing():
    """Example: Batch process multiple PDFs."""
    print("=" * 60)
    print("Example 3: Batch Processing Multiple PDFs")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        output_dir=Path("output/extracted"),
        default_language=Language.COMBINED_ENG_HIN,
        dpi=300
    )
    
    # Input directory containing NCERT PDFs
    input_dir = Path("data/raw")
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please create data/raw/ and place NCERT PDFs there")
        return
    
    # Batch process all PDFs
    results = pipeline.batch_process(
        input_dir=input_dir,
        file_pattern="NCERT_*.pdf",
        max_workers=1  # Sequential processing
    )
    
    print(f"\nProcessed {len(results)} files:")
    for input_file, output_file in results.items():
        print(f"  {Path(input_file).name} -> {output_file.name}")
    
    # Overall statistics
    stats = pipeline.get_statistics()
    print(f"\nOverall Statistics:")
    print(f"  Total pages: {stats['total_pages']}")
    print(f"  Success rate: {stats['successful_pages']/stats['total_pages']*100:.1f}%")


def example_with_config_file():
    """Example: Use configuration file."""
    print("=" * 60)
    print("Example 4: Using Configuration File")
    print("=" * 60)
    
    # Create and save default configuration
    config = ConfigManager()
    config_file = Path("config/ingestion_config.json")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config.save_to_file(config_file)
    print(f"Created default config at: {config_file}")
    
    # Load configuration
    config = ConfigManager(config_file=config_file)
    config.validate()
    
    # Initialize pipeline with config
    pipeline = IngestionPipeline(
        output_dir=config.output.output_dir,
        default_language=Language(config.ocr.default_language),
        dpi=config.ocr.dpi,
        enable_preprocessing=config.preprocessing.enabled,
        remove_margins=config.preprocessing.remove_margins,
        confidence_threshold=config.ocr.confidence_threshold
    )
    
    print("Pipeline initialized with configuration:")
    print(f"  DPI: {config.ocr.dpi}")
    print(f"  Language: {config.ocr.default_language}")
    print(f"  Preprocessing: {config.preprocessing.enabled}")
    print(f"  Output dir: {config.output.output_dir}")


def example_error_handling():
    """Example: Handling OCR errors and low confidence pages."""
    print("=" * 60)
    print("Example 5: Error Handling and Quality Control")
    print("=" * 60)
    
    pipeline = IngestionPipeline(
        output_dir=Path("output/extracted"),
        confidence_threshold=70.0  # Higher threshold for quality
    )
    
    pdf_path = Path("data/raw/NCERT_Class10_Mathematics_English.pdf")
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    # Process pages
    extracted_pages = pipeline.process_pdf(pdf_path, end_page=5)
    
    # Analyze results
    print("\nQuality Analysis:")
    print("-" * 60)
    
    for page in extracted_pages:
        page_num = page.metadata.page_number
        confidence = page.metadata.ocr_confidence
        has_errors = page.ocr_errors is not None
        
        status = "✓" if confidence >= 70.0 and not has_errors else "✗"
        
        print(f"Page {page_num}: {status} Confidence: {confidence:.1f}%")
        
        if has_errors:
            print(f"  Errors: {', '.join(page.ocr_errors)}")
        
        if confidence < 50.0:
            print(f"  ⚠ WARNING: Very low confidence, manual review needed")
        elif confidence < 70.0:
            print(f"  ⚠ CAUTION: Below threshold, consider reprocessing")
    
    print("-" * 60)
    print("\nRecommendations:")
    print("- Pages with confidence < 50%: Manual review required")
    print("- Pages with confidence 50-70%: Consider higher DPI or aggressive preprocessing")
    print("- Pages with errors: Check source quality and Tesseract installation")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NCERT Textbook Ingestion Pipeline - Usage Examples")
    print("=" * 60 + "\n")
    
    examples = [
        ("Single PDF Processing", example_process_single_pdf),
        ("Single Image Processing", example_process_single_image),
        ("Batch Processing", example_batch_processing),
        ("Configuration File", example_with_config_file),
        ("Error Handling", example_error_handling),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nTo run a specific example, modify the code to call it directly.")
    print("For production use, adapt these examples to your workflow.\n")
    
    # Run configuration example by default (safe to run)
    example_with_config_file()


if __name__ == "__main__":
    main()
