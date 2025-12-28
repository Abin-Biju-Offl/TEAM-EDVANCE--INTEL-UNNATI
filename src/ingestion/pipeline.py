"""
NCERT Textbook Ingestion Pipeline
==================================

Complete pipeline orchestrator for processing NCERT textbooks.
Handles batch processing, error recovery, and output generation.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .ocr_engine import OCREngine, Language, ExtractedPage
from .metadata_extractor import NCERTMetadataExtractor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Production-grade ingestion pipeline for NCERT textbooks.
    
    Processes PDFs and images with:
    - Page-by-page extraction
    - Metadata enrichment
    - Error handling and recovery
    - Progress tracking
    - JSON output generation
    """
    
    def __init__(
        self,
        output_dir: Path,
        tesseract_path: Optional[str] = None,
        default_language: Language = Language.COMBINED_ENG_HIN,
        dpi: int = 300,
        enable_preprocessing: bool = True,
        remove_margins: bool = True,
        confidence_threshold: float = 50.0
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            output_dir: Directory for output JSON files
            tesseract_path: Custom Tesseract executable path
            default_language: Default OCR language
            dpi: DPI for PDF conversion
            enable_preprocessing: Enable image preprocessing
            remove_margins: Remove headers/footers
            confidence_threshold: Minimum acceptable OCR confidence
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ocr_engine = OCREngine(
            tesseract_path=tesseract_path,
            default_language=default_language
        )
        self.metadata_extractor = NCERTMetadataExtractor()
        
        self.dpi = dpi
        self.enable_preprocessing = enable_preprocessing
        self.remove_margins = remove_margins
        self.confidence_threshold = confidence_threshold
        
        # Statistics
        self.stats = {
            'total_pages': 0,
            'successful_pages': 0,
            'failed_pages': 0,
            'low_confidence_pages': 0,
            'total_errors': 0
        }
    
    def process_pdf(
        self,
        pdf_path: Path,
        start_page: int = 0,
        end_page: Optional[int] = None
    ) -> List[ExtractedPage]:
        """
        Process a complete PDF file page by page.
        
        Args:
            pdf_path: Path to NCERT PDF file
            start_page: Starting page number (0-indexed)
            end_page: Ending page number (None = process all)
        
        Returns:
            List of ExtractedPage objects
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Get total pages
        try:
            import pdf2image
            num_pages = pdf2image.pdfinfo_from_path(pdf_path)['Pages']
            logger.info(f"Total pages in PDF: {num_pages}")
        except Exception as e:
            logger.warning(f"Could not determine page count: {e}")
            num_pages = 1000  # Fallback
        
        if end_page is None:
            end_page = num_pages
        
        extracted_pages = []
        chapter_context = {'number': 0, 'title': 'Unknown'}
        
        # Process pages sequentially (to maintain chapter context)
        for page_num in tqdm(range(start_page, end_page), desc="Processing pages"):
            try:
                self.stats['total_pages'] += 1
                
                # Create initial metadata
                metadata = self.metadata_extractor.create_page_metadata(
                    source_file=pdf_path,
                    page_number=page_num + 1,  # 1-indexed for display
                    chapter_number=chapter_context['number'],
                    chapter_title=chapter_context['title']
                )
                
                # Process page
                extracted_page = self.ocr_engine.process_pdf_page(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    metadata=metadata,
                    dpi=self.dpi,
                    preprocess=self.enable_preprocessing,
                    remove_margins=self.remove_margins
                )
                
                # Try to extract chapter info if this is a chapter start
                if extracted_page.text:
                    chapter_info = self.metadata_extractor.extract_chapter_info(
                        extracted_page.text,
                        page_num + 1
                    )
                    
                    # Update chapter context if new chapter detected
                    if chapter_info['chapter_number'] > 0:
                        chapter_context = {
                            'number': chapter_info['chapter_number'],
                            'title': chapter_info['chapter_title']
                        }
                        extracted_page.metadata.chapter_number = chapter_info['chapter_number']
                        extracted_page.metadata.chapter_title = chapter_info['chapter_title']
                    else:
                        # Use current chapter context
                        extracted_page.metadata.chapter_number = chapter_context['number']
                        extracted_page.metadata.chapter_title = chapter_context['title']
                    
                    # Infer content type
                    content_type = self.metadata_extractor.infer_content_type(
                        extracted_page.text
                    )
                    extracted_page.metadata.content_type = content_type
                
                # Check confidence
                if extracted_page.metadata.ocr_confidence < self.confidence_threshold:
                    self.stats['low_confidence_pages'] += 1
                    logger.warning(
                        f"Page {page_num + 1} has low confidence: "
                        f"{extracted_page.metadata.ocr_confidence:.2f}%"
                    )
                
                # Track errors
                if extracted_page.ocr_errors:
                    self.stats['total_errors'] += len(extracted_page.ocr_errors)
                    logger.error(
                        f"Page {page_num + 1} had {len(extracted_page.ocr_errors)} errors"
                    )
                
                if extracted_page.text.strip():
                    self.stats['successful_pages'] += 1
                else:
                    self.stats['failed_pages'] += 1
                
                extracted_pages.append(extracted_page)
                
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                self.stats['failed_pages'] += 1
                self.stats['total_errors'] += 1
                logger.error(f"Failed to process page {page_num + 1}: {e}")
                
                # Create error placeholder
                error_metadata = self.metadata_extractor.create_page_metadata(
                    source_file=pdf_path,
                    page_number=page_num + 1
                )
                error_page = ExtractedPage(
                    text="",
                    metadata=error_metadata,
                    ocr_errors=[f"Processing failed: {str(e)}"]
                )
                extracted_pages.append(error_page)
        
        logger.info(f"Completed processing {len(extracted_pages)} pages")
        return extracted_pages
    
    def process_image(
        self,
        image_path: Path,
        class_number: int,
        subject: str,
        language: str,
        page_number: int,
        chapter_number: int = 0,
        chapter_title: str = 'Unknown'
    ) -> ExtractedPage:
        """
        Process a single image file (scanned page).
        
        Args:
            image_path: Path to image file
            class_number: NCERT class (1-12)
            subject: Subject name
            language: Language code
            page_number: Page number
            chapter_number: Chapter number (if known)
            chapter_title: Chapter title (if known)
        
        Returns:
            ExtractedPage object
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Processing image: {image_path.name}")
        
        # Create metadata
        metadata = self.metadata_extractor.create_page_metadata(
            source_file=image_path,
            page_number=page_number,
            chapter_number=chapter_number,
            chapter_title=chapter_title
        )
        
        # Override with provided metadata
        metadata.class_number = class_number
        metadata.subject = subject
        metadata.language = language
        
        # Process image
        extracted_page = self.ocr_engine.process_image_file(
            image_path=image_path,
            metadata=metadata,
            preprocess=self.enable_preprocessing,
            remove_margins=self.remove_margins
        )
        
        # Infer content type
        if extracted_page.text:
            content_type = self.metadata_extractor.infer_content_type(
                extracted_page.text
            )
            extracted_page.metadata.content_type = content_type
        
        return extracted_page
    
    def save_results(
        self,
        extracted_pages: List[ExtractedPage],
        output_filename: str
    ) -> Path:
        """
        Save extracted pages to JSON file.
        
        Args:
            extracted_pages: List of ExtractedPage objects
            output_filename: Name for output JSON file
        
        Returns:
            Path to saved JSON file
        """
        output_path = self.output_dir / output_filename
        
        # Convert to serializable format
        data = {
            'document_metadata': {
                'total_pages': len(extracted_pages),
                'source': extracted_pages[0].metadata.source_file if extracted_pages else None,
                'class': extracted_pages[0].metadata.class_number if extracted_pages else None,
                'subject': extracted_pages[0].metadata.subject if extracted_pages else None,
                'language': extracted_pages[0].metadata.language if extracted_pages else None,
            },
            'pages': [page.to_dict() for page in extracted_pages],
            'processing_statistics': self.stats
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved results to {output_path}")
        logger.info(f"Statistics: {self.stats}")
        
        return output_path
    
    def batch_process(
        self,
        input_dir: Path,
        file_pattern: str = "*.pdf",
        max_workers: int = 1  # Sequential by default for memory management
    ) -> Dict[str, Path]:
        """
        Batch process multiple NCERT files.
        
        Args:
            input_dir: Directory containing NCERT files
            file_pattern: Glob pattern for files to process
            max_workers: Number of parallel workers (1 = sequential)
        
        Returns:
            Dictionary mapping input file to output JSON path
        """
        input_dir = Path(input_dir)
        files = list(input_dir.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files matching {file_pattern} found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(files)} files to process")
        
        results = {}
        
        if max_workers == 1:
            # Sequential processing
            for file_path in files:
                try:
                    extracted_pages = self.process_pdf(file_path)
                    output_filename = f"{file_path.stem}_extracted.json"
                    output_path = self.save_results(extracted_pages, output_filename)
                    results[str(file_path)] = output_path
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        else:
            # Parallel processing (use with caution - memory intensive)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_pdf, file_path): file_path
                    for file_path in files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        extracted_pages = future.result()
                        output_filename = f"{file_path.stem}_extracted.json"
                        output_path = self.save_results(extracted_pages, output_filename)
                        results[str(file_path)] = output_path
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
