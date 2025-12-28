"""
Metadata Extraction for NCERT Textbooks
========================================

Extracts and manages metadata for NCERT textbook pages.
Ensures traceability and provenance for all extracted content.
"""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from .ocr_engine import PageMetadata, Language

logger = logging.getLogger(__name__)


class NCERTMetadataExtractor:
    """
    Extracts metadata from NCERT textbook filenames and content.
    
    Expected filename format:
        NCERT_Class{X}_{Subject}_{Language}.pdf
        e.g., NCERT_Class10_Mathematics_English.pdf
        e.g., NCERT_Class06_Science_Hindi.pdf
    """
    
    # Standard NCERT subjects
    VALID_SUBJECTS = {
        'mathematics', 'science', 'social_science', 'english',
        'hindi', 'sanskrit', 'history', 'geography', 'civics',
        'economics', 'physics', 'chemistry', 'biology',
        'accountancy', 'business_studies', 'computer_science'
    }
    
    # Language mapping
    LANGUAGE_MAP = {
        'english': Language.ENGLISH.value,
        'hindi': Language.HINDI.value,
        'sanskrit': Language.SANSKRIT.value,
        'urdu': Language.URDU.value,
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize metadata extractor.
        
        Args:
            config_path: Optional path to configuration file with subject mappings
        """
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def extract_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract metadata from NCERT filename.
        
        Args:
            filename: Name of the NCERT file
        
        Returns:
            Dictionary with class, subject, and language
        
        Raises:
            ValueError: If filename doesn't match expected format
        """
        # Pattern: NCERT_Class{number}_{Subject}_{Language}.pdf
        pattern = r'NCERT_Class(\d+)_([A-Za-z_]+)_([A-Za-z]+)\.pdf'
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if not match:
            raise ValueError(
                f"Filename '{filename}' doesn't match expected format: "
                f"NCERT_Class{{X}}_{{Subject}}_{{Language}}.pdf"
            )
        
        class_num, subject, language = match.groups()
        
        # Validate class number
        class_number = int(class_num)
        if class_number < 1 or class_number > 12:
            raise ValueError(f"Invalid class number: {class_number}. Must be 1-12.")
        
        # Normalize and validate subject
        subject_normalized = subject.lower()
        if subject_normalized not in self.VALID_SUBJECTS:
            logger.warning(
                f"Subject '{subject}' not in standard NCERT subjects. "
                f"Proceeding anyway."
            )
        
        # Normalize and validate language
        language_normalized = language.lower()
        if language_normalized not in self.LANGUAGE_MAP:
            logger.warning(
                f"Language '{language}' not recognized. "
                f"Defaulting to English."
            )
            language_code = Language.ENGLISH.value
        else:
            language_code = self.LANGUAGE_MAP[language_normalized]
        
        return {
            'class_number': class_number,
            'subject': subject_normalized,
            'language': language_code,
            'language_name': language_normalized
        }
    
    def extract_chapter_info(
        self, 
        text: str, 
        page_number: int
    ) -> Dict[str, Any]:
        """
        Extract chapter information from page text.
        
        Args:
            text: OCR text from page
            page_number: Physical page number
        
        Returns:
            Dictionary with chapter_number and chapter_title
        """
        chapter_info = {
            'chapter_number': 0,
            'chapter_title': 'Unknown'
        }
        
        # Look for chapter headings (common patterns)
        # Pattern 1: "Chapter 5: Title"
        pattern1 = r'Chapter\s+(\d+)\s*[:\-]?\s*(.+)'
        # Pattern 2: "5. Title" (at start of line)
        pattern2 = r'^(\d+)\.\s+(.+)'
        # Pattern 3: Hindi - "अध्याय 5"
        pattern3 = r'अध्याय\s+(\d+)\s*[:\-]?\s*(.+)'
        
        lines = text.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            
            # Try English patterns
            match = re.search(pattern1, line, re.IGNORECASE)
            if not match:
                match = re.match(pattern2, line)
            
            # Try Hindi pattern
            if not match:
                match = re.search(pattern3, line)
            
            if match:
                chapter_info['chapter_number'] = int(match.group(1))
                chapter_info['chapter_title'] = match.group(2).strip()
                logger.info(
                    f"Detected Chapter {chapter_info['chapter_number']}: "
                    f"{chapter_info['chapter_title']}"
                )
                break
        
        return chapter_info
    
    def create_page_metadata(
        self,
        source_file: Path,
        page_number: int,
        chapter_number: int = 0,
        chapter_title: str = 'Unknown',
        content_type: str = 'text',
        ocr_confidence: float = 0.0
    ) -> PageMetadata:
        """
        Create complete PageMetadata object.
        
        Args:
            source_file: Path to source PDF/image file
            page_number: Page number in document
            chapter_number: Chapter number (if known)
            chapter_title: Chapter title (if known)
            content_type: Type of content on page
            ocr_confidence: OCR confidence score
        
        Returns:
            PageMetadata object
        """
        try:
            # Extract file-level metadata
            file_metadata = self.extract_from_filename(source_file.name)
        except ValueError as e:
            logger.error(f"Failed to extract metadata from filename: {e}")
            # Provide defaults
            file_metadata = {
                'class_number': 0,
                'subject': 'unknown',
                'language': Language.ENGLISH.value
            }
        
        metadata = PageMetadata(
            class_number=file_metadata['class_number'],
            subject=file_metadata['subject'],
            chapter_number=chapter_number,
            chapter_title=chapter_title,
            page_number=page_number,
            language=file_metadata['language'],
            content_type=content_type,
            source_file=str(source_file),
            ocr_confidence=ocr_confidence,
            processing_timestamp=datetime.utcnow().isoformat()
        )
        
        return metadata
    
    def infer_content_type(self, text: str) -> str:
        """
        Infer content type from extracted text.
        
        Args:
            text: Extracted text from page
        
        Returns:
            Content type (text, exercise, equation, table, mixed)
        """
        text_lower = text.lower()
        
        # Check for exercises
        exercise_patterns = [
            r'exercise\s+\d+',
            r'questions\s+for\s+practice',
            r'solve\s+the\s+following',
            r'answer\s+the\s+following',
            r'प्रश्नावली'  # Hindi for exercise
        ]
        
        for pattern in exercise_patterns:
            if re.search(pattern, text_lower):
                return 'exercise'
        
        # Check for equations (look for mathematical symbols)
        equation_indicators = ['=', '∑', '∫', '√', '÷', '×', '≠', '≤', '≥']
        equation_count = sum(text.count(ind) for ind in equation_indicators)
        
        if equation_count > 10:
            return 'equation'
        
        # Check for tables (look for structured data patterns)
        lines = text.split('\n')
        table_indicators = sum(
            1 for line in lines 
            if line.count('|') > 2 or line.count('\t') > 2
        )
        
        if table_indicators > 3:
            return 'table'
        
        # Check for mixed content
        if equation_count > 5 and len(lines) > 20:
            return 'mixed'
        
        # Default to text
        return 'text'
