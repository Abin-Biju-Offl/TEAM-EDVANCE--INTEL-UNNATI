"""
Text Cleaning Service - Phase 2

Cleans OCR text and recovers structure (definitions, examples, etc.)
Integrates existing cleaning pipeline code.
"""

import re
from typing import Dict, List
from loguru import logger


class TextCleaningService:
    """Service for cleaning OCR text and recovering structure"""
    
    def __init__(self):
        """Initialize cleaning patterns"""
        # Common OCR artifacts
        self.noise_patterns = [
            r'\[\s*\]',  # Empty brackets
            r'\(\s*\)',  # Empty parentheses
            r'_{3,}',    # Multiple underscores
            r'\s{3,}',   # Multiple spaces
            r'\n{4,}',   # Excessive newlines
        ]
        
        # Structure markers (use with re.IGNORECASE flag)
        self.structure_markers = {
            'definition': [r'definition:', r'def:', r'defined as'],
            'example': [r'example:', r'for example:', r'e\.g\.'],
            'theorem': [r'theorem:', r'theorem\s+\d+'],
            'exercise': [r'exercise:', r'questions:'],
            'note': [r'note:', r'remember:']
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean OCR artifacts from text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        # Fix common OCR errors
        cleaned = cleaned.replace('|', 'I')  # Pipe to I
        cleaned = cleaned.replace('0', 'O')  # Zero to O (context-dependent)
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def identify_structures(self, text: str) -> List[Dict]:
        """
        Identify structural elements (definitions, examples, etc.)
        
        Args:
            text: Cleaned text
            
        Returns:
            List of identified structures with their positions
        """
        structures = []
        
        for structure_type, patterns in self.structure_markers.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    structures.append({
                        'type': structure_type,
                        'start': match.start(),
                        'end': match.end(),
                        'marker': match.group()
                    })
        
        # Sort by position
        structures.sort(key=lambda x: x['start'])
        
        return structures
    
    def extract_metadata(self, text: str, filename: str = "") -> Dict:
        """
        Extract metadata (chapter, section, page) from text
        
        Args:
            text: Cleaned text
            filename: Original filename
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'chapter': None,
            'section': None,
            'class': '10',  # Default to Class 10
            'subject': 'Science',  # Default to Science
            'filename': filename
        }
        
        # Extract chapter number from filename
        # Format: jesc101.pdf -> Chapter 1
        chapter_match = re.search(r'jesc(\d+)', filename)
        if chapter_match:
            chapter_num = chapter_match.group(1).lstrip('0') or '0'
            if chapter_num != '1an' and chapter_num != '1ps':  # Skip answers/prelims
                metadata['chapter'] = chapter_num
        
        # Extract chapter from text
        chapter_patterns = [
            r'chapter\s+(\d+)',
            r'chapter\s+([ivxlcdm]+)',  # Roman numerals
        ]
        
        for pattern in chapter_patterns:
            match = re.search(pattern, text[:500], re.IGNORECASE)  # Check first 500 chars
            if match:
                metadata['chapter'] = match.group(1)
                break
        
        # Extract section
        section_match = re.search(r'section\s+(\d+\.?\d*)', text[:1000], re.IGNORECASE)
        if section_match:
            metadata['section'] = section_match.group(1)
        
        return metadata
    
    def process_document(self, ocr_result: Dict) -> Dict:
        """
        Process entire OCR document
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            Processed document with cleaned text and metadata
        """
        logger.info(f"Cleaning document: {ocr_result.get('filename', 'unknown')}")
        
        # Clean full text
        cleaned_text = self.clean_text(ocr_result.get('full_text', ''))
        
        # Identify structures
        structures = self.identify_structures(cleaned_text)
        
        # Extract metadata
        metadata = self.extract_metadata(
            cleaned_text,
            ocr_result.get('filename', '')
        )
        
        # Preserve OCR-level metadata (chapter, page numbers)
        if 'chapter' in ocr_result:
            metadata['chapter'] = ocr_result['chapter']
        if 'pages' in ocr_result:
            metadata['pages'] = ocr_result['pages']
        
        result = {
            'original_text': ocr_result.get('full_text', ''),
            'cleaned_text': cleaned_text,
            'structures': structures,
            'metadata': metadata,
            'char_count_original': len(ocr_result.get('full_text', '')),
            'char_count_cleaned': len(cleaned_text),
            'total_pages': ocr_result.get('total_pages', 0)
        }
        
        logger.success(
            f"Cleaned {ocr_result.get('filename')}: "
            f"{result['char_count_original']} -> {result['char_count_cleaned']} chars"
        )
        
        return result


# Global cleaning service instance
cleaning_service = TextCleaningService()
