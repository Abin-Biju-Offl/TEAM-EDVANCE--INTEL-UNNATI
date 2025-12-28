"""
Text Cleaning Module for NCERT Content
=======================================

Repairs OCR errors, normalizes text, and removes noise while preserving
educational structure and content integrity.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Statistics from text cleaning operation."""
    original_length: int
    cleaned_length: int
    lines_removed: int
    sentences_repaired: int
    duplicates_removed: int
    noise_patterns_cleaned: int
    
    def to_dict(self) -> Dict:
        return {
            'original_length': self.original_length,
            'cleaned_length': self.cleaned_length,
            'reduction_percent': round((1 - self.cleaned_length/self.original_length)*100, 2) if self.original_length > 0 else 0,
            'lines_removed': self.lines_removed,
            'sentences_repaired': self.sentences_repaired,
            'duplicates_removed': self.duplicates_removed,
            'noise_patterns_cleaned': self.noise_patterns_cleaned
        }


class TextCleaner:
    """
    Production-grade text cleaner for NCERT OCR output.
    
    Handles common OCR errors while preserving educational content structure.
    """
    
    # Common OCR character confusions
    OCR_REPLACEMENTS = {
        # Number/letter confusions
        'O': '0',  # Context-dependent
        'l': '1',  # Context-dependent
        'I': '1',  # Context-dependent
        # Common OCR artifacts
        '|': 'I',  # Vertical bar to I
        '—': '-',  # Em dash normalization
        '–': '-',  # En dash normalization
        ''': "'",  # Smart quote normalization
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',
        # Spacing issues
        '\xa0': ' ',  # Non-breaking space
        '\u200b': '',  # Zero-width space
        '\ufeff': '',  # Byte order mark
    }
    
    # Patterns for noise removal
    NOISE_PATTERNS = [
        # Page headers/footers that escaped cropping
        r'^Page\s+\d+\s*$',
        r'^\d+\s*$',  # Standalone numbers (likely page numbers)
        r'^NCERT\s*$',
        r'^Chapter\s+\d+\s*$',  # Duplicate chapter markers
        # OCR artifacts
        r'^[_\-=]{3,}$',  # Lines of underscores/dashes
        r'^\.{3,}$',  # Lines of dots
        r'^\s*[|]+\s*$',  # Lines of vertical bars
        # Empty or meaningless
        r'^\s*$',  # Empty lines (handled separately)
    ]
    
    # Patterns for duplicate detection
    DUPLICATE_PATTERNS = [
        # Chapter titles often repeated
        r'(Chapter\s+\d+\s*:?\s*[\w\s]+)',
        # Section headers
        r'(\d+\.\d+\s+[\w\s]{3,})',
    ]
    
    def __init__(
        self,
        preserve_equations: bool = True,
        preserve_special_formatting: bool = True,
        aggressive_deduplication: bool = False
    ):
        """
        Initialize text cleaner.
        
        Args:
            preserve_equations: Keep mathematical equations intact
            preserve_special_formatting: Preserve bold, italic markers
            aggressive_deduplication: More aggressive duplicate removal
        """
        self.preserve_equations = preserve_equations
        self.preserve_special_formatting = preserve_special_formatting
        self.aggressive_deduplication = aggressive_deduplication
        
        self.stats = CleaningStats(0, 0, 0, 0, 0, 0)
    
    def clean(self, text: str) -> Tuple[str, CleaningStats]:
        """
        Clean OCR text with comprehensive error correction.
        
        Args:
            text: Raw OCR text
        
        Returns:
            Tuple of (cleaned_text, cleaning_statistics)
        """
        if not text or not text.strip():
            return "", CleaningStats(0, 0, 0, 0, 0, 0)
        
        # Initialize statistics
        self.stats = CleaningStats(
            original_length=len(text),
            cleaned_length=0,
            lines_removed=0,
            sentences_repaired=0,
            duplicates_removed=0,
            noise_patterns_cleaned=0
        )
        
        # Cleaning pipeline
        text = self._normalize_unicode(text)
        text = self._remove_bom_and_special_chars(text)
        text = self._fix_common_ocr_errors(text)
        text = self._normalize_whitespace(text)
        text = self._remove_noise_patterns(text)
        text = self._fix_line_breaks(text)
        text = self._repair_broken_sentences(text)
        text = self._remove_duplicates(text)
        text = self._fix_punctuation_spacing(text)
        text = self._normalize_bullet_points(text)
        text = self._clean_empty_lines(text)
        
        self.stats.cleaned_length = len(text)
        
        logger.info(f"Cleaning complete: {self.stats.to_dict()}")
        return text, self.stats
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to consistent form."""
        # Normalize to NFC (Canonical Decomposition followed by Canonical Composition)
        # This ensures consistent representation of accented characters
        return unicodedata.normalize('NFC', text)
    
    def _remove_bom_and_special_chars(self, text: str) -> str:
        """Remove byte order marks and invisible characters."""
        # Remove BOM
        text = text.replace('\ufeff', '')
        # Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        # Replace non-breaking space with regular space
        text = text.replace('\xa0', ' ')
        return text
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR character recognition errors."""
        
        # Fix spaced-out words (common OCR error)
        # "M a t h e m a t i c s" -> "Mathematics"
        text = re.sub(r'\b([A-Z])\s+([a-z])\s+([a-z])', r'\1\2\3', text)
        
        # Fix broken words across lines (hyphenation)
        # "mathe-\nmatics" -> "mathematics"
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix incorrect spacing before punctuation
        # "word ." -> "word."
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Fix missing space after punctuation
        # "word.Next" -> "word. Next"
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common character confusions in context
        # "1ike" -> "like", "0ne" -> "one"
        text = re.sub(r'\b1([a-z]{2,})', r'l\1', text)  # 1 -> l at word start
        text = re.sub(r'\b0ne\b', 'one', text, flags=re.IGNORECASE)
        text = re.sub(r'\bI(\d)', r'1\1', text)  # I followed by digit
        
        self.stats.noise_patterns_cleaned += 1
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace characters."""
        # Replace tabs with spaces
        text = text.replace('\t', '    ')
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove trailing whitespace from each line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text
    
    def _remove_noise_patterns(self, text: str) -> str:
        """Remove common noise patterns from OCR output."""
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check against noise patterns
            is_noise = False
            for pattern in self.NOISE_PATTERNS:
                if re.match(pattern, line_stripped):
                    is_noise = True
                    removed_count += 1
                    break
            
            # Check for very short lines that are likely artifacts
            if len(line_stripped) > 0 and len(line_stripped) < 3 and not re.match(r'^\d+\.?$', line_stripped):
                # Single or two characters (unless it's a number)
                is_noise = True
                removed_count += 1
            
            if not is_noise:
                cleaned_lines.append(line)
        
        self.stats.lines_removed += removed_count
        return '\n'.join(cleaned_lines)
    
    def _fix_line_breaks(self, text: str) -> str:
        """Fix inappropriate line breaks in sentences."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            current_line = lines[i].strip()
            
            if not current_line:
                fixed_lines.append('')
                i += 1
                continue
            
            # Check if line ends with incomplete sentence
            # (no ending punctuation and next line starts with lowercase)
            if i < len(lines) - 1:
                next_line = lines[i + 1].strip()
                
                # Conditions for joining lines:
                # 1. Current line doesn't end with sentence-ending punctuation
                # 2. Current line doesn't end with colon (section header)
                # 3. Next line starts with lowercase or is continuation
                # 4. Not a list item
                should_join = (
                    current_line and
                    not re.search(r'[.!?:]\s*$', current_line) and
                    next_line and
                    (next_line[0].islower() or 
                     next_line.startswith('and') or 
                     next_line.startswith('or') or
                     next_line.startswith('but')) and
                    not re.match(r'^\d+\.|\([a-z]\)|\•|\-\s', current_line) and
                    not re.match(r'^\d+\.|\([a-z]\)|\•|\-\s', next_line)
                )
                
                if should_join:
                    # Join with space
                    fixed_lines.append(current_line + ' ' + next_line)
                    self.stats.sentences_repaired += 1
                    i += 2  # Skip next line as it's been merged
                    continue
            
            fixed_lines.append(current_line)
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _repair_broken_sentences(self, text: str) -> str:
        """Repair sentences broken by OCR errors."""
        
        # Fix sentences split by newline in the middle
        # Look for patterns like: "The equation is\nx + y = z"
        text = re.sub(
            r'([a-z,])\s*\n\s*([a-z])',
            r'\1 \2',
            text
        )
        
        # Fix broken equations (preserve mathematical content)
        if self.preserve_equations:
            # Rejoin lines that look like equation parts
            text = re.sub(
                r'([=+\-*/])\s*\n\s*([0-9a-z])',
                r'\1 \2',
                text,
                flags=re.IGNORECASE
            )
        
        self.stats.sentences_repaired += 1
        return text
    
    def _remove_duplicates(self, text: str) -> str:
        """Remove duplicate lines and repeated content."""
        lines = text.split('\n')
        seen = set()
        cleaned_lines = []
        duplicates_removed = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check for exact duplicates
            if line_stripped in seen:
                duplicates_removed += 1
                continue
            
            # Check for near-duplicates (chapter titles, headers)
            is_duplicate = False
            if self.aggressive_deduplication:
                for seen_line in seen:
                    # Calculate simple similarity
                    if self._are_similar(line_stripped, seen_line):
                        is_duplicate = True
                        duplicates_removed += 1
                        break
            
            if not is_duplicate:
                seen.add(line_stripped)
                cleaned_lines.append(line)
        
        self.stats.duplicates_removed = duplicates_removed
        return '\n'.join(cleaned_lines)
    
    def _are_similar(self, str1: str, str2: str, threshold: float = 0.9) -> bool:
        """Check if two strings are similar (simple implementation)."""
        if len(str1) < 10 or len(str2) < 10:
            return False
        
        # Simple character-based similarity
        longer = max(len(str1), len(str2))
        shorter = min(len(str1), len(str2))
        
        if shorter / longer < threshold:
            return False
        
        # Check common substring
        common = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return common / longer >= threshold
    
    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation marks."""
        
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Ensure space after punctuation (except in numbers)
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        
        # Fix spacing in common patterns
        text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)  # Numbers: 1, 234 -> 1,234
        text = re.sub(r'(\d)\s*\.\s*(\d)', r'\1.\2', text)  # Decimals: 3. 14 -> 3.14
        
        # Fix quotes
        text = re.sub(r'"\s+', '"', text)  # Remove space after opening quote
        text = re.sub(r'\s+"', '"', text)  # Remove space before closing quote
        
        return text
    
    def _normalize_bullet_points(self, text: str) -> str:
        """Normalize different bullet point styles."""
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Normalize various bullet styles to •
            line = re.sub(r'^\s*[*▪▫■□●○◆◇➢→]\s+', '• ', line)
            # Normalize numbered lists
            line = re.sub(r'^\s*(\d+)\)\s+', r'\1. ', line)
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _clean_empty_lines(self, text: str) -> str:
        """Remove excessive empty lines while preserving structure."""
        # Replace 3+ consecutive newlines with 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_page(self, page_text: str, page_metadata: Dict) -> Tuple[str, CleaningStats]:
        """
        Clean a single page with context from metadata.
        
        Args:
            page_text: Raw OCR text from page
            page_metadata: Metadata dict with chapter, page number, etc.
        
        Returns:
            Tuple of (cleaned_text, statistics)
        """
        # Use metadata to improve cleaning
        content_type = page_metadata.get('content_type', 'text')
        
        # Adjust cleaning based on content type
        if content_type == 'equation':
            # More conservative cleaning for equation pages
            self.preserve_equations = True
        elif content_type == 'exercise':
            # Preserve numbering structure for exercises
            pass
        
        return self.clean(page_text)
    
    def get_statistics(self) -> Dict:
        """Get cleaning statistics."""
        return self.stats.to_dict()
