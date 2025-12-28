"""
Educational Structure Recovery for NCERT Content
=================================================

Identifies and preserves educational structures in textbook content:
- Definitions
- Theorems and proofs
- Examples and solutions
- Exercises and questions
- Explanations and notes

Ensures these structures remain intact during cleaning and chunking.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of educational structures in NCERT textbooks."""
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"
    EXAMPLE = "example"
    SOLUTION = "solution"
    EXERCISE = "exercise"
    QUESTION = "question"
    NOTE = "note"
    REMARK = "remark"
    EXPLANATION = "explanation"
    SUMMARY = "summary"
    ACTIVITY = "activity"
    FORMULA = "formula"
    COROLLARY = "corollary"
    LEMMA = "lemma"


@dataclass
class EducationalBlock:
    """Represents an educational structure block."""
    structure_type: StructureType
    content: str
    start_line: int
    end_line: int
    title: Optional[str] = None
    number: Optional[str] = None  # e.g., "Example 1.5" -> "1.5"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'type': self.structure_type.value,
            'content': self.content,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'title': self.title,
            'number': self.number,
            'confidence': self.confidence
        }


class StructureRecovery:
    """
    Identifies and recovers educational structures in NCERT content.
    
    Preserves the pedagogical organization of textbook material.
    """
    
    # Patterns for identifying structure markers
    STRUCTURE_PATTERNS = {
        StructureType.DEFINITION: [
            r'^Definition\s*\d*\.?\d*\s*:?\s*',
            r'^def(?:inition)?\s*\d+\s*:',
            r'is defined as',
            r'we define',
        ],
        StructureType.THEOREM: [
            r'^Theorem\s*\d*\.?\d*\s*:?\s*',
            r'^THM\s*\d+',
        ],
        StructureType.PROOF: [
            r'^Proof\s*:?\s*',
            r'^Solution\s*:?\s*',
            r'Q\.E\.D\.',
        ],
        StructureType.EXAMPLE: [
            r'^Example\s*\d*\.?\d*\s*:?\s*',
            r'^Ex\.\s*\d+',
            r'^E\s*\d+\.',
        ],
        StructureType.SOLUTION: [
            r'^Solution\s*\d*\.?\d*\s*:?\s*',
            r'^Sol\.\s*:?\s*',
        ],
        StructureType.EXERCISE: [
            r'^Exercise\s*\d*\.?\d*\s*:?\s*',
            r'^EXERCISE\s+\d+',
        ],
        StructureType.QUESTION: [
            r'^\d+\.\s+',  # Numbered questions
            r'^\([a-z]\)\s+',  # Lettered sub-questions
            r'^Q\.\s*\d+',
        ],
        StructureType.NOTE: [
            r'^Note\s*:?\s*',
            r'^N\.B\.\s*:?',
            r'^\*\s*Note',
        ],
        StructureType.REMARK: [
            r'^Remark\s*:?\s*',
            r'^Observation\s*:?\s*',
        ],
        StructureType.SUMMARY: [
            r'^Summary\s*:?\s*',
            r'^Key\s+Points?\s*:?',
            r'^Important\s+Points?\s*:?',
        ],
        StructureType.ACTIVITY: [
            r'^Activity\s*\d*\.?\d*\s*:?\s*',
            r'^ACTIVITY\s+\d+',
        ],
        StructureType.FORMULA: [
            r'^Formula\s*\d*\.?\d*\s*:?\s*',
            r'^\[.*=.*\]$',  # Equations in brackets
        ],
        StructureType.COROLLARY: [
            r'^Corollary\s*\d*\.?\d*\s*:?\s*',
        ],
        StructureType.LEMMA: [
            r'^Lemma\s*\d*\.?\d*\s*:?\s*',
        ],
    }
    
    # Keywords that indicate explanation/elaboration
    EXPLANATION_KEYWORDS = [
        'therefore', 'hence', 'thus', 'so', 'because',
        'since', 'as', 'this means', 'in other words',
        'that is', 'i.e.', 'e.g.', 'for example',
        'for instance', 'such as', 'namely'
    ]
    
    def __init__(
        self,
        preserve_numbering: bool = True,
        detect_implicit_structures: bool = True
    ):
        """
        Initialize structure recovery.
        
        Args:
            preserve_numbering: Keep original numbering (Example 1.5)
            detect_implicit_structures: Find structures without explicit markers
        """
        self.preserve_numbering = preserve_numbering
        self.detect_implicit_structures = detect_implicit_structures
    
    def identify_structures(self, text: str) -> List[EducationalBlock]:
        """
        Identify all educational structures in text.
        
        Args:
            text: Cleaned text content
        
        Returns:
            List of EducationalBlock objects
        """
        blocks = []
        lines = text.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Check for structure markers
            detected_type, number, title = self._detect_structure_type(line)
            
            if detected_type:
                # Extract full block content
                content, end_line = self._extract_block_content(
                    lines, i, detected_type
                )
                
                block = EducationalBlock(
                    structure_type=detected_type,
                    content=content,
                    start_line=i,
                    end_line=end_line,
                    title=title,
                    number=number,
                    confidence=1.0
                )
                blocks.append(block)
                
                logger.debug(
                    f"Found {detected_type.value} at line {i}: {title or 'untitled'}"
                )
                
                i = end_line + 1
            else:
                i += 1
        
        # Detect implicit structures if enabled
        if self.detect_implicit_structures:
            implicit_blocks = self._detect_implicit_structures(text, lines, blocks)
            blocks.extend(implicit_blocks)
            blocks.sort(key=lambda b: b.start_line)
        
        return blocks
    
    def _detect_structure_type(
        self, 
        line: str
    ) -> Tuple[Optional[StructureType], Optional[str], Optional[str]]:
        """
        Detect structure type from line content.
        
        Returns:
            Tuple of (StructureType, number, title) or (None, None, None)
        """
        for struct_type, patterns in self.STRUCTURE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Extract number if present
                    number_match = re.search(r'(\d+\.?\d*)', line)
                    number = number_match.group(1) if number_match else None
                    
                    # Extract title (text after marker)
                    title = re.sub(pattern, '', line, flags=re.IGNORECASE).strip()
                    title = title.lstrip(':').strip()
                    
                    return struct_type, number, title if title else None
        
        return None, None, None
    
    def _extract_block_content(
        self,
        lines: List[str],
        start_line: int,
        structure_type: StructureType
    ) -> Tuple[str, int]:
        """
        Extract complete block content from start marker.
        
        Args:
            lines: All text lines
            start_line: Starting line index
            structure_type: Type of structure being extracted
        
        Returns:
            Tuple of (content, end_line_index)
        """
        content_lines = [lines[start_line]]
        current_line = start_line + 1
        
        # Determine block boundaries
        while current_line < len(lines):
            line = lines[current_line].strip()
            
            # Stop conditions
            if not line:
                # Empty line might end the block
                # But check next line to be sure
                if current_line + 1 < len(lines):
                    next_line = lines[current_line + 1].strip()
                    # If next line starts a new structure, stop
                    if self._is_structure_start(next_line):
                        break
                    # If next line continues, include empty line
                    if next_line and not self._is_structure_start(next_line):
                        content_lines.append(lines[current_line])
                        current_line += 1
                        continue
                break
            
            # Check if new structure starts
            if self._is_structure_start(line) and current_line != start_line:
                break
            
            # Check for proof end markers
            if structure_type == StructureType.PROOF:
                if re.search(r'Q\.E\.D\.|∎|■', line):
                    content_lines.append(lines[current_line])
                    current_line += 1
                    break
            
            content_lines.append(lines[current_line])
            current_line += 1
        
        content = '\n'.join(content_lines)
        end_line = current_line - 1
        
        return content, end_line
    
    def _is_structure_start(self, line: str) -> bool:
        """Check if line starts a new structure."""
        detected_type, _, _ = self._detect_structure_type(line)
        return detected_type is not None
    
    def _detect_implicit_structures(
        self,
        text: str,
        lines: List[str],
        explicit_blocks: List[EducationalBlock]
    ) -> List[EducationalBlock]:
        """
        Detect structures without explicit markers.
        
        For example, explanatory paragraphs, implicit examples, etc.
        """
        implicit_blocks = []
        
        # Get line ranges covered by explicit blocks
        covered_lines = set()
        for block in explicit_blocks:
            covered_lines.update(range(block.start_line, block.end_line + 1))
        
        # Analyze uncovered regions
        i = 0
        while i < len(lines):
            if i in covered_lines:
                i += 1
                continue
            
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Detect explanatory content
            if self._is_explanation(line):
                content, end_line = self._extract_explanation_block(lines, i)
                if content:
                    block = EducationalBlock(
                        structure_type=StructureType.EXPLANATION,
                        content=content,
                        start_line=i,
                        end_line=end_line,
                        confidence=0.7  # Lower confidence for implicit
                    )
                    implicit_blocks.append(block)
                    i = end_line + 1
                    continue
            
            i += 1
        
        return implicit_blocks
    
    def _is_explanation(self, line: str) -> bool:
        """Check if line contains explanatory content."""
        line_lower = line.lower()
        
        # Check for explanation keywords
        for keyword in self.EXPLANATION_KEYWORDS:
            if keyword in line_lower:
                return True
        
        # Check for typical explanation patterns
        if re.search(r'this\s+(means|shows|implies|indicates)', line_lower):
            return True
        if re.search(r'we\s+(can|see|observe|note|find)', line_lower):
            return True
        
        return False
    
    def _extract_explanation_block(
        self,
        lines: List[str],
        start_line: int
    ) -> Tuple[str, int]:
        """Extract explanatory paragraph."""
        content_lines = []
        current_line = start_line
        
        while current_line < len(lines):
            line = lines[current_line].strip()
            
            if not line:
                # End of paragraph
                break
            
            if self._is_structure_start(line):
                break
            
            content_lines.append(lines[current_line])
            current_line += 1
        
        return '\n'.join(content_lines), current_line - 1
    
    def annotate_text_with_structures(
        self,
        text: str,
        blocks: List[EducationalBlock]
    ) -> str:
        """
        Annotate text with structure markers for chunking.
        
        Args:
            text: Original text
            blocks: Identified structure blocks
        
        Returns:
            Text with structure annotations
        """
        lines = text.split('\n')
        annotated_lines = []
        
        # Create block lookup by line
        block_map = {}
        for block in blocks:
            block_map[block.start_line] = block
        
        for i, line in enumerate(lines):
            if i in block_map:
                block = block_map[i]
                # Add structure marker
                marker = f"[{block.structure_type.value.upper()}"
                if block.number:
                    marker += f" {block.number}"
                marker += "]"
                annotated_lines.append(marker)
            
            annotated_lines.append(line)
        
        return '\n'.join(annotated_lines)
    
    def get_structure_summary(self, blocks: List[EducationalBlock]) -> Dict:
        """
        Get summary statistics of identified structures.
        
        Args:
            blocks: List of educational blocks
        
        Returns:
            Dictionary with structure counts
        """
        summary = {
            'total_blocks': len(blocks),
            'by_type': {}
        }
        
        for block in blocks:
            struct_type = block.structure_type.value
            if struct_type not in summary['by_type']:
                summary['by_type'][struct_type] = 0
            summary['by_type'][struct_type] += 1
        
        return summary
