"""
Citation Formatter
==================

Formats citations and source attributions for generated answers.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class CitationStyle(Enum):
    """Citation formatting styles."""
    INLINE = "inline"  # [Source 1]
    NUMBERED = "numbered"  # [1]
    FOOTNOTE = "footnote"  # ¹
    SIMPLE = "simple"  # (Chapter 5, Page 95)


@dataclass
class Citation:
    """A single citation with metadata."""
    source_id: int
    chunk_id: str
    textbook: str
    class_number: int
    subject: str
    chapter_number: int
    chapter_title: str
    page_numbers: List[int]
    chunk_type: str
    confidence: float
    
    def to_inline_format(self) -> str:
        """Format as inline citation: [Source 1]"""
        return f"[Source {self.source_id}]"
    
    def to_numbered_format(self) -> str:
        """Format as numbered citation: [1]"""
        return f"[{self.source_id}]"
    
    def to_footnote_format(self) -> str:
        """Format as footnote: ¹"""
        # Unicode superscript numbers
        superscripts = "⁰¹²³⁴⁵⁶⁷⁸⁹"
        if self.source_id < 10:
            return superscripts[self.source_id]
        else:
            # For numbers >= 10, use [1] style
            return f"[{self.source_id}]"
    
    def to_simple_format(self) -> str:
        """Format as simple citation: (Chapter 5, Page 95)"""
        page_str = f"Page {self.page_numbers[0]}" if self.page_numbers else "Page Unknown"
        return f"(Chapter {self.chapter_number}, {page_str})"
    
    def to_full_reference(self) -> str:
        """
        Format as full reference for source list.
        
        Example:
            Source 1: NCERT Mathematics, Class 10, Chapter 5 (Arithmetic Progressions), Page 95
        """
        page_str = f"Page {self.page_numbers[0]}" if self.page_numbers else "Page Unknown"
        
        return (
            f"Source {self.source_id}: "
            f"NCERT {self.subject.title()}, "
            f"Class {self.class_number}, "
            f"Chapter {self.chapter_number} ({self.chapter_title}), "
            f"{page_str}"
        )


class CitationFormatter:
    """Formats citations and manages source lists."""
    
    def __init__(self, style: CitationStyle = CitationStyle.INLINE):
        """
        Initialize citation formatter.
        
        Args:
            style: Citation style to use
        """
        self.style = style
        self.citations: List[Citation] = []
    
    def add_citation_from_chunk(self, chunk: Any, source_id: int) -> Citation:
        """
        Create citation from a retrieved chunk.
        
        Args:
            chunk: RetrievalResult object
            source_id: Sequential source number (1, 2, 3, ...)
            
        Returns:
            Citation object
        """
        meta = chunk.metadata
        
        citation = Citation(
            source_id=source_id,
            chunk_id=chunk.chunk_id,
            textbook=f"NCERT {meta.get('subject', 'Unknown').title()}",
            class_number=meta.get('class_number', 0),
            subject=meta.get('subject', 'Unknown'),
            chapter_number=meta.get('chapter_number', 0),
            chapter_title=meta.get('chapter_title', 'Unknown'),
            page_numbers=meta.get('page_numbers', []),
            chunk_type=meta.get('chunk_type', 'content'),
            confidence=chunk.confidence if hasattr(chunk, 'confidence') else 1.0
        )
        
        self.citations.append(citation)
        return citation
    
    def format_citation(self, source_id: int) -> str:
        """
        Format a citation reference in the chosen style.
        
        Args:
            source_id: Source number to format
            
        Returns:
            Formatted citation string
        """
        # Find citation by source_id
        citation = next((c for c in self.citations if c.source_id == source_id), None)
        
        if not citation:
            return f"[Source {source_id}]"  # Fallback
        
        if self.style == CitationStyle.INLINE:
            return citation.to_inline_format()
        elif self.style == CitationStyle.NUMBERED:
            return citation.to_numbered_format()
        elif self.style == CitationStyle.FOOTNOTE:
            return citation.to_footnote_format()
        elif self.style == CitationStyle.SIMPLE:
            return citation.to_simple_format()
        else:
            return citation.to_inline_format()
    
    def format_source_list(self, include_confidence: bool = False) -> str:
        """
        Format the complete source list.
        
        Args:
            include_confidence: Whether to include confidence scores
            
        Returns:
            Formatted source list as string
        """
        if not self.citations:
            return ""
        
        lines = ["**Sources:**"]
        
        for citation in self.citations:
            line = f"- {citation.to_full_reference()}"
            
            if include_confidence:
                line += f" (Confidence: {citation.confidence:.2f})"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_citation_by_id(self, source_id: int) -> Citation:
        """Get citation by source ID."""
        return next((c for c in self.citations if c.source_id == source_id), None)
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations."""
        return self.citations.copy()
    
    def clear(self):
        """Clear all citations."""
        self.citations.clear()


def extract_citations_from_answer(answer: str) -> List[int]:
    """
    Extract citation numbers from an answer.
    
    Args:
        answer: Generated answer text
        
    Returns:
        List of cited source IDs
    """
    import re
    
    # Match [Source 1], [Source 2, 3], [1], [2, 3], etc.
    patterns = [
        r'\[Source\s+(\d+)\]',  # [Source 1]
        r'\[(\d+)\]',            # [1]
    ]
    
    cited_ids = set()
    
    for pattern in patterns:
        matches = re.findall(pattern, answer)
        for match in matches:
            cited_ids.add(int(match))
    
    return sorted(list(cited_ids))


def verify_citations_exist(answer: str, available_sources: List[int]) -> bool:
    """
    Verify that all citations in the answer reference available sources.
    
    Args:
        answer: Generated answer text
        available_sources: List of available source IDs
        
    Returns:
        True if all citations are valid, False otherwise
    """
    cited_ids = extract_citations_from_answer(answer)
    
    for cited_id in cited_ids:
        if cited_id not in available_sources:
            return False
    
    return True


def format_answer_with_sources(answer: str, citations: List[Citation]) -> str:
    """
    Combine answer text with formatted source list.
    
    Args:
        answer: Generated answer text
        citations: List of Citation objects
        
    Returns:
        Complete answer with source attribution
    """
    # Format sources
    formatter = CitationFormatter()
    formatter.citations = citations
    source_list = formatter.format_source_list(include_confidence=False)
    
    # Combine
    if source_list:
        return f"{answer}\n\n{source_list}"
    else:
        return answer
