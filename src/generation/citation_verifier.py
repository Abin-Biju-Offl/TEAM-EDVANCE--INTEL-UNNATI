"""
Citation Verification Layer for RAG Responses
==============================================

Validates that every answer sentence is properly cited and maps to retrieved chunks.

Citation Format: (NCERT Class X, Subject, Chapter Y, Page Z)
Example: (NCERT Class 10, Mathematics, Chapter 5, Page 95)
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CitationReference:
    """A parsed citation reference."""
    class_number: int
    subject: str
    chapter_number: int
    page_number: int
    
    def to_string(self) -> str:
        """Format as citation string."""
        return (
            f"(NCERT Class {self.class_number}, "
            f"{self.subject.title()}, "
            f"Chapter {self.chapter_number}, "
            f"Page {self.page_number})"
        )
    
    def matches_chunk(self, chunk) -> bool:
        """Check if this citation matches a chunk's metadata."""
        meta = chunk.metadata
        
        return (
            meta.get('class_number') == self.class_number and
            meta.get('subject', '').lower() == self.subject.lower() and
            meta.get('chapter_number') == self.chapter_number and
            self.page_number in meta.get('page_numbers', [])
        )


@dataclass
class AnswerSentence:
    """A sentence from the answer with its citation."""
    text: str
    sentence_index: int
    citation: Optional[CitationReference]
    has_citation: bool
    
    def __repr__(self):
        cite_str = self.citation.to_string() if self.citation else "NO CITATION"
        return f"Sentence {self.sentence_index}: {self.text[:50]}... [{cite_str}]"


@dataclass
class ValidationResult:
    """Result of citation validation."""
    is_valid: bool
    validation_type: str  # "success", "missing_citations", "invalid_citations", "unmapped_sentences"
    error_message: str
    
    # Detailed validation info
    total_sentences: int
    cited_sentences: int
    uncited_sentences: int
    valid_citations: int
    invalid_citations: int
    
    # Which sentences failed
    failed_sentences: List[AnswerSentence]
    
    # Which citations don't map to chunks
    unmapped_citations: List[CitationReference]


class CitationVerifier:
    """
    Verifies that RAG answers are properly cited and grounded.
    
    Validation Rules:
    1. Every sentence must have a citation
    2. Every citation must reference an actual retrieved chunk
    3. Citation format must be: (NCERT Class X, Subject, Chapter Y, Page Z)
    4. No uncited claims allowed
    """
    
    def __init__(self, require_all_sentences_cited: bool = True):
        """
        Initialize citation verifier.
        
        Args:
            require_all_sentences_cited: If True, reject answers with any uncited sentences
        """
        self.require_all_sentences_cited = require_all_sentences_cited
        
        # Citation regex pattern
        # Matches: (NCERT Class 10, Mathematics, Chapter 5, Page 95)
        self.citation_pattern = re.compile(
            r'\(NCERT Class (\d+), '  # Class number
            r'([A-Za-z\s]+), '         # Subject (can be multi-word like "Social Science")
            r'Chapter (\d+), '         # Chapter number
            r'Page (\d+)\)'            # Page number
        )
    
    def verify(
        self,
        answer: str,
        retrieved_chunks: List[Any],
        strict_mode: bool = True
    ) -> ValidationResult:
        """
        Verify that answer is properly cited.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: List of RetrievalResult objects from retrieval
            strict_mode: If True, require every sentence to be cited
            
        Returns:
            ValidationResult with detailed validation info
        """
        # Special case: "I don't know" response is always valid
        if "I don't know based on NCERT textbooks" in answer:
            return ValidationResult(
                is_valid=True,
                validation_type="success",
                error_message="",
                total_sentences=1,
                cited_sentences=1,
                uncited_sentences=0,
                valid_citations=0,
                invalid_citations=0,
                failed_sentences=[],
                unmapped_citations=[]
            )
        
        # Parse answer into sentences
        sentences = self._parse_sentences(answer)
        
        # Validate each sentence
        failed_sentences = []
        unmapped_citations = []
        
        for sentence in sentences:
            if not sentence.has_citation:
                failed_sentences.append(sentence)
            elif sentence.citation:
                # Check if citation maps to a retrieved chunk
                if not self._citation_maps_to_chunk(sentence.citation, retrieved_chunks):
                    failed_sentences.append(sentence)
                    unmapped_citations.append(sentence.citation)
        
        # Calculate stats
        total_sentences = len(sentences)
        cited_sentences = sum(1 for s in sentences if s.has_citation)
        uncited_sentences = total_sentences - cited_sentences
        
        all_citations = [s.citation for s in sentences if s.citation]
        valid_citations = sum(
            1 for cite in all_citations 
            if self._citation_maps_to_chunk(cite, retrieved_chunks)
        )
        invalid_citations = len(all_citations) - valid_citations
        
        # Determine validation result
        if uncited_sentences > 0 and strict_mode:
            return ValidationResult(
                is_valid=False,
                validation_type="missing_citations",
                error_message=f"{uncited_sentences} sentence(s) lack citations",
                total_sentences=total_sentences,
                cited_sentences=cited_sentences,
                uncited_sentences=uncited_sentences,
                valid_citations=valid_citations,
                invalid_citations=invalid_citations,
                failed_sentences=failed_sentences,
                unmapped_citations=unmapped_citations
            )
        
        if invalid_citations > 0:
            return ValidationResult(
                is_valid=False,
                validation_type="invalid_citations",
                error_message=f"{invalid_citations} citation(s) don't map to retrieved chunks",
                total_sentences=total_sentences,
                cited_sentences=cited_sentences,
                uncited_sentences=uncited_sentences,
                valid_citations=valid_citations,
                invalid_citations=invalid_citations,
                failed_sentences=failed_sentences,
                unmapped_citations=unmapped_citations
            )
        
        # All checks passed
        return ValidationResult(
            is_valid=True,
            validation_type="success",
            error_message="",
            total_sentences=total_sentences,
            cited_sentences=cited_sentences,
            uncited_sentences=uncited_sentences,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            failed_sentences=[],
            unmapped_citations=[]
        )
    
    def _parse_sentences(self, answer: str) -> List[AnswerSentence]:
        """
        Parse answer into sentences with citations.
        
        Args:
            answer: Answer text
            
        Returns:
            List of AnswerSentence objects
        """
        # Remove source list section (everything after "**Sources:**")
        if "**Sources:**" in answer:
            answer = answer.split("**Sources:**")[0]
        
        # Split into sentences (basic sentence splitting)
        # This handles periods followed by space or newline
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
        
        sentences = []
        for i, sent in enumerate(raw_sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            # Look for citation at end of sentence
            citation_match = self.citation_pattern.search(sent)
            
            if citation_match:
                # Extract citation
                class_num = int(citation_match.group(1))
                subject = citation_match.group(2).strip()
                chapter_num = int(citation_match.group(3))
                page_num = int(citation_match.group(4))
                
                citation = CitationReference(
                    class_number=class_num,
                    subject=subject,
                    chapter_number=chapter_num,
                    page_number=page_num
                )
                
                sentences.append(AnswerSentence(
                    text=sent,
                    sentence_index=i,
                    citation=citation,
                    has_citation=True
                ))
            else:
                # No citation found
                sentences.append(AnswerSentence(
                    text=sent,
                    sentence_index=i,
                    citation=None,
                    has_citation=False
                ))
        
        return sentences
    
    def _citation_maps_to_chunk(
        self,
        citation: CitationReference,
        retrieved_chunks: List[Any]
    ) -> bool:
        """
        Check if citation references an actual retrieved chunk.
        
        Args:
            citation: Citation to verify
            retrieved_chunks: Retrieved chunks
            
        Returns:
            True if citation matches at least one chunk
        """
        for chunk in retrieved_chunks:
            if citation.matches_chunk(chunk):
                return True
        
        return False
    
    def get_validation_report(self, result: ValidationResult) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            result: ValidationResult object
            
        Returns:
            Formatted report string
        """
        if result.is_valid:
            return f"""
✓ CITATION VALIDATION PASSED

Total Sentences: {result.total_sentences}
Cited Sentences: {result.cited_sentences}
Valid Citations: {result.valid_citations}

All sentences properly cited and mapped to retrieved content.
"""
        else:
            report = [f"\n✗ CITATION VALIDATION FAILED"]
            report.append(f"\nValidation Type: {result.validation_type.upper()}")
            report.append(f"Error: {result.error_message}")
            report.append(f"\nStatistics:")
            report.append(f"  Total Sentences: {result.total_sentences}")
            report.append(f"  Cited Sentences: {result.cited_sentences}")
            report.append(f"  Uncited Sentences: {result.uncited_sentences}")
            report.append(f"  Valid Citations: {result.valid_citations}")
            report.append(f"  Invalid Citations: {result.invalid_citations}")
            
            if result.failed_sentences:
                report.append(f"\nFailed Sentences ({len(result.failed_sentences)}):")
                for sent in result.failed_sentences[:3]:  # Show first 3
                    report.append(f"  • Sentence {sent.sentence_index}: {sent.text[:80]}...")
                    if not sent.has_citation:
                        report.append(f"    → MISSING CITATION")
                    else:
                        report.append(f"    → INVALID CITATION: {sent.citation.to_string()}")
            
            if result.unmapped_citations:
                report.append(f"\nUnmapped Citations ({len(result.unmapped_citations)}):")
                for cite in result.unmapped_citations[:3]:  # Show first 3
                    report.append(f"  • {cite.to_string()}")
                    report.append(f"    → Does not match any retrieved chunk")
            
            return "\n".join(report)


class CitationFormatter:
    """
    Formats citations in the required format.
    
    Converts from internal chunk metadata to:
    (NCERT Class X, Subject, Chapter Y, Page Z)
    """
    
    @staticmethod
    def format_citation_from_chunk(chunk) -> str:
        """
        Format citation from chunk metadata.
        
        Args:
            chunk: RetrievalResult or Chunk object
            
        Returns:
            Formatted citation string
        """
        meta = chunk.metadata
        
        class_num = meta.get('class_number', 'X')
        subject = meta.get('subject', 'Unknown').title()
        chapter_num = meta.get('chapter_number', 'Y')
        pages = meta.get('page_numbers', [])
        page_num = pages[0] if pages else 'Z'
        
        return (
            f"(NCERT Class {class_num}, "
            f"{subject}, "
            f"Chapter {chapter_num}, "
            f"Page {page_num})"
        )
    
    @staticmethod
    def format_answer_with_citations(
        answer_text: str,
        chunk_citations: Dict[int, List[int]]
    ) -> str:
        """
        Add citations to answer text.
        
        Args:
            answer_text: Plain answer text
            chunk_citations: Mapping of sentence index to chunk indices
            
        Returns:
            Answer with inline citations
        """
        # This is a simplified version - real implementation would be more sophisticated
        return answer_text


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def verify_answer_citations(
    answer: str,
    retrieved_chunks: List[Any],
    strict: bool = True
) -> Tuple[bool, str]:
    """
    Convenience function to verify answer citations.
    
    Args:
        answer: Generated answer
        retrieved_chunks: Retrieved chunks
        strict: Strict mode (require all sentences cited)
        
    Returns:
        (is_valid, report)
    """
    verifier = CitationVerifier(require_all_sentences_cited=strict)
    result = verifier.verify(answer, retrieved_chunks, strict_mode=strict)
    report = verifier.get_validation_report(result)
    
    return result.is_valid, report


def validate_or_reject(
    answer: str,
    retrieved_chunks: List[Any]
) -> str:
    """
    Validate answer or return rejection message.
    
    Args:
        answer: Generated answer
        retrieved_chunks: Retrieved chunks
        
    Returns:
        Original answer if valid, rejection message if invalid
    """
    is_valid, report = verify_answer_citations(answer, retrieved_chunks, strict=True)
    
    if is_valid:
        return answer
    else:
        logger.warning(f"Answer rejected due to citation validation failure:\n{report}")
        return "I don't know based on NCERT textbooks."
