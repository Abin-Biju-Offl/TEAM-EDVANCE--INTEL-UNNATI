"""
Safety Mechanism Service - Phase 7

5-layer safety checks to prevent hallucinations and ensure answer quality.
Implements "I don't know" mechanism from Phase 7.
"""

from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from loguru import logger
import re

from app.core.config import settings
from app.models.schemas import Citation


class RejectionReason(Enum):
    """Reasons for rejecting a query/answer"""
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    OFF_TOPIC = "off_topic"
    MISSING_CITATIONS = "missing_citations"
    POOR_GROUNDING = "poor_grounding"


@dataclass
class SafetyCheckResult:
    """Result of a safety check"""
    passed: bool
    reason: Optional[RejectionReason] = None
    message: Optional[str] = None
    score: Optional[float] = None


class SafetyMechanismService:
    """Service for 5-layer safety checks"""
    
    def __init__(self):
        """Initialize safety thresholds from config"""
        self.min_retrieval_confidence = settings.min_retrieval_confidence
        self.min_chunks_required = settings.min_chunks_required
        self.min_grounding_score = settings.min_grounding_score
        
        logger.info(
            f"Safety mechanism initialized: "
            f"confidence>={self.min_retrieval_confidence}, "
            f"chunks>={self.min_chunks_required}, "
            f"grounding>={self.min_grounding_score}"
        )
    
    def check_all(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        answer: Optional[str] = None,
        citations: Optional[List[Citation]] = None,
        grounding_score: Optional[float] = None
    ) -> SafetyCheckResult:
        """
        Run all 5 safety checks
        
        Args:
            question: User question
            retrieved_chunks: Retrieved chunks from FAISS
            answer: Generated answer (optional, for post-generation checks)
            citations: Extracted citations (optional)
            grounding_score: Grounding score (optional)
            
        Returns:
            SafetyCheckResult with pass/fail and reason
        """
        # Layer 1: Retrieval Confidence Check
        result = self.check_retrieval_confidence(retrieved_chunks)
        if not result.passed:
            return result
        
        # Layer 2: Context Sufficiency Check
        result = self.check_context_sufficiency(retrieved_chunks)
        if not result.passed:
            return result
        
        # Layer 3: Topic Relevance Check
        result = self.check_topic_relevance(question, retrieved_chunks)
        if not result.passed:
            return result
        
        # If answer is provided, run post-generation checks
        if answer is not None:
            # Layer 4: Citation Validation Check
            result = self.check_citations(answer, citations)
            if not result.passed:
                return result
            
            # Layer 5: Answer Grounding Check
            result = self.check_answer_grounding(grounding_score)
            if not result.passed:
                return result
        
        # All checks passed
        return SafetyCheckResult(
            passed=True,
            message="All safety checks passed"
        )
    
    def check_retrieval_confidence(self, chunks: List[Dict]) -> SafetyCheckResult:
        """
        Layer 1: Check if retrieval confidence is sufficient
        
        WHY: Low similarity scores indicate question may be out of scope
        """
        if not chunks:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.LOW_CONFIDENCE,
                message="No relevant content found in NCERT textbooks. The question may not be covered in the selected class and subject.",
                score=0.0
            )
        
        # Calculate average confidence
        scores = [chunk.get('score', 0.0) for chunk in chunks]
        avg_confidence = sum(scores) / len(scores) if scores else 0.0
        
        if avg_confidence < self.min_retrieval_confidence:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.LOW_CONFIDENCE,
                message=f"Retrieval confidence too low ({avg_confidence:.2f})",
                score=avg_confidence
            )
        
        return SafetyCheckResult(passed=True, score=avg_confidence)
    
    def check_context_sufficiency(self, chunks: List[Dict]) -> SafetyCheckResult:
        """
        Layer 2: Check if sufficient context is available
        
        WHY: Insufficient context leads to incomplete or wrong answers
        """
        if len(chunks) < self.min_chunks_required:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.INSUFFICIENT_CONTEXT,
                message=f"Only {len(chunks)} relevant chunks found (need {self.min_chunks_required})"
            )
        
        # Check total context length
        total_chars = sum(len(chunk.get('text', '')) for chunk in chunks)
        
        if total_chars < 100:  # Minimum 100 characters of context
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.INSUFFICIENT_CONTEXT,
                message=f"Context too short ({total_chars} chars)"
            )
        
        return SafetyCheckResult(passed=True)
    
    def check_topic_relevance(self, question: str, chunks: List[Dict]) -> SafetyCheckResult:
        """
        Layer 3: Check if retrieved chunks are actually relevant to question
        
        WHY: High similarity but wrong topic still happens
        """
        if not chunks:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.OFF_TOPIC,
                message="No relevant content found"
            )
        
        # Get best similarity score
        best_score = max(chunk.get('score', 0.0) for chunk in chunks)
        
        # Very low best score means likely off-topic
        if best_score < 0.3:  # Lower threshold for best match
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.OFF_TOPIC,
                message=f"Question appears outside NCERT scope (best match: {best_score:.2f})"
            )
        
        return SafetyCheckResult(passed=True, score=best_score)
    
    def check_citations(self, answer: str, citations: Optional[List[Citation]]) -> SafetyCheckResult:
        """
        Layer 4: Check if answer has proper citations
        
        WHY: Citations are mandatory for transparency
        """
        if not answer:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.MISSING_CITATIONS,
                message="No answer generated"
            )
        
        # Check for citation markers in answer
        citation_markers = re.findall(r'\[Source \d+\]', answer)
        
        if not citation_markers:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.MISSING_CITATIONS,
                message="Answer has no citations"
            )
        
        if not citations or len(citations) == 0:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.MISSING_CITATIONS,
                message="No valid citations extracted"
            )
        
        # Check if most sentences have citations
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if sentences and len(citation_markers) < len(sentences) * 0.5:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.MISSING_CITATIONS,
                message="Insufficient citation coverage"
            )
        
        return SafetyCheckResult(passed=True)
    
    def check_answer_grounding(self, grounding_score: Optional[float]) -> SafetyCheckResult:
        """
        Layer 5: Check if answer is well-grounded in context
        
        WHY: Final check that answer actually comes from retrieved text
        """
        if grounding_score is None:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.POOR_GROUNDING,
                message="Grounding score not available"
            )
        
        if grounding_score < self.min_grounding_score:
            return SafetyCheckResult(
                passed=False,
                reason=RejectionReason.POOR_GROUNDING,
                message=f"Answer not well-grounded in context ({grounding_score:.2f})",
                score=grounding_score
            )
        
        return SafetyCheckResult(passed=True, score=grounding_score)


# Global safety mechanism instance
safety_service = SafetyMechanismService()
