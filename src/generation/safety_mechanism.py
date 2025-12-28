"""
"I Don't Know" Safety Mechanism
================================

Robust safety system that rejects answers when system lacks confidence.

Philosophy: "It's better to say 'I don't know' than to hallucinate."

Rejection Triggers:
1. Low retrieval confidence (< 0.6)
2. Insufficient context (empty or too few chunks)
3. Missing/invalid citations
4. Off-topic queries (outside NCERT scope)
5. Low answer grounding score
6. Hallucination patterns detected
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Reasons for rejecting an answer."""
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    MISSING_CITATIONS = "missing_citations"
    INVALID_CITATIONS = "invalid_citations"
    OFF_TOPIC = "off_topic"
    LOW_GROUNDING = "low_grounding"
    HALLUCINATION_DETECTED = "hallucination_detected"
    MULTIPLE_ISSUES = "multiple_issues"


@dataclass
class SafetyThresholds:
    """Configurable safety thresholds."""
    # Retrieval confidence
    min_retrieval_confidence: float = 0.6
    min_similarity_score: float = 0.5
    min_rerank_score: float = 0.5
    
    # Context requirements
    min_chunks_required: int = 1
    min_total_context_length: int = 100  # characters
    
    # Citation requirements
    require_citations: bool = True
    min_citation_coverage: float = 0.8  # 80% of sentences must be cited
    
    # Grounding requirements
    min_grounding_score: float = 0.7  # 70% of sentences must overlap with context
    
    # Off-topic detection
    max_topic_mismatch_score: float = 0.3  # If best match < 0.3, likely off-topic


@dataclass
class SafetyCheckResult:
    """Result of safety checks."""
    should_reject: bool
    rejection_reason: Optional[RejectionReason]
    confidence_score: float
    
    # Detailed check results
    passed_checks: List[str]
    failed_checks: List[str]
    
    # Explanation for user/logs
    explanation: str
    
    # Diagnostic info
    diagnostic_info: Dict[str, Any]


class SafetyMechanism:
    """
    Comprehensive safety mechanism for RAG system.
    
    Implements multiple layers of safety checks to determine
    when system should respond with "I don't know based on NCERT textbooks."
    
    Design Philosophy:
    - False negatives (rejecting valid queries) are acceptable
    - False positives (accepting invalid queries) are NOT acceptable
    - When in doubt, reject
    """
    
    def __init__(self, thresholds: Optional[SafetyThresholds] = None):
        """
        Initialize safety mechanism.
        
        Args:
            thresholds: Custom safety thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or SafetyThresholds()
        self.rejection_message = "I don't know based on NCERT textbooks."
    
    def should_reject_query(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        class_number: Optional[int] = None,
        subject: Optional[str] = None
    ) -> SafetyCheckResult:
        """
        Comprehensive safety check - determine if query should be rejected.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved chunks from Phase 5
            generated_answer: Generated answer (if available)
            class_number: Student's class
            subject: Expected subject
            
        Returns:
            SafetyCheckResult with decision and explanation
        """
        passed_checks = []
        failed_checks = []
        diagnostic_info = {}
        
        # ================================================================
        # CHECK 1: RETRIEVAL CONFIDENCE
        # ================================================================
        confidence_check = self._check_retrieval_confidence(retrieved_chunks)
        
        if confidence_check['passed']:
            passed_checks.append("Retrieval confidence acceptable")
        else:
            failed_checks.append(confidence_check['reason'])
        
        diagnostic_info['confidence'] = confidence_check
        
        # ================================================================
        # CHECK 2: CONTEXT SUFFICIENCY
        # ================================================================
        context_check = self._check_context_sufficiency(retrieved_chunks)
        
        if context_check['passed']:
            passed_checks.append("Context sufficient")
        else:
            failed_checks.append(context_check['reason'])
        
        diagnostic_info['context'] = context_check
        
        # ================================================================
        # CHECK 3: TOPIC RELEVANCE (Off-topic detection)
        # ================================================================
        topic_check = self._check_topic_relevance(
            query, retrieved_chunks, class_number, subject
        )
        
        if topic_check['passed']:
            passed_checks.append("Topic relevant to NCERT scope")
        else:
            failed_checks.append(topic_check['reason'])
        
        diagnostic_info['topic'] = topic_check
        
        # ================================================================
        # CHECK 4: CITATIONS (if answer provided)
        # ================================================================
        if generated_answer:
            citation_check = self._check_citations(generated_answer, retrieved_chunks)
            
            if citation_check['passed']:
                passed_checks.append("Citations valid")
            else:
                failed_checks.append(citation_check['reason'])
            
            diagnostic_info['citations'] = citation_check
        
        # ================================================================
        # CHECK 5: ANSWER GROUNDING (if answer provided)
        # ================================================================
        if generated_answer:
            grounding_check = self._check_answer_grounding(
                generated_answer, retrieved_chunks
            )
            
            if grounding_check['passed']:
                passed_checks.append("Answer well-grounded")
            else:
                failed_checks.append(grounding_check['reason'])
            
            diagnostic_info['grounding'] = grounding_check
        
        # ================================================================
        # DECISION: Should we reject?
        # ================================================================
        should_reject = len(failed_checks) > 0
        
        if should_reject:
            # Determine primary rejection reason
            if 'confidence' in [check['check_name'] for check in 
                                [diagnostic_info.get('confidence', {})] if not check.get('passed')]:
                rejection_reason = RejectionReason.LOW_CONFIDENCE
            elif 'context' in [check['check_name'] for check in 
                               [diagnostic_info.get('context', {})] if not check.get('passed')]:
                rejection_reason = RejectionReason.INSUFFICIENT_CONTEXT
            elif 'topic' in [check['check_name'] for check in 
                             [diagnostic_info.get('topic', {})] if not check.get('passed')]:
                rejection_reason = RejectionReason.OFF_TOPIC
            elif generated_answer and 'citations' in [check['check_name'] for check in 
                                                       [diagnostic_info.get('citations', {})] if not check.get('passed')]:
                rejection_reason = RejectionReason.MISSING_CITATIONS
            elif generated_answer and 'grounding' in [check['check_name'] for check in 
                                                       [diagnostic_info.get('grounding', {})] if not check.get('passed')]:
                rejection_reason = RejectionReason.LOW_GROUNDING
            else:
                rejection_reason = RejectionReason.MULTIPLE_ISSUES
            
            explanation = self._build_rejection_explanation(
                failed_checks, rejection_reason
            )
            confidence_score = 0.0
        else:
            rejection_reason = None
            explanation = "All safety checks passed"
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(diagnostic_info)
        
        return SafetyCheckResult(
            should_reject=should_reject,
            rejection_reason=rejection_reason,
            confidence_score=confidence_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            explanation=explanation,
            diagnostic_info=diagnostic_info
        )
    
    def _check_retrieval_confidence(self, retrieved_chunks: List[Any]) -> Dict[str, Any]:
        """
        Check if retrieval confidence meets threshold.
        
        Args:
            retrieved_chunks: Retrieved chunks
            
        Returns:
            Check result dict
        """
        if not retrieved_chunks:
            return {
                'check_name': 'confidence',
                'passed': False,
                'reason': 'No chunks retrieved - zero confidence',
                'details': {'avg_confidence': 0.0}
            }
        
        # Calculate average confidence
        confidences = [
            chunk.confidence for chunk in retrieved_chunks 
            if hasattr(chunk, 'confidence')
        ]
        
        if not confidences:
            # Fallback: use similarity scores
            confidences = [
                chunk.similarity_score for chunk in retrieved_chunks
                if hasattr(chunk, 'similarity_score')
            ]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        top_confidence = max(confidences) if confidences else 0.0
        
        # Check thresholds
        if avg_confidence < self.thresholds.min_retrieval_confidence:
            return {
                'check_name': 'confidence',
                'passed': False,
                'reason': f'Average confidence {avg_confidence:.2f} below threshold {self.thresholds.min_retrieval_confidence}',
                'details': {
                    'avg_confidence': avg_confidence,
                    'top_confidence': top_confidence,
                    'threshold': self.thresholds.min_retrieval_confidence
                }
            }
        
        return {
            'check_name': 'confidence',
            'passed': True,
            'reason': '',
            'details': {
                'avg_confidence': avg_confidence,
                'top_confidence': top_confidence,
                'threshold': self.thresholds.min_retrieval_confidence
            }
        }
    
    def _check_context_sufficiency(self, retrieved_chunks: List[Any]) -> Dict[str, Any]:
        """
        Check if retrieved context is sufficient.
        
        Args:
            retrieved_chunks: Retrieved chunks
            
        Returns:
            Check result dict
        """
        # Check 1: Minimum number of chunks
        if len(retrieved_chunks) < self.thresholds.min_chunks_required:
            return {
                'check_name': 'context',
                'passed': False,
                'reason': f'Only {len(retrieved_chunks)} chunk(s) retrieved, need at least {self.thresholds.min_chunks_required}',
                'details': {
                    'num_chunks': len(retrieved_chunks),
                    'min_required': self.thresholds.min_chunks_required
                }
            }
        
        # Check 2: Total context length
        total_length = sum(len(chunk.content) for chunk in retrieved_chunks)
        
        if total_length < self.thresholds.min_total_context_length:
            return {
                'check_name': 'context',
                'passed': False,
                'reason': f'Context length {total_length} chars below minimum {self.thresholds.min_total_context_length}',
                'details': {
                    'total_length': total_length,
                    'min_required': self.thresholds.min_total_context_length
                }
            }
        
        return {
            'check_name': 'context',
            'passed': True,
            'reason': '',
            'details': {
                'num_chunks': len(retrieved_chunks),
                'total_length': total_length
            }
        }
    
    def _check_topic_relevance(
        self,
        query: str,
        retrieved_chunks: List[Any],
        class_number: Optional[int],
        subject: Optional[str]
    ) -> Dict[str, Any]:
        """
        Check if query is relevant to NCERT scope.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved chunks
            class_number: Expected class
            subject: Expected subject
            
        Returns:
            Check result dict
        """
        if not retrieved_chunks:
            return {
                'check_name': 'topic',
                'passed': False,
                'reason': 'No relevant content found - likely off-topic',
                'details': {'best_score': 0.0}
            }
        
        # Get best similarity score
        best_score = max(
            (chunk.similarity_score for chunk in retrieved_chunks
             if hasattr(chunk, 'similarity_score')),
            default=0.0
        )
        
        # If best match is very low, query is likely off-topic
        if best_score < self.thresholds.max_topic_mismatch_score:
            return {
                'check_name': 'topic',
                'passed': False,
                'reason': f'Best match score {best_score:.2f} too low - query likely outside NCERT scope',
                'details': {
                    'best_score': best_score,
                    'threshold': self.thresholds.max_topic_mismatch_score
                }
            }
        
        # Check if retrieved chunks match expected class/subject
        if class_number or subject:
            mismatches = []
            
            for chunk in retrieved_chunks:
                meta = chunk.metadata if hasattr(chunk, 'metadata') else {}
                
                if class_number and meta.get('class_number') != class_number:
                    mismatches.append(f"Wrong class: expected {class_number}, got {meta.get('class_number')}")
                
                if subject and meta.get('subject', '').lower() != subject.lower():
                    mismatches.append(f"Wrong subject: expected {subject}, got {meta.get('subject')}")
            
            # If all chunks mismatch, query is off-topic
            if len(mismatches) == len(retrieved_chunks):
                return {
                    'check_name': 'topic',
                    'passed': False,
                    'reason': 'All retrieved chunks from wrong class/subject - query off-topic',
                    'details': {
                        'best_score': best_score,
                        'mismatches': mismatches[:3]  # Show first 3
                    }
                }
        
        return {
            'check_name': 'topic',
            'passed': True,
            'reason': '',
            'details': {'best_score': best_score}
        }
    
    def _check_citations(
        self,
        answer: str,
        retrieved_chunks: List[Any]
    ) -> Dict[str, Any]:
        """
        Check if answer has valid citations.
        
        Args:
            answer: Generated answer
            retrieved_chunks: Retrieved chunks
            
        Returns:
            Check result dict
        """
        if not self.thresholds.require_citations:
            return {'check_name': 'citations', 'passed': True, 'reason': '', 'details': {}}
        
        # Use citation verifier
        try:
            from src.generation.citation_verifier import CitationVerifier
            
            verifier = CitationVerifier()
            result = verifier.verify(answer, retrieved_chunks, strict_mode=True)
            
            if not result.is_valid:
                return {
                    'check_name': 'citations',
                    'passed': False,
                    'reason': result.error_message,
                    'details': {
                        'cited_sentences': result.cited_sentences,
                        'uncited_sentences': result.uncited_sentences,
                        'invalid_citations': result.invalid_citations
                    }
                }
            
            return {
                'check_name': 'citations',
                'passed': True,
                'reason': '',
                'details': {
                    'cited_sentences': result.cited_sentences,
                    'valid_citations': result.valid_citations
                }
            }
        
        except ImportError:
            # Fallback: simple citation check
            import re
            has_citations = bool(re.search(
                r'\(NCERT Class \d+, [A-Za-z\s]+, Chapter \d+, Page \d+\)',
                answer
            ))
            
            if not has_citations:
                return {
                    'check_name': 'citations',
                    'passed': False,
                    'reason': 'No citations found in answer',
                    'details': {}
                }
            
            return {'check_name': 'citations', 'passed': True, 'reason': '', 'details': {}}
    
    def _check_answer_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Any]
    ) -> Dict[str, Any]:
        """
        Check if answer is well-grounded in retrieved context.
        
        Args:
            answer: Generated answer
            retrieved_chunks: Retrieved chunks
            
        Returns:
            Check result dict
        """
        # Use hallucination detector
        try:
            from src.generation.answer_generator import HallucinationDetector
            
            detector = HallucinationDetector()
            is_grounded, grounding_score = detector.verify_answer_grounding(
                answer, retrieved_chunks
            )
            
            if not is_grounded or grounding_score < self.thresholds.min_grounding_score:
                return {
                    'check_name': 'grounding',
                    'passed': False,
                    'reason': f'Answer grounding score {grounding_score:.2f} below threshold {self.thresholds.min_grounding_score}',
                    'details': {
                        'grounding_score': grounding_score,
                        'threshold': self.thresholds.min_grounding_score
                    }
                }
            
            return {
                'check_name': 'grounding',
                'passed': True,
                'reason': '',
                'details': {'grounding_score': grounding_score}
            }
        
        except ImportError:
            # Skip if hallucination detector not available
            return {'check_name': 'grounding', 'passed': True, 'reason': '', 'details': {}}
    
    def _calculate_overall_confidence(self, diagnostic_info: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score from diagnostic info.
        
        Args:
            diagnostic_info: Diagnostic information from checks
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence_details = diagnostic_info.get('confidence', {}).get('details', {})
        avg_confidence = confidence_details.get('avg_confidence', 0.0)
        
        grounding_details = diagnostic_info.get('grounding', {}).get('details', {})
        grounding_score = grounding_details.get('grounding_score', 1.0)
        
        topic_details = diagnostic_info.get('topic', {}).get('details', {})
        topic_score = topic_details.get('best_score', 0.0)
        
        # Weighted average
        overall = (
            0.5 * avg_confidence +
            0.3 * grounding_score +
            0.2 * topic_score
        )
        
        return min(1.0, max(0.0, overall))
    
    def _build_rejection_explanation(
        self,
        failed_checks: List[str],
        rejection_reason: RejectionReason
    ) -> str:
        """
        Build human-readable explanation for rejection.
        
        Args:
            failed_checks: List of failed check descriptions
            rejection_reason: Primary rejection reason
            
        Returns:
            Explanation string
        """
        explanations = {
            RejectionReason.LOW_CONFIDENCE: (
                "The system has low confidence in the retrieved information. "
                "This query may not have sufficient coverage in NCERT textbooks."
            ),
            RejectionReason.INSUFFICIENT_CONTEXT: (
                "Not enough relevant information was found in NCERT textbooks "
                "to answer this question confidently."
            ),
            RejectionReason.OFF_TOPIC: (
                "This question appears to be outside the scope of NCERT textbooks. "
                "Please ask about topics covered in your NCERT textbooks."
            ),
            RejectionReason.MISSING_CITATIONS: (
                "The generated answer lacked proper citations to textbook sources. "
                "For safety, the system will not provide uncited information."
            ),
            RejectionReason.LOW_GROUNDING: (
                "The generated answer was not sufficiently grounded in the retrieved context. "
                "This suggests potential hallucination."
            ),
            RejectionReason.MULTIPLE_ISSUES: (
                "Multiple safety checks failed. The system cannot provide a reliable answer."
            )
        }
        
        base_explanation = explanations.get(
            rejection_reason,
            "The system cannot provide a reliable answer for this query."
        )
        
        # Add specific failed checks
        if len(failed_checks) <= 2:
            details = "\n\nSpecific issues:\n" + "\n".join(f"  • {check}" for check in failed_checks)
        else:
            details = f"\n\n{len(failed_checks)} safety checks failed."
        
        return base_explanation + details


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def check_safety(
    query: str,
    retrieved_chunks: List[Any],
    generated_answer: Optional[str] = None,
    class_number: Optional[int] = None,
    subject: Optional[str] = None,
    thresholds: Optional[SafetyThresholds] = None
) -> SafetyCheckResult:
    """
    Convenience function to check if query/answer is safe.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved chunks
        generated_answer: Generated answer (optional)
        class_number: Expected class
        subject: Expected subject
        thresholds: Custom thresholds
        
    Returns:
        SafetyCheckResult
    """
    mechanism = SafetyMechanism(thresholds)
    return mechanism.should_reject_query(
        query, retrieved_chunks, generated_answer, class_number, subject
    )


def get_safe_answer(
    query: str,
    retrieved_chunks: List[Any],
    generated_answer: str,
    class_number: Optional[int] = None,
    subject: Optional[str] = None
) -> str:
    """
    Return safe answer or rejection message.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved chunks
        generated_answer: Generated answer
        class_number: Expected class
        subject: Expected subject
        
    Returns:
        Original answer if safe, rejection message otherwise
    """
    result = check_safety(query, retrieved_chunks, generated_answer, class_number, subject)
    
    if result.should_reject:
        logger.warning(f"Answer rejected: {result.explanation}")
        return "I don't know based on NCERT textbooks."
    else:
        return generated_answer
