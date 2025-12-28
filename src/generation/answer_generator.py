"""
RAG Answer Generator with Strict Hallucination Prevention
==========================================================

Generates grounded answers with mandatory citations and multiple safety checks.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import logging

from src.generation.prompt_templates import (
    create_strict_rag_prompt,
    classify_query_intent,
    detect_query_language,
    INSUFFICIENT_CONTEXT_RESPONSE,
    create_low_confidence_response
)
from src.generation.citation_formatter import (
    CitationFormatter,
    Citation,
    CitationStyle,
    extract_citations_from_answer,
    verify_citations_exist,
    format_answer_with_sources
)

logger = logging.getLogger(__name__)


class AnswerStatus(Enum):
    """Status of answer generation."""
    SUCCESS = "success"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    LOW_CONFIDENCE = "low_confidence"
    HALLUCINATION_DETECTED = "hallucination_detected"
    ERROR = "error"


@dataclass
class GenerationConfig:
    """Configuration for answer generation."""
    # Model settings
    model_name: str = "gpt-4"  # or "gpt-3.5-turbo", "llama-2-70b", etc.
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 500
    
    # Confidence thresholds (from Phase 5)
    min_confidence: float = 0.6  # Reject below this
    high_confidence: float = 0.8  # Safe above this
    
    # Citation settings
    citation_style: CitationStyle = CitationStyle.INLINE
    require_citations: bool = True
    
    # Hallucination prevention
    enable_hallucination_detection: bool = True
    enable_answer_verification: bool = True
    
    # Language settings
    auto_detect_language: bool = True
    enforce_language_match: bool = True


@dataclass
class GeneratedAnswer:
    """Container for a generated answer with metadata."""
    answer: str
    status: AnswerStatus
    confidence: float
    citations: List[Citation]
    sources_used: List[str]  # List of chunk IDs
    language: str
    
    # Hallucination detection
    hallucination_detected: bool
    hallucination_reason: str
    
    # Metadata
    query: str
    num_sources: int
    generation_time_ms: float
    
    def is_safe(self) -> bool:
        """Check if answer is safe to use."""
        return (
            self.status == AnswerStatus.SUCCESS and
            not self.hallucination_detected and
            self.confidence >= 0.6
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'answer': self.answer,
            'status': self.status.value,
            'confidence': self.confidence,
            'citations': [c.to_full_reference() for c in self.citations],
            'sources_used': self.sources_used,
            'language': self.language,
            'hallucination_detected': self.hallucination_detected,
            'hallucination_reason': self.hallucination_reason,
            'is_safe': self.is_safe(),
        }


class HallucinationDetector:
    """Detects potential hallucinations in generated answers."""
    
    def __init__(self):
        """Initialize hallucination detector."""
        # Known hallucination patterns
        self.hallucination_indicators = [
            # Phrases that suggest external knowledge
            r"as we know",
            r"in general",
            r"it is well known",
            r"common knowledge",
            r"everyone knows",
            r"typically",
            r"usually",
            r"often",
            
            # Hedging that suggests uncertainty
            r"might be",
            r"could be",
            r"possibly",
            r"perhaps",
            
            # References to external sources
            r"according to",
            r"research shows",
            r"studies indicate",
            r"experts say",
        ]
        
        self.citation_patterns = [
            r'\[Source\s+\d+\]',
            r'\[\d+\]',
        ]
    
    def detect(
        self,
        answer: str,
        retrieved_chunks: List[Any],
        require_citations: bool = True
    ) -> Tuple[bool, str]:
        """
        Detect potential hallucinations in answer.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved context chunks
            require_citations: Whether citations are required
            
        Returns:
            (is_hallucination, reason)
        """
        # Check 1: "I don't know" response is always safe
        if "I don't know based on NCERT textbooks" in answer:
            return False, ""
        
        # Check 2: Answer must have citations (if required)
        if require_citations:
            has_citations = bool(re.search(r'\[Source\s+\d+\]|\[\d+\]', answer))
            if not has_citations:
                return True, "Answer contains no citations - likely hallucinated"
        
        # Check 3: Check for hallucination indicator phrases
        answer_lower = answer.lower()
        for pattern in self.hallucination_indicators:
            if re.search(pattern, answer_lower):
                return True, f"Answer contains hallucination indicator: '{pattern}'"
        
        # Check 4: Verify citations reference available sources
        cited_ids = extract_citations_from_answer(answer)
        available_ids = list(range(1, len(retrieved_chunks) + 1))
        
        for cited_id in cited_ids:
            if cited_id not in available_ids:
                return True, f"Answer cites non-existent Source {cited_id}"
        
        # Check 5: Answer length suspiciously longer than context
        total_context_length = sum(len(chunk.content) for chunk in retrieved_chunks)
        if len(answer) > 2 * total_context_length:
            return True, "Answer is much longer than context - likely added information"
        
        # All checks passed
        return False, ""
    
    def verify_answer_grounding(
        self,
        answer: str,
        retrieved_chunks: List[Any]
    ) -> Tuple[bool, float]:
        """
        Verify that answer is grounded in retrieved context.
        
        Args:
            answer: Generated answer text
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            (is_grounded, grounding_score)
        """
        # Extract key phrases from answer (simplified n-gram matching)
        answer_clean = re.sub(r'\[Source\s+\d+\]|\[\d+\]', '', answer)
        answer_sentences = [s.strip() for s in answer_clean.split('.') if s.strip()]
        
        if not answer_sentences:
            return False, 0.0
        
        # Check how many answer sentences have overlap with context
        grounded_sentences = 0
        
        for sentence in answer_sentences:
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue
            
            # Check if sentence overlaps with any chunk
            sentence_lower = sentence.lower()
            
            for chunk in retrieved_chunks:
                chunk_lower = chunk.content.lower()
                
                # Simple overlap check: if 50%+ of words appear in chunk
                sentence_words = set(sentence_lower.split())
                chunk_words = set(chunk_lower.split())
                
                overlap = len(sentence_words & chunk_words) / len(sentence_words)
                
                if overlap >= 0.5:
                    grounded_sentences += 1
                    break
        
        grounding_score = grounded_sentences / len(answer_sentences)
        is_grounded = grounding_score >= 0.7  # 70% of sentences must be grounded
        
        return is_grounded, grounding_score


class RAGAnswerGenerator:
    """
    Strict RAG answer generator with hallucination prevention.
    
    Key Features:
    - Only uses retrieved context
    - Mandatory citations
    - Multiple hallucination checks
    - Grade-appropriate language
    - Multilingual support
    """
    
    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        llm_client: Optional[Any] = None
    ):
        """
        Initialize answer generator.
        
        Args:
            config: Generation configuration
            llm_client: Optional LLM client (OpenAI, etc.)
        """
        self.config = config or GenerationConfig()
        self.llm_client = llm_client
        self.hallucination_detector = HallucinationDetector()
        
        logger.info(f"Initialized RAG generator with model: {self.config.model_name}")
    
    def generate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        class_number: int = 10,
    ) -> GeneratedAnswer:
        """
        Generate an answer with strict grounding.
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks from Phase 5
            class_number: Student's class/grade level
            
        Returns:
            GeneratedAnswer with answer text and metadata
        """
        import time
        start_time = time.time()
        
        # Check 1: Do we have any context?
        if not retrieved_chunks:
            return self._create_insufficient_context_response(
                query=query,
                generation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check 2: Is confidence too low?
        avg_confidence = sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks)
        
        if avg_confidence < self.config.min_confidence:
            return self._create_low_confidence_response(
                query=query,
                retrieved_chunks=retrieved_chunks,
                avg_confidence=avg_confidence,
                generation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check 3: Detect query language
        if self.config.auto_detect_language:
            query_language = detect_query_language(query)
        else:
            query_language = 'eng'
        
        # Build prompt
        prompt_type = classify_query_intent(query)
        prompt = create_strict_rag_prompt(
            query=query,
            retrieved_chunks=retrieved_chunks,
            class_number=class_number,
            prompt_type=prompt_type
        )
        
        # Generate answer
        try:
            answer_text = self._call_llm(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._create_error_response(
                query=query,
                error=str(e),
                generation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Check 4: Detect hallucinations
        hallucination_detected = False
        hallucination_reason = ""
        
        if self.config.enable_hallucination_detection:
            hallucination_detected, hallucination_reason = \
                self.hallucination_detector.detect(
                    answer=answer_text,
                    retrieved_chunks=retrieved_chunks,
                    require_citations=self.config.require_citations
                )
            
            if hallucination_detected:
                logger.warning(f"Hallucination detected: {hallucination_reason}")
                return self._create_hallucination_response(
                    query=query,
                    reason=hallucination_reason,
                    generation_time_ms=(time.time() - start_time) * 1000
                )
        
        # Check 5: Verify grounding
        if self.config.enable_answer_verification:
            is_grounded, grounding_score = \
                self.hallucination_detector.verify_answer_grounding(
                    answer=answer_text,
                    retrieved_chunks=retrieved_chunks
                )
            
            if not is_grounded:
                logger.warning(f"Answer not grounded (score: {grounding_score:.2f})")
                return self._create_insufficient_context_response(
                    query=query,
                    generation_time_ms=(time.time() - start_time) * 1000
                )
        
        # Create citations
        formatter = CitationFormatter(style=self.config.citation_style)
        citations = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = formatter.add_citation_from_chunk(chunk, source_id=i)
            citations.append(citation)
        
        # Check 6: Verify cited sources exist
        cited_ids = extract_citations_from_answer(answer_text)
        available_ids = [c.source_id for c in citations]
        
        if not verify_citations_exist(answer_text, available_ids):
            return self._create_hallucination_response(
                query=query,
                reason="Answer cites non-existent sources",
                generation_time_ms=(time.time() - start_time) * 1000
            )
        
        # Format final answer with sources
        final_answer = format_answer_with_sources(answer_text, citations)
        
        generation_time = (time.time() - start_time) * 1000
        
        return GeneratedAnswer(
            answer=final_answer,
            status=AnswerStatus.SUCCESS,
            confidence=avg_confidence,
            citations=citations,
            sources_used=[c.chunk_id for c in retrieved_chunks],
            language=query_language,
            hallucination_detected=False,
            hallucination_reason="",
            query=query,
            num_sources=len(retrieved_chunks),
            generation_time_ms=generation_time
        )
    
    def _call_llm(self, prompt) -> str:
        """
        Call LLM to generate answer.
        
        Args:
            prompt: PromptTemplate object
            
        Returns:
            Generated answer text
        """
        # If no client provided, return mock response
        if self.llm_client is None:
            return self._mock_llm_call(prompt)
        
        # Call OpenAI API (example)
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.model_name,
                messages=prompt.to_messages(),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def _mock_llm_call(self, prompt) -> str:
        """
        Mock LLM call for testing without API.
        
        Returns a template response that demonstrates proper format.
        """
        return """An arithmetic progression (AP) is a sequence of numbers where each term after the first is obtained by adding a fixed number to the preceding term [Source 1]. This fixed number is called the common difference and is denoted by d [Source 1].

For example, in the AP: 3, 7, 11, 15, the common difference d = 7 - 3 = 4 [Source 2].

The nth term of an arithmetic progression with first term a and common difference d is given by the formula: an = a + (n-1)d [Source 3]."""
    
    def _create_insufficient_context_response(
        self,
        query: str,
        generation_time_ms: float
    ) -> GeneratedAnswer:
        """Create response when context is insufficient."""
        return GeneratedAnswer(
            answer=INSUFFICIENT_CONTEXT_RESPONSE,
            status=AnswerStatus.INSUFFICIENT_CONTEXT,
            confidence=0.0,
            citations=[],
            sources_used=[],
            language=detect_query_language(query),
            hallucination_detected=False,
            hallucination_reason="",
            query=query,
            num_sources=0,
            generation_time_ms=generation_time_ms
        )
    
    def _create_low_confidence_response(
        self,
        query: str,
        retrieved_chunks: List[Any],
        avg_confidence: float,
        generation_time_ms: float
    ) -> GeneratedAnswer:
        """Create response when confidence is too low."""
        return GeneratedAnswer(
            answer=INSUFFICIENT_CONTEXT_RESPONSE,
            status=AnswerStatus.LOW_CONFIDENCE,
            confidence=avg_confidence,
            citations=[],
            sources_used=[c.chunk_id for c in retrieved_chunks],
            language=detect_query_language(query),
            hallucination_detected=False,
            hallucination_reason=f"Average confidence {avg_confidence:.2f} below threshold {self.config.min_confidence}",
            query=query,
            num_sources=len(retrieved_chunks),
            generation_time_ms=generation_time_ms
        )
    
    def _create_hallucination_response(
        self,
        query: str,
        reason: str,
        generation_time_ms: float
    ) -> GeneratedAnswer:
        """Create response when hallucination is detected."""
        return GeneratedAnswer(
            answer=INSUFFICIENT_CONTEXT_RESPONSE,
            status=AnswerStatus.HALLUCINATION_DETECTED,
            confidence=0.0,
            citations=[],
            sources_used=[],
            language=detect_query_language(query),
            hallucination_detected=True,
            hallucination_reason=reason,
            query=query,
            num_sources=0,
            generation_time_ms=generation_time_ms
        )
    
    def _create_error_response(
        self,
        query: str,
        error: str,
        generation_time_ms: float
    ) -> GeneratedAnswer:
        """Create response when an error occurs."""
        return GeneratedAnswer(
            answer=INSUFFICIENT_CONTEXT_RESPONSE,
            status=AnswerStatus.ERROR,
            confidence=0.0,
            citations=[],
            sources_used=[],
            language=detect_query_language(query),
            hallucination_detected=False,
            hallucination_reason=f"Error: {error}",
            query=query,
            num_sources=0,
            generation_time_ms=generation_time_ms
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_answer(
    query: str,
    retrieved_chunks: List[Any],
    class_number: int = 10,
    llm_client: Optional[Any] = None
) -> GeneratedAnswer:
    """
    Convenience function for quick answer generation.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved context chunks
        class_number: Student's class/grade
        llm_client: Optional LLM client
        
    Returns:
        GeneratedAnswer
    """
    generator = RAGAnswerGenerator(llm_client=llm_client)
    return generator.generate(query, retrieved_chunks, class_number)
