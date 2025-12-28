"""
Answer Generation Module
========================

Strict RAG-based answer generation with hallucination prevention.

Key Features:
- Strict grounding: Answers ONLY from retrieved context
- Mandatory citations: Every claim must be cited
- Grade-appropriate language: Matches student level
- Multilingual support: Answers in query language
- Hallucination prevention: Multiple safety checks

Usage:
    from src.generation import RAGAnswerGenerator, PromptTemplate
    
    generator = RAGAnswerGenerator()
    answer = generator.generate(
        query="What is an arithmetic progression?",
        retrieved_chunks=[...],
        class_number=10
    )
"""

from src.generation.prompt_templates import (
    PromptTemplate,
    SystemPromptBuilder,
    STRICT_RAG_SYSTEM_PROMPT,
    CITATION_FORMAT_INSTRUCTIONS
)

from src.generation.answer_generator import (
    RAGAnswerGenerator,
    GenerationConfig,
    GeneratedAnswer,
    HallucinationDetector
)

from src.generation.citation_formatter import (
    CitationFormatter,
    Citation,
    CitationStyle
)

from src.generation.citation_verifier import (
    CitationVerifier,
    CitationReference,
    ValidationResult,
    verify_answer_citations,
    validate_or_reject
)

from src.generation.safety_mechanism import (
    SafetyMechanism,
    SafetyThresholds,
    SafetyCheckResult,
    RejectionReason,
    check_safety,
    get_safe_answer
)

__all__ = [
    # Prompt templates
    'PromptTemplate',
    'SystemPromptBuilder',
    'STRICT_RAG_SYSTEM_PROMPT',
    'CITATION_FORMAT_INSTRUCTIONS',
    
    # Answer generation
    'RAGAnswerGenerator',
    'GenerationConfig',
    'GeneratedAnswer',
    'HallucinationDetector',
    
    # Citation formatting
    'CitationFormatter',
    'Citation',
    'CitationStyle',
    
    # Citation verification
    'CitationVerifier',
    'CitationReference',
    'ValidationResult',
    'verify_answer_citations',
    'validate_or_reject',
    
    # Safety mechanism
    'SafetyMechanism',
    'SafetyThresholds',
    'SafetyCheckResult',
    'RejectionReason',
    'check_safety',
    'get_safe_answer',
]
