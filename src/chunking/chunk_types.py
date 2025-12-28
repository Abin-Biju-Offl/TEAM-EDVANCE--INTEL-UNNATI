"""
Educational Chunk Types for NCERT Content
==========================================

Defines specific chunk types optimized for educational QA.
Each type has different characteristics for retrieval and generation.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict


class ChunkType(Enum):
    """
    Educational chunk types with specific pedagogical purposes.
    
    These types help the RAG system understand the nature of content
    and generate appropriate responses.
    """
    
    # Core conceptual content
    DEFINITION = "definition"           # Formal definitions of concepts
    THEOREM = "theorem"                 # Mathematical theorems and laws
    PROOF = "proof"                     # Logical proofs and derivations
    FORMULA = "formula"                 # Mathematical formulas and equations
    
    # Explanatory content
    EXPLANATION = "explanation"         # Concept explanations and elaborations
    EXAMPLE = "example"                 # Worked examples with solutions
    NOTE = "note"                       # Important notes and remarks
    
    # Practice content
    EXERCISE = "exercise"               # Practice problems
    QUESTION = "question"               # Individual questions
    SOLUTION = "solution"               # Solution to exercises/questions
    
    # Supplementary content
    SUMMARY = "summary"                 # Chapter/section summaries
    ACTIVITY = "activity"               # Hands-on activities
    CONTEXT = "context"                 # General text and context
    MIXED = "mixed"                     # Multiple structure types


@dataclass
class ChunkMetadata:
    """
    Complete metadata for each chunk.
    
    Provides full curriculum context and traceability.
    """
    # Curriculum identifiers
    class_number: int                   # NCERT class (1-12)
    subject: str                        # Subject name
    chapter_number: int                 # Chapter number
    chapter_title: str                  # Chapter title
    section_number: Optional[str] = None  # Section (e.g., "5.3")
    
    # Source tracking
    source_file: str = ""               # Original PDF/file
    page_numbers: list = None           # List of page numbers in chunk
    
    # Language and content
    language: str = "eng"               # Language code
    chunk_type: str = "context"         # Type from ChunkType enum
    
    # Chunk characteristics
    chunk_id: str = ""                  # Unique identifier
    token_count: int = 0                # Approximate token count
    char_count: int = 0                 # Character count
    
    # Educational context
    has_equations: bool = False         # Contains mathematical equations
    has_examples: bool = False          # Contains worked examples
    has_exercises: bool = False         # Contains practice problems
    prerequisite_concepts: list = None  # Related concepts (optional)
    
    # Quality indicators
    structure_confidence: float = 1.0   # Confidence in structure detection
    completeness: str = "complete"      # complete, partial, fragment
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'class_number': self.class_number,
            'subject': self.subject,
            'chapter_number': self.chapter_number,
            'chapter_title': self.chapter_title,
            'section_number': self.section_number,
            'source_file': self.source_file,
            'page_numbers': self.page_numbers or [],
            'language': self.language,
            'chunk_type': self.chunk_type,
            'chunk_id': self.chunk_id,
            'token_count': self.token_count,
            'char_count': self.char_count,
            'has_equations': self.has_equations,
            'has_examples': self.has_examples,
            'has_exercises': self.has_exercises,
            'prerequisite_concepts': self.prerequisite_concepts or [],
            'structure_confidence': self.structure_confidence,
            'completeness': self.completeness
        }


# Chunk type characteristics for retrieval optimization
CHUNK_TYPE_CHARACTERISTICS = {
    ChunkType.DEFINITION: {
        "priority": "critical",          # High importance for concept understanding
        "retrieval_weight": 1.5,         # Boost in retrieval scoring
        "should_be_atomic": True,        # Don't split definitions
        "typical_length": "short",       # Usually concise
        "requires_context": False        # Self-contained
    },
    ChunkType.THEOREM: {
        "priority": "critical",
        "retrieval_weight": 1.4,
        "should_be_atomic": True,
        "typical_length": "medium",
        "requires_context": False
    },
    ChunkType.PROOF: {
        "priority": "high",
        "retrieval_weight": 1.2,
        "should_be_atomic": True,
        "typical_length": "long",
        "requires_context": True         # Usually follows theorem
    },
    ChunkType.EXAMPLE: {
        "priority": "high",
        "retrieval_weight": 1.3,
        "should_be_atomic": True,
        "typical_length": "medium",
        "requires_context": False
    },
    ChunkType.EXPLANATION: {
        "priority": "medium",
        "retrieval_weight": 1.0,
        "should_be_atomic": False,
        "typical_length": "medium",
        "requires_context": True
    },
    ChunkType.EXERCISE: {
        "priority": "high",
        "retrieval_weight": 1.1,
        "should_be_atomic": False,       # Can split exercises
        "typical_length": "variable",
        "requires_context": False
    },
    ChunkType.QUESTION: {
        "priority": "high",
        "retrieval_weight": 1.2,
        "should_be_atomic": True,
        "typical_length": "short",
        "requires_context": False
    },
    ChunkType.SUMMARY: {
        "priority": "medium",
        "retrieval_weight": 1.1,
        "should_be_atomic": True,
        "typical_length": "medium",
        "requires_context": False
    },
    ChunkType.CONTEXT: {
        "priority": "low",
        "retrieval_weight": 0.9,
        "should_be_atomic": False,
        "typical_length": "medium",
        "requires_context": True
    }
}


def get_chunk_type_from_structure(structure_type: str) -> ChunkType:
    """
    Map structure type from cleaning module to chunk type.
    
    Args:
        structure_type: Type from StructureRecovery
    
    Returns:
        Corresponding ChunkType
    """
    mapping = {
        'definition': ChunkType.DEFINITION,
        'theorem': ChunkType.THEOREM,
        'proof': ChunkType.PROOF,
        'formula': ChunkType.FORMULA,
        'example': ChunkType.EXAMPLE,
        'solution': ChunkType.SOLUTION,
        'exercise': ChunkType.EXERCISE,
        'question': ChunkType.QUESTION,
        'note': ChunkType.NOTE,
        'remark': ChunkType.NOTE,
        'summary': ChunkType.SUMMARY,
        'activity': ChunkType.ACTIVITY,
        'explanation': ChunkType.EXPLANATION,
        'corollary': ChunkType.THEOREM,
        'lemma': ChunkType.THEOREM,
    }
    
    return mapping.get(structure_type.lower(), ChunkType.CONTEXT)
