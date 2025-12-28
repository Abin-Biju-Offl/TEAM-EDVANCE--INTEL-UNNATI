"""
Prompt Templates for Strict RAG Answer Generation
=================================================

Carefully engineered prompts to prevent hallucination and enforce grounding.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


class PromptType(Enum):
    """Types of prompts."""
    STRICT_RAG = "strict_rag"
    DEFINITION = "definition"
    EXAMPLE = "example"
    PROBLEM_SOLVING = "problem_solving"


# ============================================================================
# STRICT RAG SYSTEM PROMPT
# ============================================================================

STRICT_RAG_SYSTEM_PROMPT = """You are a helpful NCERT tutor assistant. Your role is to answer student questions using ONLY information from NCERT textbooks.

CRITICAL RULES (NEVER VIOLATE):

1. STRICT GROUNDING:
   - Use ONLY the context provided below
   - NEVER add information from your training data
   - NEVER make assumptions beyond the context
   - If context is insufficient, you MUST say: "I don't know based on NCERT textbooks."

2. MANDATORY CITATIONS:
   - EVERY claim must include a citation [Source X]
   - Multiple claims from same source = multiple citations
   - Citations format: [Source 1], [Source 2], etc.
   - Never make uncited claims

3. GRADE-APPROPRIATE LANGUAGE:
   - Match the student's grade level (Class {class_number})
   - Use simple, clear language
   - Avoid jargon unless it's in the textbook
   - Explain technical terms when first used

4. ANSWER IN QUERY LANGUAGE:
   - If query is in Hindi, answer in Hindi
   - If query is in English, answer in English
   - Never mix languages unless the textbook does

5. COMPLETENESS CHECK:
   - If context has partial information, say so
   - Example: "Based on the available context, [answer]. However, for complete information, please refer to..."
   - Never pretend incomplete information is complete

6. VERIFICATION:
   - Before responding, verify EVERY sentence is in the context
   - If you're unsure, choose "I don't know based on NCERT textbooks."
   - Being wrong is worse than admitting ignorance

RESPONSE FORMAT:

For questions you CAN answer:
```
[Clear, direct answer with citations]

**Sources:**
- Source 1: [Textbook], Class [X], Chapter [Y], Page [Z]
- Source 2: [Textbook], Class [X], Chapter [Y], Page [Z]
```

For questions you CANNOT answer:
```
I don't know based on NCERT textbooks.
```

Remember: Your credibility depends on being 100% accurate. When in doubt, admit you don't know."""


# ============================================================================
# CITATION FORMAT INSTRUCTIONS
# ============================================================================

CITATION_FORMAT_INSTRUCTIONS = """
CITATION GUIDELINES:

1. In-text citations:
   - Use [Source 1], [Source 2], etc.
   - Place immediately after the claim
   - Example: "An arithmetic progression is a sequence of numbers [Source 1]."

2. Source list format:
   - Source 1: NCERT Mathematics, Class 10, Chapter 5 (Arithmetic Progressions), Page 95
   - Source 2: NCERT Mathematics, Class 10, Chapter 5 (Arithmetic Progressions), Page 96

3. Multiple sources for one claim:
   - Use [Source 1, Source 2]
   - Example: "The nth term formula is an = a + (n-1)d [Source 1, Source 2]."

4. When NOT to cite:
   - Never cite if claim is not in the context
   - Instead, omit the claim or say "I don't know"
"""


# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

def format_context_for_prompt(retrieved_chunks: List[Any]) -> str:
    """
    Format retrieved chunks as context for the prompt.
    
    Args:
        retrieved_chunks: List of RetrievalResult objects
        
    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return "No context available."
    
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Extract metadata
        meta = chunk.metadata
        textbook = f"NCERT {meta.get('subject', 'Unknown').title()}"
        class_num = meta.get('class_number', 'Unknown')
        chapter_num = meta.get('chapter_number', 'Unknown')
        chapter_title = meta.get('chapter_title', 'Unknown')
        pages = meta.get('page_numbers', [])
        page_str = f"Page {pages[0]}" if pages else "Page Unknown"
        chunk_type = meta.get('chunk_type', 'content')
        
        # Format source header
        source_header = f"Source {i}: {textbook}, Class {class_num}, Chapter {chapter_num} ({chapter_title}), {page_str}"
        source_type = f"Type: {chunk_type.upper()}"
        
        # Format content
        content = chunk.content.strip()
        
        # Combine
        context_parts.append(f"{source_header}\n{source_type}\n\n{content}\n")
    
    return "\n" + "="*70 + "\n".join(context_parts) + "="*70


# ============================================================================
# PROMPT BUILDERS
# ============================================================================

@dataclass
class PromptTemplate:
    """Container for a complete prompt."""
    system_prompt: str
    user_prompt: str
    context: str
    
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI message format."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{self.context}\n\nQUESTION:\n{self.user_prompt}\n\nANSWER:"}
        ]


class SystemPromptBuilder:
    """Builder for customized system prompts."""
    
    def __init__(self):
        self.base_prompt = STRICT_RAG_SYSTEM_PROMPT
        self.class_number = 10
        self.additional_rules = []
    
    def set_class_number(self, class_number: int) -> 'SystemPromptBuilder':
        """Set the student's class/grade level."""
        self.class_number = class_number
        return self
    
    def add_rule(self, rule: str) -> 'SystemPromptBuilder':
        """Add an additional rule."""
        self.additional_rules.append(rule)
        return self
    
    def for_definition_query(self) -> 'SystemPromptBuilder':
        """Customize for definition queries."""
        self.add_rule(
            "DEFINITION QUERIES: Provide the exact definition from the textbook. "
            "Include the definition number if present (e.g., Definition 5.1)."
        )
        return self
    
    def for_example_query(self) -> 'SystemPromptBuilder':
        """Customize for example queries."""
        self.add_rule(
            "EXAMPLE QUERIES: Show the complete example including the question and solution. "
            "Preserve all steps and calculations exactly as in the textbook."
        )
        return self
    
    def for_problem_solving_query(self) -> 'SystemPromptBuilder':
        """Customize for problem-solving queries."""
        self.add_rule(
            "PROBLEM-SOLVING QUERIES: Guide the student step-by-step. "
            "Reference relevant examples from the textbook. "
            "Do NOT solve the problem directly unless an example shows the exact method."
        )
        return self
    
    def build(self) -> str:
        """Build the final system prompt."""
        prompt = self.base_prompt.format(class_number=self.class_number)
        
        if self.additional_rules:
            prompt += "\n\nADDITIONAL RULES:\n"
            for i, rule in enumerate(self.additional_rules, 1):
                prompt += f"{i}. {rule}\n"
        
        prompt += CITATION_FORMAT_INSTRUCTIONS
        
        return prompt


# ============================================================================
# SPECIALIZED PROMPTS
# ============================================================================

def create_strict_rag_prompt(
    query: str,
    retrieved_chunks: List[Any],
    class_number: int = 10,
    prompt_type: PromptType = PromptType.STRICT_RAG
) -> PromptTemplate:
    """
    Create a strict RAG prompt with context.
    
    Args:
        query: User's question
        retrieved_chunks: Retrieved context chunks
        class_number: Student's class/grade
        prompt_type: Type of prompt to generate
        
    Returns:
        PromptTemplate ready for LLM
    """
    # Build system prompt
    builder = SystemPromptBuilder().set_class_number(class_number)
    
    if prompt_type == PromptType.DEFINITION:
        builder.for_definition_query()
    elif prompt_type == PromptType.EXAMPLE:
        builder.for_example_query()
    elif prompt_type == PromptType.PROBLEM_SOLVING:
        builder.for_problem_solving_query()
    
    system_prompt = builder.build()
    
    # Format context
    context = format_context_for_prompt(retrieved_chunks)
    
    # User prompt is just the query
    user_prompt = query
    
    return PromptTemplate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context=context
    )


# ============================================================================
# FALLBACK RESPONSES
# ============================================================================

INSUFFICIENT_CONTEXT_RESPONSE = "I don't know based on NCERT textbooks."

LOW_CONFIDENCE_RESPONSE_TEMPLATE = """I found some information, but I'm not confident it fully answers your question.

{partial_answer}

**Note:** This information may be incomplete. Please refer to your NCERT textbook for complete details.

{sources}"""


def create_low_confidence_response(partial_answer: str, sources: str) -> str:
    """Create a response for low-confidence retrievals."""
    return LOW_CONFIDENCE_RESPONSE_TEMPLATE.format(
        partial_answer=partial_answer,
        sources=sources
    )


# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

def detect_query_language(query: str) -> str:
    """
    Detect the language of the query.
    
    Args:
        query: User's question
        
    Returns:
        Language code: 'eng', 'hin', 'san', or 'unknown'
    """
    # Simple heuristic: check for Devanagari script
    has_devanagari = any('\u0900' <= char <= '\u097F' for char in query)
    
    if has_devanagari:
        # Could be Hindi or Sanskrit
        # For now, default to Hindi (more common in queries)
        return 'hin'
    else:
        # Assume English
        return 'eng'


def create_language_instruction(detected_language: str) -> str:
    """
    Create instruction for answering in detected language.
    
    Args:
        detected_language: Language code
        
    Returns:
        Instruction string
    """
    language_map = {
        'eng': 'English',
        'hin': 'Hindi',
        'san': 'Sanskrit'
    }
    
    language_name = language_map.get(detected_language, 'the query language')
    
    return f"\n\nIMPORTANT: The query is in {language_name}. You MUST answer in {language_name}."


# ============================================================================
# QUERY INTENT CLASSIFICATION
# ============================================================================

def classify_query_intent(query: str) -> PromptType:
    """
    Classify the intent of the query.
    
    Args:
        query: User's question
        
    Returns:
        PromptType indicating query intent
    """
    query_lower = query.lower()
    
    # Definition queries
    if any(word in query_lower for word in ['what is', 'define', 'definition of', 'meaning of']):
        return PromptType.DEFINITION
    
    # Example queries
    elif any(word in query_lower for word in ['example', 'show me', 'demonstrate', 'illustrate']):
        return PromptType.EXAMPLE
    
    # Problem-solving queries
    elif any(word in query_lower for word in ['how to', 'solve', 'find', 'calculate', 'compute']):
        return PromptType.PROBLEM_SOLVING
    
    # Default to strict RAG
    else:
        return PromptType.STRICT_RAG
