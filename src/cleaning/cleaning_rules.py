"""
Cleaning Rules and Heuristics for NCERT Content
================================================

Comprehensive documentation of all cleaning rules, heuristics, and strategies
used in the text cleaning and structure recovery pipeline.
"""

# =============================================================================
# TEXT CLEANING RULES
# =============================================================================

CLEANING_RULES = {
    "unicode_normalization": {
        "description": "Normalize Unicode characters to consistent representation",
        "method": "NFC (Canonical Decomposition + Canonical Composition)",
        "rationale": "Ensures consistent representation of accented characters",
        "examples": [
            ("Café (decomposed)", "Café (composed)"),
            ("naïve (multiple forms)", "naïve (single form)")
        ]
    },
    
    "invisible_characters": {
        "description": "Remove byte order marks and zero-width characters",
        "patterns": {
            "\\ufeff": "BOM (Byte Order Mark)",
            "\\u200b": "Zero-width space",
            "\\u200c": "Zero-width non-joiner",
            "\\u200d": "Zero-width joiner",
            "\\xa0": "Non-breaking space -> regular space"
        },
        "rationale": "These characters are invisible but interfere with text processing"
    },
    
    "spaced_words": {
        "description": "Fix OCR error where letters are spaced out",
        "pattern": r'\b([A-Z])\s+([a-z])\s+([a-z])',
        "examples": [
            ("M a t h e m a t i c s", "Mathematics"),
            ("C h e m i s t r y", "Chemistry"),
            ("A r i t h m e t i c", "Arithmetic")
        ],
        "rationale": "Common OCR error in book titles and headings"
    },
    
    "hyphenation": {
        "description": "Rejoin words split across lines with hyphenation",
        "pattern": r'(\w+)-\s*\n\s*(\w+)',
        "examples": [
            ("mathe-\nmatics", "mathematics"),
            ("develop-\nment", "development"),
            ("geo-\nmetry", "geometry")
        ],
        "rationale": "Books use hyphenation for line breaks; should be removed"
    },
    
    "punctuation_spacing": {
        "description": "Fix spacing around punctuation marks",
        "rules": [
            {
                "pattern": r'\s+([.,;:!?])',
                "replacement": r'\1',
                "example": ("word .", "word.")
            },
            {
                "pattern": r'([.!?])([A-Z])',
                "replacement": r'\1 \2',
                "example": ("word.Next", "word. Next")
            }
        ],
        "rationale": "OCR often adds incorrect spaces before punctuation"
    },
    
    "character_confusion": {
        "description": "Fix common OCR character recognition errors",
        "confusions": {
            "1 vs l": {
                "pattern": r'\b1([a-z]{2,})',
                "fix": r'l\1',
                "examples": [("1ike", "like"), ("1earn", "learn")]
            },
            "0 vs O": {
                "pattern": r'\b0ne\b',
                "fix": "one",
                "examples": [("0ne", "one"), ("0nly", "only")]
            },
            "I vs 1": {
                "pattern": r'\bI(\d)',
                "fix": r'1\1',
                "examples": [("I5", "15"), ("I0", "10")]
            }
        },
        "rationale": "OCR frequently confuses similar-looking characters"
    },
    
    "line_break_repair": {
        "description": "Fix sentences broken across lines inappropriately",
        "conditions": [
            "Current line doesn't end with sentence-ending punctuation",
            "Current line doesn't end with colon (section header)",
            "Next line starts with lowercase letter",
            "Neither line is a list item"
        ],
        "examples": [
            (
                "The equation represents\na linear relationship",
                "The equation represents a linear relationship"
            ),
            (
                "This is important\nbecause it shows",
                "This is important because it shows"
            )
        ],
        "rationale": "OCR treats all line breaks as paragraph breaks"
    },
    
    "noise_patterns": {
        "description": "Remove common noise from OCR output",
        "patterns": [
            {
                "pattern": r'^Page\s+\d+\s*$',
                "type": "Page number",
                "action": "Remove"
            },
            {
                "pattern": r'^\d+\s*$',
                "type": "Standalone number",
                "action": "Remove (likely page number)"
            },
            {
                "pattern": r'^NCERT\s*$',
                "type": "Header text",
                "action": "Remove"
            },
            {
                "pattern": r'^[_\-=]{3,}$',
                "type": "Decorative lines",
                "action": "Remove"
            },
            {
                "pattern": r'^\.{3,}$',
                "type": "Dot leaders",
                "action": "Remove"
            }
        ],
        "rationale": "These patterns are formatting artifacts, not content"
    },
    
    "duplicate_removal": {
        "description": "Remove exact and near-duplicate lines",
        "strategies": [
            {
                "type": "Exact duplicates",
                "method": "Set-based deduplication",
                "rationale": "Chapter titles, headers often appear multiple times"
            },
            {
                "type": "Near duplicates (aggressive mode)",
                "method": "Character-based similarity >= 90%",
                "rationale": "OCR may create slightly different versions of same line"
            }
        ],
        "examples": [
            ("Chapter 5: Algebra\nChapter 5: Algebra", "Chapter 5: Algebra"),
            ("Exercise 3.4\nExercise 3. 4", "Exercise 3.4")
        ]
    },
    
    "bullet_normalization": {
        "description": "Normalize various bullet point styles",
        "mappings": {
            "*": "•",
            "▪": "•",
            "▫": "•",
            "■": "•",
            "●": "•",
            "○": "•",
            "→": "•",
            "1)": "1.",
            "2)": "2."
        },
        "rationale": "Consistency in list formatting"
    },
    
    "whitespace_normalization": {
        "description": "Normalize all whitespace to consistent format",
        "rules": [
            "Replace tabs with 4 spaces",
            "Replace 3+ newlines with 2 newlines",
            "Remove trailing whitespace from lines",
            "Remove leading/trailing whitespace from document"
        ],
        "rationale": "Consistent whitespace aids parsing and chunking"
    }
}


# =============================================================================
# STRUCTURE RECOVERY HEURISTICS
# =============================================================================

STRUCTURE_HEURISTICS = {
    "definitions": {
        "markers": [
            "Definition",
            "Def.",
            "is defined as",
            "we define"
        ],
        "detection_rules": [
            "Line starts with 'Definition' followed by optional number",
            "Sentence contains 'is defined as' or 'we define'",
            "Usually followed by formal mathematical notation"
        ],
        "preservation_strategy": "Keep as single block, don't split in chunking",
        "importance": "CRITICAL - Core concept introduction"
    },
    
    "theorems": {
        "markers": [
            "Theorem",
            "THM"
        ],
        "structure": "Theorem statement → Proof → Q.E.D.",
        "detection_rules": [
            "Line starts with 'Theorem' followed by number",
            "Usually followed by colon and formal statement",
            "May be followed by 'Proof:' section"
        ],
        "preservation_strategy": "Keep theorem and proof together",
        "importance": "CRITICAL - Core mathematical results"
    },
    
    "examples": {
        "markers": [
            "Example",
            "Ex.",
            "E"
        ],
        "structure": "Example statement → Solution/Explanation",
        "numbering": "Format: Example X.Y where X is chapter, Y is example number",
        "detection_rules": [
            "Line starts with 'Example' followed by number",
            "Usually numbered: Example 1, Example 5.3, etc.",
            "May be followed by 'Solution:' or direct explanation"
        ],
        "preservation_strategy": "Keep example and solution together",
        "importance": "HIGH - Demonstrates concept application"
    },
    
    "exercises": {
        "markers": [
            "Exercise",
            "EXERCISE"
        ],
        "structure": "Exercise header → Numbered questions",
        "detection_rules": [
            "Line starts with 'Exercise' followed by number",
            "Followed by numbered questions (1., 2., 3., etc.)",
            "Questions may have sub-parts (a), (b), (c)"
        ],
        "preservation_strategy": "Keep exercise as unit, can split individual questions",
        "importance": "HIGH - Practice problems"
    },
    
    "questions": {
        "markers": [
            "Numbered: 1., 2., 3.",
            "Lettered: (a), (b), (c)",
            "Q. prefix"
        ],
        "detection_rules": [
            "Line starts with number followed by period and space",
            "Line starts with letter in parentheses",
            "Line starts with 'Q.' followed by number"
        ],
        "preservation_strategy": "Individual questions can be separate chunks",
        "importance": "MEDIUM - Assessment items"
    },
    
    "proofs": {
        "markers": [
            "Proof:",
            "Solution:"
        ],
        "end_markers": [
            "Q.E.D.",
            "∎",
            "■"
        ],
        "detection_rules": [
            "Line starts with 'Proof:' or 'Solution:'",
            "Ends with Q.E.D. or proof symbol",
            "Usually follows a theorem or problem statement"
        ],
        "preservation_strategy": "Keep complete proof as single unit",
        "importance": "HIGH - Logical reasoning"
    },
    
    "notes_remarks": {
        "markers": [
            "Note:",
            "N.B.:",
            "Remark:",
            "Observation:"
        ],
        "detection_rules": [
            "Line starts with 'Note:' or similar",
            "Usually short, 1-3 sentences",
            "Provides clarification or additional insight"
        ],
        "preservation_strategy": "Can be separate chunk or attached to related concept",
        "importance": "MEDIUM - Supplementary information"
    },
    
    "summaries": {
        "markers": [
            "Summary",
            "Key Points",
            "Important Points"
        ],
        "detection_rules": [
            "Line starts with 'Summary' or 'Key Points'",
            "Usually at end of chapter or section",
            "Contains bulleted list of main concepts"
        ],
        "preservation_strategy": "Keep as complete summary chunk",
        "importance": "HIGH - Review material"
    },
    
    "formulas": {
        "markers": [
            "Formula",
            "Equation in brackets",
            "Mathematical notation"
        ],
        "detection_rules": [
            "Line starts with 'Formula'",
            "Line contains equation with = sign",
            "May be numbered: (1), (2), etc."
        ],
        "preservation_strategy": "Keep formula with surrounding explanation",
        "importance": "CRITICAL - Mathematical relationships"
    },
    
    "activities": {
        "markers": [
            "Activity",
            "ACTIVITY"
        ],
        "detection_rules": [
            "Line starts with 'Activity' followed by number",
            "Contains instructions for hands-on learning",
            "Usually multi-step procedure"
        ],
        "preservation_strategy": "Keep complete activity as single unit",
        "importance": "MEDIUM - Experiential learning"
    },
    
    "explanations": {
        "markers": [
            "Implicit - detected by keywords",
            "therefore", "hence", "thus",
            "because", "since",
            "this means", "in other words"
        ],
        "detection_rules": [
            "Contains explanation keywords",
            "Connects concepts with logical relationships",
            "Usually follows definition or example"
        ],
        "preservation_strategy": "Keep with related concept",
        "importance": "HIGH - Understanding building"
    }
}


# =============================================================================
# QUALITY CONTROL RULES
# =============================================================================

QUALITY_RULES = {
    "minimum_sentence_length": {
        "value": 10,
        "rationale": "Sentences shorter than 10 characters are likely OCR artifacts",
        "action": "Flag for review or remove"
    },
    
    "maximum_consecutive_caps": {
        "value": 50,
        "rationale": "More than 50 consecutive capitals suggests OCR error",
        "action": "Check for proper case normalization needed"
    },
    
    "equation_preservation": {
        "rule": "Never split equations across chunks",
        "detection": "Look for =, +, -, *, /, mathematical symbols",
        "handling": "Keep equation with surrounding explanation"
    },
    
    "structure_integrity": {
        "rule": "Never split definition/theorem/example from its explanation",
        "detection": "Structure markers + content analysis",
        "handling": "Treat as atomic unit for chunking"
    },
    
    "list_preservation": {
        "rule": "Keep numbered/bulleted lists together",
        "detection": "Sequential numbering or bullet points",
        "handling": "Either keep all items or split at top level only"
    },
    
    "confidence_thresholds": {
        "explicit_structures": 1.0,
        "implicit_structures": 0.7,
        "minimum_for_use": 0.5,
        "rationale": "Lower confidence structures need verification"
    }
}


# =============================================================================
# CONTEXT-SPECIFIC RULES
# =============================================================================

CONTENT_TYPE_RULES = {
    "text": {
        "cleaning_aggressiveness": "standard",
        "sentence_repair": "enabled",
        "duplicate_removal": "enabled"
    },
    
    "equation": {
        "cleaning_aggressiveness": "conservative",
        "sentence_repair": "disabled",
        "preserve_symbols": True,
        "preserve_spacing": True
    },
    
    "exercise": {
        "cleaning_aggressiveness": "standard",
        "preserve_numbering": "critical",
        "sentence_repair": "enabled"
    },
    
    "table": {
        "cleaning_aggressiveness": "minimal",
        "preserve_structure": "critical",
        "sentence_repair": "disabled"
    },
    
    "diagram": {
        "cleaning_aggressiveness": "minimal",
        "note": "Text from diagrams is usually unreliable",
        "action": "Flag for manual review"
    }
}


# =============================================================================
# PRIORITY HIERARCHY
# =============================================================================

CLEANING_PRIORITY = """
Priority order for cleaning operations:

1. PRESERVE (Highest Priority)
   - Educational structure markers
   - Mathematical equations and formulas
   - Numbered lists and sequential information
   - Definition and theorem statements

2. REPAIR
   - Broken sentences
   - Hyphenated words
   - Character confusion errors
   - Punctuation spacing

3. NORMALIZE
   - Unicode characters
   - Whitespace
   - Bullet points
   - Quote marks

4. REMOVE (Lowest Priority)
   - Noise patterns
   - Duplicates
   - Excessive whitespace
   - Headers/footers

Rationale: Better to keep questionable content than lose critical information.
In case of conflict, always err on the side of preservation.
"""


# =============================================================================
# SPECIAL CASES
# =============================================================================

SPECIAL_CASES = {
    "hindi_content": {
        "issues": [
            "Devanagari script spacing may differ",
            "Combined characters (conjuncts) need special handling",
            "Number representation may mix Hindi and Arabic numerals"
        ],
        "handling": "Unicode normalization handles most issues"
    },
    
    "mathematical_notation": {
        "issues": [
            "Greek letters may be recognized incorrectly",
            "Superscripts/subscripts may be inline",
            "Matrix notation may be mangled"
        ],
        "handling": "Conservative cleaning, preserve all mathematical symbols"
    },
    
    "chemistry_formulas": {
        "issues": [
            "Subscripts are critical (H2O not H2 O)",
            "Charges (+, -) must be preserved",
            "Arrows (→, ⇌) are semantic"
        ],
        "handling": "No whitespace normalization within formulas"
    },
    
    "multiple_choice": {
        "structure": "(a) option 1\n(b) option 2\n(c) option 3\n(d) option 4",
        "preservation": "Keep all options together with question",
        "handling": "Don't remove parentheses around letters"
    }
}


# =============================================================================
# DECISION MATRIX
# =============================================================================

DECISION_MATRIX = """
When encountering ambiguous content:

Q: Should this line be removed?
   ├─ Is it a known noise pattern? → YES, remove
   ├─ Is it very short (<3 chars)? → YES, remove
   ├─ Does it contain educational content? → NO, keep
   └─ Uncertain? → NO, keep (preserve by default)

Q: Should these lines be joined?
   ├─ Both part of same sentence? → YES, join
   ├─ First ends with punctuation? → NO, don't join
   ├─ Second starts with capital? → NO, don't join
   ├─ Part of numbered list? → NO, don't join
   └─ Uncertain? → NO, don't join (preserve breaks)

Q: Is this a structure marker?
   ├─ Matches known pattern? → YES, mark structure
   ├─ Contains keywords? → MAYBE, mark with low confidence
   └─ Uncertain? → NO, treat as regular text

Q: Should this be normalized?
   ├─ Unicode issue? → YES, normalize
   ├─ Whitespace issue? → YES, normalize
   ├─ Stylistic choice? → NO, preserve
   └─ Uncertain? → NO, preserve
"""

if __name__ == "__main__":
    print("NCERT Text Cleaning Rules and Heuristics")
    print("=" * 60)
    print("\nThis module documents all rules used in text cleaning.")
    print("Import this module to access rule definitions programmatically.")
    print("\nKey rule categories:")
    print("- CLEANING_RULES: Text normalization and repair")
    print("- STRUCTURE_HEURISTICS: Educational structure detection")
    print("- QUALITY_RULES: Quality control thresholds")
    print("- CONTENT_TYPE_RULES: Content-specific handling")
    print("- SPECIAL_CASES: Edge cases and exceptions")
