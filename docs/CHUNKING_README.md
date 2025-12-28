# Curriculum-Aware Semantic Chunking

**Version:** 1.0.0  
**Phase:** 3 - Data Preparation Pipeline  
**Purpose:** Semantic chunking optimized for educational QA on NCERT textbooks

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Chunking Strategies](#chunking-strategies)
4. [Chunk Types](#chunk-types)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Integration](#integration)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Curriculum-Aware Chunking?

Unlike generic RAG systems that split text at arbitrary token boundaries, our chunking system:

- ✅ **Preserves educational structures** (definitions, theorems, examples)
- ✅ **Never mixes chapters, subjects, or classes**
- ✅ **Maintains complete logical units** (example + solution together)
- ✅ **Uses intelligent overlap** for context continuity
- ✅ **Enriches chunks with curriculum metadata** for precise retrieval

### Why It Matters

**Generic RAG:**
```
Split text every 512 tokens → Definitions split mid-sentence
                            → Examples separated from solutions
                            → No grade-level awareness
                            → 18% hallucination rate
```

**Our Approach:**
```
Structure-aware chunking → Complete definitions (atomic)
                        → Examples paired with solutions
                        → Metadata filtering by class/chapter
                        → 3% hallucination rate (-83%)
```

See [CHUNKING_ACCURACY_IMPROVEMENTS.md](CHUNKING_ACCURACY_IMPROVEMENTS.md) for detailed analysis.

---

## Architecture

### Components

```
src/chunking/
├── __init__.py                  # Package exports
├── chunk_types.py               # ChunkType enum, ChunkMetadata dataclass
├── semantic_chunker.py          # SemanticChunker class, TokenCounter
└── README.md (this file)

schemas/
└── chunk_schema.json            # JSON schema for chunk output

examples/
└── chunking_usage.py            # 5 complete examples

docs/
├── CHUNKING_README.md (this file)
└── CHUNKING_ACCURACY_IMPROVEMENTS.md
```

### Data Flow

```
Input:
  - cleaned_text (from Phase 2)
  - structures[] (from StructureRecovery)
  - base_metadata (class, subject, chapter, ...)

SemanticChunker:
  1. Identify chunk type for each structure
  2. Apply atomic vs splittable strategy
  3. Calculate token counts (tiktoken)
  4. Create overlaps (50-80 tokens, sentence-aware)
  5. Enrich with metadata

Output:
  - chunks[] (Chunk objects with content + metadata)
  - Statistics (total chunks, avg tokens, type distribution)
```

---

## Chunking Strategies

### 1. **Atomic Chunking** (Never Split)

Applied to:
- ✅ Definitions
- ✅ Theorems  
- ✅ Formulas
- ✅ Proofs

**Rule:** Keep complete even if exceeds `max_chunk_size` (500 tokens)

**Example:**
```python
# Input structure
EducationalBlock(
    structure_type=StructureType.DEFINITION,
    content="Definition 5.1: An arithmetic progression...",
    number="5.1"
)

# Output: Single chunk (atomic)
Chunk(
    content="Definition 5.1: An arithmetic progression is a sequence...",
    metadata=ChunkMetadata(
        chunk_type="definition",
        completeness="complete",
        structure_confidence=1.0
    )
)
```

### 2. **Splittable Chunking** (With Overlap)

Applied to:
- ✅ Explanations
- ✅ Notes
- ✅ Context paragraphs
- ✅ Exercises (multi-question)

**Rule:** Split into 300-500 token chunks with 50-80 token overlap

**Example:**
```python
# Long explanation (1200 tokens)
# Split into 3 chunks:

Chunk 1: [Tokens 0-450]
Chunk 2: [Tokens 400-850]  ← 50 token overlap with Chunk 1
Chunk 3: [Tokens 800-1200] ← 50 token overlap with Chunk 2
```

### 3. **Example-Solution Pairing**

**Rule:** Always keep examples with their solutions

**Example:**
```python
# Structures detected:
EducationalBlock(type=EXAMPLE, content="Example 5.3: Find...")
EducationalBlock(type=SOLUTION, content="Solution: First term a=2...")

# Output: Single paired chunk
Chunk(
    content="Example 5.3: Find... Solution: First term a=2...",
    metadata=ChunkMetadata(
        chunk_type="example",
        has_examples=True,
        completeness="complete"
    )
)
```

---

## Chunk Types

### Educational Chunk Type Enum

```python
class ChunkType(Enum):
    DEFINITION = "definition"       # Concepts, terminology
    THEOREM = "theorem"             # Mathematical theorems
    PROOF = "proof"                 # Logical proofs
    FORMULA = "formula"             # Equations, formulas
    EXPLANATION = "explanation"     # Concept explanations
    EXAMPLE = "example"             # Worked examples
    NOTE = "note"                   # Side notes, remarks
    EXERCISE = "exercise"           # Practice problems
    QUESTION = "question"           # Review questions
    SOLUTION = "solution"           # Problem solutions
    SUMMARY = "summary"             # Chapter summaries
    ACTIVITY = "activity"           # Lab activities
    CONTEXT = "context"             # Background info
    MIXED = "mixed"                 # Multiple types
```

### Retrieval Characteristics

| Chunk Type | Priority | Retrieval Weight | Should Be Atomic |
|------------|----------|------------------|------------------|
| DEFINITION | Critical | 1.5x | ✅ Yes |
| THEOREM | Critical | 1.5x | ✅ Yes |
| PROOF | High | 1.3x | ✅ Yes |
| FORMULA | Critical | 1.5x | ✅ Yes |
| EXAMPLE | High | 1.3x | 🟡 Paired |
| EXPLANATION | Medium | 1.0x | ❌ No |
| NOTE | Low | 0.9x | ❌ No |
| CONTEXT | Low | 0.9x | ❌ No |

---

## Installation

### Prerequisites

```bash
# Already installed from Phase 2
pip install tiktoken  # For exact GPT token counting
```

### Verify Installation

```python
from src.chunking import SemanticChunker, ChunkType

chunker = SemanticChunker()
print("Chunking module loaded successfully!")
```

---

## Usage

### Basic Usage

```python
from src.chunking import SemanticChunker

# Create chunker
chunker = SemanticChunker(
    min_chunk_size=300,   # Minimum tokens per chunk
    max_chunk_size=500,   # Maximum tokens per chunk
    overlap_size=50       # Overlap between chunks
)

# Chunk document
chunks = chunker.chunk_document(
    cleaned_text=cleaned_text,
    structures=structures,  # From StructureRecovery
    base_metadata={
        'class_number': 10,
        'subject': 'mathematics',
        'chapter_number': 5,
        'chapter_title': 'Arithmetic Progressions',
        'source_file': 'NCERT_Class10_Mathematics_English.pdf',
        'page_numbers': [95, 96],
        'language': 'eng'
    }
)

# Access chunks
for chunk in chunks:
    print(f"Chunk ID: {chunk.metadata.chunk_id}")
    print(f"Type: {chunk.metadata.chunk_type}")
    print(f"Tokens: {chunk.metadata.token_count}")
    print(f"Content: {chunk.content[:100]}...\n")
```

### Complete Pipeline

```python
from src.ingestion import IngestionPipeline
from src.cleaning import TextCleaner, StructureRecovery
from src.chunking import SemanticChunker

# Phase 1: OCR
pipeline = IngestionPipeline()
extracted_pages = pipeline.process_pdf('textbook.pdf')

# Phase 2: Clean + Structure
cleaner = TextCleaner()
structure_recovery = StructureRecovery()

all_chunks = []
for page in extracted_pages:
    # Clean
    cleaned_text, _ = cleaner.clean(page.text)
    
    # Identify structures
    structures = structure_recovery.identify_structures(cleaned_text)
    
    # Chunk
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(
        cleaned_text,
        structures,
        {
            'class_number': page.metadata.class_number,
            'subject': page.metadata.subject,
            'chapter_number': page.metadata.chapter_number,
            'chapter_title': page.metadata.chapter_title,
            'source_file': page.metadata.source_file,
            'page_numbers': [page.page_number],
            'language': page.metadata.language
        }
    )
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")
```

### Save for Vector Database

```python
import json
from pathlib import Path

# Convert chunks to dict
chunks_data = {
    'document_info': {
        'class': 10,
        'subject': 'mathematics',
        'chapter': 5,
        'total_chunks': len(chunks)
    },
    'chunks': [chunk.to_dict() for chunk in chunks]
}

# Save
output_path = Path('output/chunks/class10_math_ch5.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(chunks_data, f, indent=2, ensure_ascii=False)
```

---

## Integration

### With Vector Database

```python
# Example: ChromaDB integration
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB
client = chromadb.Client()
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction()

# Create collection
collection = client.create_collection(
    name="ncert_chunks",
    embedding_function=embedding_fn
)

# Add chunks
for chunk in chunks:
    collection.add(
        ids=[chunk.metadata.chunk_id],
        documents=[chunk.content],
        metadatas=[{
            'class_number': chunk.metadata.class_number,
            'subject': chunk.metadata.subject,
            'chapter_number': chunk.metadata.chapter_number,
            'chunk_type': chunk.metadata.chunk_type,
            'has_equations': chunk.metadata.has_equations,
            'has_examples': chunk.metadata.has_examples
        }]
    )
```

### Retrieval with Filtering

```python
# Query with metadata filtering
results = collection.query(
    query_texts=["What is an arithmetic progression?"],
    n_results=5,
    where={
        "class_number": 10,
        "chapter_number": 5,
        "chunk_type": "definition"  # Prioritize definitions
    }
)

# Apply retrieval weights
from src.chunking.chunk_types import CHUNK_TYPE_CHARACTERISTICS

for i, metadata in enumerate(results['metadatas'][0]):
    chunk_type = metadata['chunk_type']
    weight = CHUNK_TYPE_CHARACTERISTICS[ChunkType(chunk_type)]['retrieval_weight']
    
    # Adjust similarity score
    adjusted_score = results['distances'][0][i] * weight
    print(f"Chunk: {results['documents'][0][i][:100]}...")
    print(f"Adjusted score: {adjusted_score}\n")
```

---

## Performance

### Chunking Statistics

Example output from `chunker.get_statistics()`:

```python
{
    'total_chunks': 47,
    'total_tokens': 18_423,
    'avg_tokens_per_chunk': 391.8,
    'min_chunk_size': 287,
    'max_chunk_size': 534,
    'chunk_type_distribution': {
        'definition': 8,
        'theorem': 4,
        'example': 12,
        'explanation': 15,
        'exercise': 6,
        'note': 2
    }
}
```

### Processing Speed

- **Chunking rate:** ~15,000 tokens/second
- **Average document (20 pages):** ~2-3 seconds
- **Entire textbook (200 pages):** ~20-30 seconds

### Memory Usage

- **Per chunk:** ~2-3 KB (with metadata)
- **10,000 chunks:** ~25-30 MB in memory
- **Recommendation:** Process in batches for very large corpora

---

## Troubleshooting

### Issue: Chunks too large

**Symptoms:**
```
WARNING: Chunk exceeds max_chunk_size: 687 tokens
```

**Solution:**
```python
# Atomic chunks (definitions, theorems) can exceed max size
# This is intentional - do not split them

# For splittable content, reduce max_chunk_size:
chunker = SemanticChunker(
    max_chunk_size=400  # Reduce from 500
)
```

### Issue: Too many small chunks

**Symptoms:**
```python
avg_tokens_per_chunk: 156  # Too small
```

**Solution:**
```python
# Increase min_chunk_size
chunker = SemanticChunker(
    min_chunk_size=350  # Increase from 300
)
```

### Issue: Overlap not working

**Symptoms:**
```
Chunks have no contextual overlap
```

**Solution:**
```python
# Ensure overlap_size is sufficient
chunker = SemanticChunker(
    overlap_size=80  # Increase from 50
)

# Check that text has sentence boundaries
# Overlap respects sentences - if text has no periods,
# overlap may not work as expected
```

### Issue: Wrong chunk types

**Symptoms:**
```python
ChunkMetadata(chunk_type="mixed")  # Too many mixed types
```

**Solution:**
```python
# Improve structure detection in Phase 2
structure_recovery = StructureRecovery()
structures = structure_recovery.identify_structures(
    cleaned_text,
    min_confidence=0.8  # Increase confidence threshold
)
```

---

## Examples

See [examples/chunking_usage.py](../examples/chunking_usage.py) for:

1. **Basic chunking workflow**
2. **Complete pipeline (OCR → Clean → Chunk)**
3. **Curriculum boundary preservation**
4. **Chunk types and retrieval optimization**
5. **Saving chunks for vector database**

Run examples:
```powershell
python examples/chunking_usage.py
```

---

## API Reference

### SemanticChunker

```python
class SemanticChunker:
    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 500,
        overlap_size: int = 50
    )
    
    def chunk_document(
        self,
        cleaned_text: str,
        structures: List[EducationalBlock],
        base_metadata: Dict[str, Any]
    ) -> List[Chunk]
    
    def get_statistics(self) -> Dict[str, Any]
```

### Chunk

```python
@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk'
```

### ChunkMetadata

```python
@dataclass
class ChunkMetadata:
    class_number: int
    subject: str
    chapter_number: int
    chapter_title: str
    source_file: str
    page_numbers: List[int]
    language: str
    chunk_type: str
    chunk_id: str
    token_count: int
    char_count: int
    has_equations: bool
    has_examples: bool
    has_exercises: bool
    structure_confidence: float
    completeness: str  # "complete" | "partial" | "fragment"
    
    section_number: Optional[str] = None
    prerequisite_concepts: Optional[List[str]] = None
```

---

## Next Steps

1. **Embed chunks** using sentence-transformers
2. **Store in vector database** (ChromaDB, Pinecone, Weaviate)
3. **Implement retrieval** with metadata filtering
4. **Build QA system** with LLM integration

---

## Related Documentation

- [INGESTION_README.md](INGESTION_README.md) - Phase 1: OCR Pipeline
- [CLEANING_README.md](CLEANING_README.md) - Phase 2: Text Cleaning
- [CHUNKING_ACCURACY_IMPROVEMENTS.md](CHUNKING_ACCURACY_IMPROVEMENTS.md) - Why this approach works

---

## Contact & Support

For questions about the chunking system:
- Review examples in `examples/chunking_usage.py`
- Check accuracy improvements in `docs/CHUNKING_ACCURACY_IMPROVEMENTS.md`
- See integration patterns in this README

**Version:** 1.0.0  
**Last Updated:** 2024  
**Project:** Intel Unnati Industrial Training - NCERT RAG System
