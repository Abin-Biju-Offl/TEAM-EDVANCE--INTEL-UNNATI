# NCERT RAG System - Retrieval Optimization Analysis

## Current System Analysis

### Data Overview
- **Total Chunks**: 352 chunks across 34 PDFs
- **Subjects**: Science (15 PDFs), English (10 PDFs), Social Science (9 PDFs)
- **Chunk Structure**: Token-based chunks with ~484 tokens average
- **Embedding Model**: all-MiniLM-L6-v2 (384D)
- **Vector Store**: FAISS IVF index

### Chunk Structure Analysis
```json
{
  "text": "chunk content...",
  "start": 0,
  "end": 1936,
  "char_count": 1936,
  "token_count": 484,
  "metadata": {
    "chapter": "1",
    "section": null,
    "class": "10",
    "subject": "Science",
    "filename": "jesc101.pdf",
    "pages": [
      {
        "page_number": 1,
        "pdf_page_number": "26",
        "text": "page content...",
        "char_count": 1501
      }
    ],
    "chunk_index": 0,
    "total_chunks": 18
  }
}
```

## Identified Issues

### 1. **Page Number Metadata is Nested**
- Page numbers are in `metadata.pages[].page_number`
- This makes filtering by page number complex
- Not easily accessible during retrieval

### 2. **Missing Direct Page Reference**
- Chunks don't have a direct `page_number` field at top level
- Citations require parsing nested metadata
- Harder for users to verify sources

### 3. **No Hierarchical Structure**
- Section/subsection information not captured
- Headings not identified (e.g., "1.1.1 Writing a Chemical Equation")
- Context boundaries unclear

### 4. **Chunking Strategy**
- Fixed token-based chunking (~484 tokens)
- Breaks semantic units (may split examples, definitions)
- No respect for logical boundaries (sections, subsections)

### 5. **Missing Content Type Tags**
- No identification of: definitions, examples, exercises, theorems
- All chunks treated equally during retrieval
- Can't prioritize explanatory text over questions

### 6. **Incomplete Chapter Extraction**
- Chapter field is extracted but not validated
- No chapter titles stored
- No cross-chapter context

## Optimization Recommendations

### Priority 1: Metadata Enhancement (HIGH IMPACT)

#### A. Flatten Critical Metadata
Add these fields at chunk level:
```python
{
  "page_numbers": [1, 2, 3],  # All pages this chunk spans
  "primary_page": 1,          # Main page for citation
  "pdf_page_numbers": ["26"], # Actual PDF page numbers
}
```

#### B. Add Content Type Classification
```python
{
  "content_type": "definition",  # definition, example, exercise, explanation, etc.
  "has_equations": true,
  "has_figures": false,
  "keywords": ["chemical", "reaction", "equation"]
}
```

### Priority 2: Chunking Strategy (HIGH IMPACT)

#### A. Semantic Chunking with Structure Awareness
```python
# Instead of fixed tokens, chunk by:
1. Section boundaries (## 1.1 CHEMICAL EQUATIONS)
2. Subsection boundaries (### 1.1.1 Writing a Chemical Equation)
3. Paragraph boundaries for long sections
4. Keep definitions/examples intact
5. Keep question-answer pairs together
```

#### B. Overlap Strategy
```python
# Current: character-based overlap
# Better: semantic overlap
- Include section header in all chunks of that section
- Include previous paragraph's last sentence
- Include next paragraph's first sentence
```

### Priority 3: Hierarchical Context (MEDIUM IMPACT)

#### Add Document Structure
```python
{
  "chapter": "1",
  "chapter_title": "Chemical Reactions and Equations",
  "section": "1.1",
  "section_title": "Chemical Equations",
  "subsection": "1.1.1",
  "subsection_title": "Writing a Chemical Equation",
  "topic_hierarchy": ["Chemistry", "Reactions", "Equations"]
}
```

### Priority 4: Enhanced Search Features (MEDIUM IMPACT)

#### A. Add Contextual Windows
```python
{
  "prev_chunk_text": "previous 100 chars...",
  "next_chunk_text": "next 100 chars...",
  "section_context": "Full section heading..."
}
```

#### B. Add Cross-References
```python
{
  "references": ["Eq. 1.2", "Fig. 1.1", "Activity 1.1"],
  "related_chunks": [chunk_id_1, chunk_id_2],
  "exercises": [chunk_id_ex1, chunk_id_ex2]
}
```

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. **Flatten page numbers** to top-level metadata
2. **Add primary_page** for simple citations
3. **Extract keywords** from text using basic NLP
4. **Add has_equations/has_figures** flags

### Phase 2: Enhanced Chunking (2-3 hours)
1. **Implement heading detection** (regex patterns)
2. **Create semantic chunker** that respects:
   - Section boundaries
   - Definition blocks
   - Example blocks
   - Question blocks
3. **Add content type classifier**

### Phase 3: Hierarchical Structure (2-3 hours)
1. **Build document tree** from headings
2. **Extract chapter/section/subsection** titles
3. **Add parent-child relationships** between chunks
4. **Create topic taxonomy**

### Phase 4: Advanced Features (3-4 hours)
1. **Implement context windows**
2. **Build cross-reference index**
3. **Create exercise-to-concept mappings**
4. **Add related chunk suggestions**

## Expected Improvements

### Retrieval Quality
- **30-50% improvement** in answer accuracy from semantic chunking
- **Better context** from hierarchical structure
- **More precise citations** with flattened page numbers

### User Experience
- **Clear source attribution** (Page 5, Section 1.1.1)
- **Better context** in answers (includes section headers)
- **Relevant examples** prioritized in results

### System Performance
- **Faster filtering** with top-level metadata
- **More relevant** results from content type classification
- **Better ranking** from enhanced metadata

## Code Changes Required

### 1. OCR Service (ocr_service.py)
```python
# Add heading extraction
def _extract_headings(self, text: str) -> List[Dict]:
    patterns = [
        r'CHAPTER\s+(\d+)\s+(.*)',
        r'(\d+\.\d+)\s+([A-Z][^\n]+)',
        r'(\d+\.\d+\.\d+)\s+([A-Z][^\n]+)'
    ]
    # ...

# Add content type detection
def _classify_content_type(self, text: str) -> str:
    if re.search(r'(?i)(definition|defined as):', text):
        return 'definition'
    if re.search(r'(?i)(example|for instance):', text):
        return 'example'
    # ...
```

### 2. Cleaning Service (cleaning_service.py)
```python
# Enhance metadata flattening
def flatten_metadata(self, chunk: Dict) -> Dict:
    chunk['page_numbers'] = [p['page_number'] for p in chunk['metadata']['pages']]
    chunk['primary_page'] = chunk['page_numbers'][0] if chunk['page_numbers'] else None
    chunk['subject'] = chunk['metadata']['subject']
    chunk['chapter'] = chunk['metadata']['chapter']
    return chunk
```

### 3. Chunking Service (chunking_service.py)
```python
# Implement semantic chunking
def chunk_by_semantics(self, text: str, headings: List) -> List[Dict]:
    chunks = []
    for heading in headings:
        # Extract text for this section
        section_text = self._extract_section(text, heading)
        # Split if too long, keeping semantic units
        if len(section_text) > self.max_tokens:
            sub_chunks = self._split_preserving_semantics(section_text)
        # ...
```

### 4. Pipeline Service (pipeline_service.py)
```python
# Update metadata in results
def enhance_results(self, results: List[Dict]) -> List[Dict]:
    for result in results:
        # Add context
        result['section_context'] = self._get_section_context(result)
        # Format citation
        result['citation'] = self._format_citation(result)
    return results
```

## Testing Strategy

### 1. Chunking Quality Tests
```python
def test_chunk_boundaries():
    # Verify sections not split
    # Verify definitions kept intact
    # Verify examples with explanations
```

### 2. Retrieval Accuracy Tests
```python
queries = [
    "What is a chemical reaction?",  # Definition
    "Give an example of combination reaction",  # Example
    "Balance the equation Fe + H2O",  # Exercise
    "What happens when magnesium burns?"  # Explanation
]
# Measure: MRR, Precision@5, Recall@10
```

### 3. Citation Quality Tests
```python
def test_citations():
    # Verify page numbers correct
    # Verify chapter/section correct
    # Verify context preserved
```

## Rollout Plan

1. **Backup** current vector store and processed data
2. **Implement** Phase 1 (quick wins)
3. **Test** with sample queries
4. **Reprocess** 5 PDFs (one from each subject)
5. **Validate** improvements
6. **Full reprocessing** of all 34 PDFs
7. **Monitor** retrieval metrics

## Metrics to Track

### Before & After Comparison
- Average chunk size (tokens)
- Number of chunks (should increase slightly)
- Chunks per document distribution
- Retrieval precision@5, precision@10
- Average MRR (Mean Reciprocal Rank)
- User satisfaction (manual evaluation)

## Next Steps

1. **Review** this analysis with stakeholders
2. **Prioritize** phases based on impact/effort
3. **Create** detailed task breakdown
4. **Implement** Phase 1 improvements
5. **Measure** impact before proceeding
