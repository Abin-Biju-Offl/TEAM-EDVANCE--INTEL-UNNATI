"""
Chunking Service - Phase 3

Semantic chunking with structure-awareness.
"""

import re
from typing import List, Dict
from loguru import logger

from app.core.config import settings


class ChunkingService:
    """Service for semantic chunking of text"""
    
    def __init__(self):
        """Initialize chunking service"""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def chunk_by_structure(self, text: str, structures: List[Dict]) -> List[Dict]:
        """
        Chunk text based on identified structures
        
        Args:
            text: Cleaned text
            structures: List of structure markers
            
        Returns:
            List of chunks with metadata
        """
        if not structures:
            # Fallback to token-based chunking
            return self.chunk_by_tokens(text)
        
        chunks = []
        
        # Add boundaries at start and end
        boundaries = [0] + [s['start'] for s in structures] + [len(text)]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) < 50:  # Skip very small chunks
                continue
            
            # Get structure type if available
            structure_type = None
            if i < len(structures):
                structure_type = structures[i]['type']
            
            chunks.append({
                'text': chunk_text,
                'start': start,
                'end': end,
                'structure_type': structure_type,
                'char_count': len(chunk_text)
            })
        
        return chunks
    
    def enhance_chunk_metadata(self, chunk: Dict) -> Dict:
        """
        Enhance chunk with flattened metadata and content classification
        
        Args:
            chunk: Chunk dictionary with metadata
            
        Returns:
            Enhanced chunk with additional fields
        """
        # Flatten page numbers from nested structure
        if 'metadata' in chunk and 'pages' in chunk['metadata']:
            pages = chunk['metadata']['pages']
            chunk['page_numbers'] = [p.get('page_number') for p in pages if p.get('page_number')]
            chunk['pdf_page_numbers'] = [p.get('pdf_page_number') for p in pages if p.get('pdf_page_number')]
            chunk['primary_page'] = chunk['page_numbers'][0] if chunk['page_numbers'] else None
        else:
            chunk['page_numbers'] = []
            chunk['pdf_page_numbers'] = []
            chunk['primary_page'] = None
        
        # Add top-level subject, chapter for easy filtering
        if 'metadata' in chunk:
            chunk['subject'] = chunk['metadata'].get('subject')
            chunk['chapter'] = chunk['metadata'].get('chapter')
            chunk['class'] = chunk['metadata'].get('class')
        
        # Classify content type
        chunk['content_type'] = self._classify_content_type(chunk.get('text', ''))
        
        # Detect special content
        text = chunk.get('text', '')
        chunk['has_equations'] = bool(re.search(r'[=+\-*/()]|\d+\.\d+|→|—', text))
        chunk['has_questions'] = bool(re.search(r'\?|(question|exercise|activity)', text, re.IGNORECASE))
        
        # Extract keywords (simple approach)
        chunk['keywords'] = self._extract_keywords(text)
        
        return chunk
    
    def _classify_content_type(self, text: str) -> str:
        """
        Classify the type of content in the chunk
        """
        text_lower = text.lower()
        
        # Check for different content types
        if re.search(r'(definition|defined as|is called|is known as)[:.]', text, re.IGNORECASE):
            return 'definition'
        elif re.search(r'(example|for example|for instance|let us consider)[:.]', text, re.IGNORECASE):
            return 'example'
        elif re.search(r'(theorem|law|principle)[:.]', text, re.IGNORECASE):
            return 'theorem'
        elif re.search(r'(exercise|activity|question|solve|calculate)s?[:.]', text, re.IGNORECASE):
            return 'exercise'
        elif re.search(r'(note|remember|important)[:.]', text, re.IGNORECASE):
            return 'note'
        elif re.search(r'(figure|fig\.|diagram|table)\s+\d', text, re.IGNORECASE):
            return 'figure'
        elif 'chapter' in text_lower and len(text) < 500:
            return 'chapter_heading'
        else:
            return 'explanation'
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        Extract key terms from text (simple frequency-based approach)
        """
        # Remove common words and extract meaningful terms
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                     'this', 'that', 'these', 'those', 'it', 'its', 'will', 'can', 'may'}
        
        # Extract words (3+ chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Count frequency (excluding stopwords)
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top N keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:max_keywords]]
    
    def chunk_by_tokens(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text by token count with overlap
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunks with enhanced metadata
        """
        # Approximate tokens (1 token ≈ 4 characters)
        char_size = self.chunk_size * 4
        char_overlap = self.chunk_overlap * 4
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + char_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within overlap range
                search_start = max(end - char_overlap, start)
                sentence_end = max(
                    text.rfind('. ', search_start, end),
                    text.rfind('.\n', search_start, end),
                    text.rfind('?\n', search_start, end),
                    text.rfind('!\n', search_start, end)
                )
                
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_data = {
                    'text': chunk_text,
                    'start': start,
                    'end': end,
                    'char_count': len(chunk_text),
                    'token_count': len(chunk_text) // 4  # Approximate
                }
                
                # Add metadata if provided
                if metadata:
                    chunk_data['metadata'] = metadata
                
                # Enhance chunk with metadata classification
                chunk_data = self.enhance_chunk_metadata(chunk_data)
                
                chunks.append(chunk_data)
            
            # Move start position with overlap
            start = end - char_overlap if end < len(text) else len(text)
        
        return chunks
    
    def process_document(self, cleaned_doc: Dict) -> List[Dict]:
        """
        Process entire document into chunks
        
        Args:
            cleaned_doc: Cleaned document from cleaning service
            
        Returns:
            List of chunks with full metadata
        """
        logger.info(f"Chunking document: {cleaned_doc['metadata'].get('filename')}")
        
        text = cleaned_doc['cleaned_text']
        structures = cleaned_doc.get('structures', [])
        metadata = cleaned_doc['metadata']
        
        # Try structure-based chunking first
        if structures:
            chunks = self.chunk_by_structure(text, structures)
        else:
            chunks = self.chunk_by_tokens(text, metadata)
        
        # Add document metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['metadata'] = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
        
        logger.success(
            f"Created {len(chunks)} chunks from "
            f"{cleaned_doc['metadata'].get('filename')}"
        )
        
        return chunks


# Global chunking service instance
chunking_service = ChunkingService()
