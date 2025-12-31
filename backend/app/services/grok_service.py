"""
Groq LLM Service

Handles interaction with Groq API for answer generation.
"""

from typing import List, Dict, Tuple
from loguru import logger
from openai import OpenAI
import re

from app.models.schemas import Citation
from app.core.config import settings


class GroqService:
    """Service for generating answers using Groq LLM"""
    
    def __init__(self):
        """Initialize Groq client"""
        self.client = None
        self.available = False
        self.model = settings.groq_model
        
        if settings.groq_api_key and settings.groq_api_key != "your_groq_api_key_here":
            try:
                self.client = OpenAI(
                    api_key=settings.groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                self.available = True
                logger.info(f"Groq LLM service initialized with model: {self.model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self.available = False
        else:
            logger.warning("Groq API key not configured, LLM generation disabled")
    
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return self.available and settings.use_llm
    
    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        class_num: int,
        subject: str,
        language: str,
    ) -> Tuple[str, List[Citation], float]:
        """
        Generate answer using Groq LLM
        
        Args:
            question: User's question
            retrieved_chunks: Retrieved context from FAISS
            class_num: Class number
            subject: Subject name
            language: Language code (en/hi)
            
        Returns:
            Tuple of (answer, citations, grounding_score)
        """
        if not self.is_available():
            raise RuntimeError("Groq service is not available")
        
        if not retrieved_chunks:
            return (
                "I don't know based on NCERT textbooks." if language == "en" 
                else "मुझे NCERT पाठ्यपुस्तकों के आधार पर नहीं पता।",
                [],
                0.0
            )
        
        try:
            # Prepare context from retrieved chunks
            context = self._prepare_context(retrieved_chunks, max_chunks=5)
            
            # Build prompt based on language
            prompt = self._build_prompt(question, context, language)
            
            # Call Groq API
            logger.debug(f"Calling Groq API with model: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(language)
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
            )
            
            grok_answer = response.choices[0].message.content.strip()
            
            # Extract citations from retrieved chunks
            citations = self._extract_citations(retrieved_chunks, class_num, subject)
            
            # Get textbook extraction for reference
            textbook_extract = self._get_textbook_extract(retrieved_chunks[:2], language)
            
            # Format final answer: Grok explanation + textbook reference + follow-up
            final_answer = self._format_complete_answer(grok_answer, textbook_extract, language)
            
            # Calculate grounding score based on citation usage
            grounding_score = self._calculate_grounding_score(grok_answer, citations)
            
            logger.success(f"Generated answer with {len(citations)} citations")
            
            return final_answer, citations, grounding_score
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            raise
    
    def _get_system_prompt(self, language: str) -> str:
        """Get system prompt for Grok based on language"""
        if language == "hi":
            return """आप एक NCERT शैक्षिक सहायक हैं। आपका कार्य छात्रों को अवधारणाओं को स्पष्ट, सरल तरीके से समझाना है।

नियम:
1. प्रदान किए गए NCERT संदर्भ का उपयोग करके विस्तृत स्पष्टीकरण दें
2. जटिल अवधारणाओं को सरल भाषा में समझाएं
3. छात्रों के लिए आकर्षक और शैक्षिक बनाएं
4. महत्वपूर्ण बिंदुओं के बाद [Source N] का संदर्भ दें
5. यदि संदर्भ में उत्तर नहीं है, तो ईमानदारी से बताएं"""
        else:
            return """You are an NCERT educational assistant. Your task is to explain concepts to students in a clear, simple manner.

Rules:
1. Provide detailed explanations using the provided NCERT context
2. Break down complex concepts into simple language
3. Make it engaging and educational for students
4. Reference [Source N] after key points
5. If the answer isn't in the context, state it honestly"""
    
    def _build_prompt(self, question: str, context: str, language: str) -> str:
        """Build prompt for Grok"""
        if language == "hi":
            return f"""संदर्भ (NCERT पाठ्यपुस्तकों से):
{context}

प्रश्न: {question}

कृपया इस प्रश्न का विस्तृत, स्पष्ट उत्तर दें जो छात्रों के लिए समझने में आसान हो। अवधारणा को अच्छी तरह समझाएं और महत्वपूर्ण बिंदुओं के बाद [Source 1], [Source 2], आदि जोड़ें।"""
        else:
            return f"""Context (from NCERT textbooks):
{context}

Question: {question}

Please provide a detailed, clear explanation that's easy for students to understand. Explain the concept well and add citations [Source 1], [Source 2], etc. after key points."""
    
    def _prepare_context(self, chunks: List[Dict], max_chunks: int = 5) -> str:
        """Prepare context string from retrieved chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {}) or {}
            
            # Add metadata info
            chapter = meta.get("chapter", "Unknown")
            page = chunk.get('primary_page') or meta.get("page", "Unknown")
            
            context_parts.append(
                f"[Source {i}] Chapter: {chapter}, Page: {page}\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_citations(
        self,
        chunks: List[Dict],
        class_num: int,
        subject: str
    ) -> List[Citation]:
        """Extract citations from retrieved chunks"""
        citations = []
        seen_sources = set()
        
        for chunk in chunks[:5]:  # Top 5 chunks
            meta = chunk.get("metadata", {}) or {}
            primary_page = chunk.get('primary_page') or meta.get("page", "Unknown")
            
            # Create unique identifier to avoid duplicate citations
            source_id = (
                str(meta.get("class", class_num)),
                str(meta.get("subject", subject)),
                str(meta.get("chapter", "Unknown")),
                str(primary_page)
            )
            
            if source_id not in seen_sources:
                citation = Citation(
                    class_number=str(meta.get("class", class_num)),
                    subject=str(meta.get("subject", subject)),
                    chapter=str(meta.get("chapter", "Unknown")),
                    page=str(primary_page),
                    section=meta.get("section"),
                )
                citations.append(citation)
                seen_sources.add(source_id)
        
        return citations
    
    def _calculate_grounding_score(self, answer: str, citations: List[Citation]) -> float:
        """
        Calculate grounding score based on citation usage
        
        Higher score means better grounding in source material
        """
        if not answer or not citations:
            return 0.0
        
        # Count citation markers in answer
        citation_pattern = r'\[Source \d+\]'
        citation_count = len(re.findall(citation_pattern, answer))
        
        # Base score on citation usage
        if citation_count == 0:
            return 0.3  # Low score if no citations used
        elif citation_count >= len(citations):
            return 0.9  # High score if all sources cited
        else:
            # Proportional score
            return 0.5 + (0.4 * citation_count / len(citations))
    
    def _get_textbook_extract(self, chunks: List[Dict], language: str) -> str:
        """Extract relevant text from top chunks"""
        if not chunks:
            return ""
        
        extracts = []
        for i, chunk in enumerate(chunks[:2], 1):  # Top 2 chunks
            text = chunk.get("text", "").strip()
            if text:
                # Take first 2-3 sentences
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 20]
                excerpt = ' '.join(sentences[:3]) if sentences else text[:300]
                extracts.append(f"[{i}] {excerpt}")
        
        return '\n\n'.join(extracts) if extracts else ""
    
    def _format_complete_answer(self, grok_answer: str, textbook_extract: str, language: str) -> str:
        """Format complete answer with Grok explanation + textbook reference + follow-up"""
        if language == "hi":
            separator = "\n\n" + "="*50 + "\n"
            textbook_header = "📚 NCERT पाठ्यपुस्तक से (सन्दर्भ):"
            followup = "\n\n" + "-"*50 + "\n💡 क्या आपका कोई और सवाल है? कृपया पूछने में संकोच न करें!"
        else:
            separator = "\n\n" + "="*50 + "\n"
            textbook_header = "📚 From NCERT Textbook (Reference):"
            followup = "\n\n" + "-"*50 + "\n💡 Do you have any doubts? Please feel free to ask!"
        
        # Combine all parts
        complete_answer = grok_answer
        
        if textbook_extract:
            complete_answer += f"{separator}{textbook_header}\n{textbook_extract}"
        
        complete_answer += followup
        
        return complete_answer


# Global service instance
groq_service = GroqService()
