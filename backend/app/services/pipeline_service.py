"""
Pipeline Service

Orchestrates the complete RAG pipeline from query to answer.
Integrates all phases: Retrieval -> Generation -> Safety Checks
"""

from typing import Union, List, Dict, Tuple
from loguru import logger
import re

from app.models.schemas import (
    QueryRequest, SuccessResponse, RejectionResponse,
    Citation
)
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from app.services.safety_service import safety_service, RejectionReason
from app.services.grok_service import groq_service
from app.core.config import settings
import os


class PipelineService:
    """Complete RAG pipeline orchestrator with Grok LLM"""
    
    def __init__(self):
        """Initialize pipeline"""
        self.embedding_service = embedding_service
        self.faiss_service = faiss_service
        self.safety_service = safety_service
        self.groq_service = groq_service
        
        mode = "with Groq LLM" if groq_service.is_available() else "retrieval-only"
        logger.info(f"Pipeline service initialized ({mode})")

    def _generate_extractive_fallback(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        class_num: int,
        subject: str,
        language: str,
    ) -> Tuple[str, List[Citation], float]:
        """Generate a minimal, citation-backed answer without an LLM.

        This is used only when the configured LLM provider is unavailable
        (e.g., invalid API key / insufficient quota). It keeps the system
        functional by returning excerpted NCERT text with citations.
        """
        if not retrieved_chunks:
            return (
                "I don't know based on NCERT textbooks.",
                [],
                0.0,
            )

        top = retrieved_chunks[0]
        text = (top.get("text") or "").strip()

        # Take a short excerpt (first 2-3 sentences) and add citations.
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if len(s.strip()) > 20]
        excerpt = sentences[:3] if sentences else [text[:400]]
        excerpt = [s for s in excerpt if s]

        if language == "hi":
            answer = "\n".join([f"{s} [Source 1]" for s in excerpt])
        else:
            answer = "\n".join([f"{s}. [Source 1]" for s in excerpt])

        meta = top.get("metadata", {}) or {}
        
        # Use enhanced metadata if available (from Phase 1 improvements)
        primary_page = top.get('primary_page') or meta.get("page", "Unknown")
        content_type = top.get('content_type', 'explanation')
        
        citation = Citation(
            class_number=str(meta.get("class", class_num)),
            subject=str(meta.get("subject", subject)),
            chapter=str(meta.get("chapter", "Unknown")),
            page=str(primary_page),
            section=meta.get("section"),
        )
        
        # Log content type for debugging
        logger.debug(f"Retrieved {content_type} content from page {primary_page}")

        # Since we are directly quoting the retrieved text, grounding is high.
        grounding_score = 0.9
        return answer, [citation], grounding_score
    
    async def process_query(self, query: QueryRequest) -> Union[SuccessResponse, RejectionResponse]:
        """
        Process query pipeline with Grok LLM integration
        
        Steps:
        1. Embed question
        2. Retrieve from FAISS
        3. Safety check
        4. Generate answer using Grok LLM (or extractive fallback)
        
        Args:
            query: Query request from frontend
            
        Returns:
            Success or rejection response
        """
        logger.info(f"Processing query: '{query.question[:50]}...'")
        
        try:
            # Step 1: Embed question
            logger.debug("Step 1: Embedding question")
            query_embedding = self.embedding_service.encode_single(query.question)
            
            # Step 2: Retrieve from FAISS
            logger.debug(f"Step 2: Retrieving top 30 chunks")
            retrieved_chunks = self.faiss_service.search(
                query_embedding,
                k=30
            )
            
            # Log retrieval results
            if retrieved_chunks:
                top_score = retrieved_chunks[0].get('score', 0)
                logger.info(f"Retrieved {len(retrieved_chunks)} chunks, top score: {top_score:.3f}")
            
            # Step 3: Safety checks
            logger.debug("Step 3: Safety checks")
            safety_result = self.safety_service.check_all(
                question=query.question,
                retrieved_chunks=retrieved_chunks,
                answer=None,
                citations=None,
                grounding_score=None
            )
            
            if not safety_result.passed:
                logger.warning(f"Query rejected: {safety_result.reason.value}")
                return RejectionResponse(
                    reason=safety_result.message,
                    rejection_type=safety_result.reason.value,
                    metadata={
                        "stage": "retrieval",
                        "score": safety_result.score
                    }
                )
            
            # Step 4: Generate answer using Groq LLM or extractive fallback
            if self.groq_service.is_available():
                logger.debug("Step 4: Generating answer with Groq LLM")
                try:
                    answer, citations, grounding_score = self.groq_service.generate_answer(
                        question=query.question,
                        retrieved_chunks=retrieved_chunks,
                        class_num=query.class_number,
                        subject=query.subject,
                        language=query.language,
                    )
                    mode = "groq_llm"
                except Exception as llm_error:
                    logger.warning(f"Groq LLM failed, using extractive fallback: {llm_error}")
                    answer, citations, grounding_score = self._generate_extractive_fallback(
                        question=query.question,
                        retrieved_chunks=retrieved_chunks,
                        class_num=query.class_number,
                        subject=query.subject,
                        language=query.language,
                    )
                    mode = "extractive_fallback"
            else:
                logger.debug("Step 4: Generating extractive answer (LLM not available)")
                answer, citations, grounding_score = self._generate_extractive_fallback(
                    question=query.question,
                    retrieved_chunks=retrieved_chunks,
                    class_num=query.class_number,
                    subject=query.subject,
                    language=query.language,
                )
                mode = "retrieval_only"
            
            logger.success(
                f"Query successful: {len(answer)} chars, "
                f"{len(citations)} citations"
            )
            
            return SuccessResponse(
                answer=answer,
                citations=citations,
                grounding_score=grounding_score,
                metadata={
                    "retrieved_chunks": len(retrieved_chunks),
                    "avg_retrieval_score": sum(c.get('score', 0) for c in retrieved_chunks) / len(retrieved_chunks),
                    "mode": mode
                }
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            
            return RejectionResponse(
                reason=f"Internal error: {str(e)}",
                rejection_type=RejectionReason.INSUFFICIENT_CONTEXT.value,
                metadata={"error": str(e)}
            )


# Global pipeline instance
pipeline_service = PipelineService()
