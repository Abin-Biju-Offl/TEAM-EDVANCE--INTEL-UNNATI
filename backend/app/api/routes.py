"""
FastAPI Routes

API endpoints matching frontend contract.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
import json
from datetime import datetime
from pathlib import Path

from app.models.schemas import (
    QueryRequest, SuccessResponse, RejectionResponse,
    FeedbackRequest, FeedbackResponse
)
from app.services.pipeline_service import pipeline_service
from app.core.config import settings

router = APIRouter()


@router.post("/query", response_model=SuccessResponse | RejectionResponse)
async def process_query(query: QueryRequest):
    """
    Process user query and return answer or rejection
    
    Matches frontend contract exactly:
    - Success: { status: "success", answer, citations, grounding_score }
    - Rejection: { status: "rejected", reason, rejection_type }
    """
    logger.info(
        f"Query received: class={query.class_number}, "
        f"subject={query.subject}, "
        f"lang={query.language}"
    )
    
    try:
        result = await pipeline_service.process_query(query)
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Log user feedback for evaluation
    
    Stores feedback to JSON file for later analysis.
    """
    logger.info(f"Feedback received: helpful={feedback.helpful}")
    
    try:
        # Create feedback log entry
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": feedback.question,
            "answer": feedback.answer,
            "helpful": feedback.helpful,
            "comment": feedback.comment,
            "metadata": feedback.metadata
        }
        
        # Append to feedback log file
        feedback_file = Path(settings.logs_dir) / "feedback.jsonl"
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        logger.success(f"Feedback logged to {feedback_file}")
        
        return FeedbackResponse(success=True)
        
    except Exception as e:
        logger.error(f"Feedback logging failed: {str(e)}")
        # Don't fail the request if logging fails
        return FeedbackResponse(
            success=False,
            message=f"Feedback logging failed: {str(e)}"
        )
