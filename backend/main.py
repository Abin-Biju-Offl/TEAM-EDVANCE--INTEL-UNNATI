"""
FastAPI Main Application

Entry point for the NCERT RAG system backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from datetime import datetime
import sys

from app.core.config import settings
from app.api.routes import router
from app.models.schemas import HealthResponse

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level=settings.log_level
)
logger.add(
    f"{settings.logs_dir}/app_{{time}}.log",
    rotation="1 day",
    retention="7 days",
    level=settings.log_level
)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="NCERT Educational RAG System - Intel Unnati Project",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["api"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'Development' if settings.debug else 'Production'}")
    
    # Don't pre-load any index - let the pipeline load class-specific indices on demand
    logger.info("Ready to load class-specific indices on demand")
    logger.success("System initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(f"Shutting down {settings.app_name}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from app.services.faiss_service import faiss_service
    from app.services.groq_service import groq_service
    
    # Check LLM status
    if groq_service.is_available():
        llm_status = "groq_available"
    else:
        llm_status = "retrieval_only"
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(),
        components={
            "faiss_index": "ready" if faiss_service.index is not None else "not_loaded",
            "llm": llm_status
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NCERT Educational RAG System",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers
    )
