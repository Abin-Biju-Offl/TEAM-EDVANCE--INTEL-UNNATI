"""
Configuration Module

Loads environment variables and provides application configuration.
Uses pydantic-settings for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Configuration
    app_name: str = Field(default="NCERT RAG System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # CORS Configuration
    cors_origins: str = Field(
        default="http://localhost:8080,http://localhost:3000,http://localhost:5500",
        env="CORS_ORIGINS"
    )
    
    # Data Paths
    data_dir: str = Field(default="../data", env="DATA_DIR")
    processed_data_dir: str = Field(default="./processed_data", env="PROCESSED_DATA_DIR")
    vector_store_dir: str = Field(default="./vector_store", env="VECTOR_STORE_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # Embedding Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=64, env="BATCH_SIZE")
    
    # FAISS Configuration
    faiss_index_type: str = Field(default="IVF", env="FAISS_INDEX_TYPE")
    faiss_nlist: int = Field(default=100, env="FAISS_NLIST")
    faiss_nprobe: int = Field(default=32, env="FAISS_NPROBE")
    
    # Chunking Configuration
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Retrieval Configuration
    top_k_retrieval: int = Field(default=20, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, env="TOP_K_RERANK")
    
    # Safety Thresholds
    min_retrieval_confidence: float = Field(default=0.38, env="MIN_RETRIEVAL_CONFIDENCE")
    min_chunks_required: int = Field(default=1, env="MIN_CHUNKS_REQUIRED")
    min_grounding_score: float = Field(default=0.65, env="MIN_GROUNDING_SCORE")
    
    # LLM Configuration (Groq)
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")
    groq_temperature: float = Field(default=0.3, env="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=1000, env="GROQ_MAX_TOKENS")
    use_llm: bool = Field(default=True, env="USE_LLM")
    
    # OCR Configuration
    tesseract_path: str = Field(
        default="C:/Program Files/Tesseract-OCR/tesseract.exe",
        env="TESSERACT_PATH"
    )
    ocr_dpi: int = Field(default=300, env="OCR_DPI")
    ocr_language: str = Field(default="eng", env="OCR_LANGUAGE")
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    class Config:
        # Always load the backend-local .env (not workspace root), regardless of CWD.
        env_file = str(Path(__file__).resolve().parents[2] / ".env")
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.processed_data_dir, exist_ok=True)
os.makedirs(settings.vector_store_dir, exist_ok=True)
os.makedirs(settings.logs_dir, exist_ok=True)
