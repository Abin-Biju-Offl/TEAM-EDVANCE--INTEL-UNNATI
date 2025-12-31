"""
Process English-X Dataset Only and Append to Existing Index

This script processes only the English textbook PDFs and appends them
to the existing FAISS index (which already contains Social Science and Science).
"""

import sys
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import json

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.services.ocr_service import ocr_service
from app.services.cleaning_service import cleaning_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service


def main():
    """Process English dataset and append to existing index"""
    
    logger.info("="*80)
    logger.info("Processing English-X Dataset ONLY")
    logger.info("="*80)
    
    # Define paths
    base_data_dir = Path(settings.data_dir) / "CLASS-X"
    english_dir = base_data_dir / "English-X"
    
    # Get English PDF files
    english_pdf_files = []
    if english_dir.exists():
        english_pdf_files = sorted([f for f in english_dir.glob("*.pdf")])
    
    if not english_pdf_files:
        logger.error(f"No PDF files found in {english_dir}")
        return
    
    logger.info(f"Found {len(english_pdf_files)} PDFs in English-X")
    for pdf in english_pdf_files:
        logger.info(f"  - {pdf.name}")
    
    # Use temporary directory for English processing
    english_processed_dir = Path(settings.processed_data_dir) / "english-only"
    english_processed_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # PHASE 1: OCR Processing
    # ========================================================================
    logger.info("="*80)
    logger.info("PHASE 1: OCR Processing (English Only)")
    logger.info("="*80)
    
    # Use process_directory method
    logger.info(f"Processing PDFs from {english_dir}...")
    ocr_results = ocr_service.process_directory(str(english_dir))
    logger.success(f"OCR completed: {len(ocr_results)} documents processed")
    
    # Save OCR results
    ocr_output = english_processed_dir / "ocr_results.json"
    with open(ocr_output, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    logger.success(f"OCR results saved: {ocr_output}")
    
    # ========================================================================
    # PHASE 2: Text Cleaning
    # ========================================================================
    logger.info("="*80)
    logger.info("PHASE 2: Text Cleaning")
    logger.info("="*80)
    
    cleaned_docs = []
    for doc in tqdm(ocr_results, desc="Cleaning"):
        try:
            cleaned = cleaning_service.process_document(doc)
            cleaned_docs.append(cleaned)
        except Exception as e:
            logger.error(f"Cleaning failed for {doc.get('filename', 'unknown')}: {e}")
            continue
    
    # Save cleaned documents
    cleaned_output = english_processed_dir / "cleaned_documents.json"
    with open(cleaned_output, 'w', encoding='utf-8') as f:
        json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)
    logger.success(f"Processed and saved {len(cleaned_docs)} documents")
    
    # ========================================================================
    # PHASE 3: Semantic Chunking
    # ========================================================================
    logger.info("="*80)
    logger.info("PHASE 3: Semantic Chunking")
    logger.info("="*80)
    
    all_chunks = []
    for doc in tqdm(cleaned_docs, desc="Chunking"):
        try:
            chunks = chunking_service.process_document(doc)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Chunking failed for {doc['filename']}: {e}")
            continue
    
    logger.success(f"Created {len(all_chunks)} total chunks from English dataset")
    
    # ========================================================================
    # PHASE 4: Embedding Generation
    # ========================================================================
    logger.info("="*80)
    logger.info("PHASE 4: Embedding Generation")
    logger.info("="*80)
    
    all_chunks = embedding_service.embed_chunks(all_chunks)
    logger.success(f"Generated embeddings for {len(all_chunks)} chunks")
    
    # ========================================================================
    # PHASE 5: Load Existing Index and Append
    # ========================================================================
    logger.info("="*80)
    logger.info("PHASE 5: Appending to Existing FAISS Index")
    logger.info("="*80)
    
    # Load existing index
    existing_index_path = Path(settings.vector_store_dir) / "all-subjects"
    if not existing_index_path.exists():
        logger.error(f"Existing index not found at {existing_index_path}")
        logger.info("Please ensure the all-subjects index exists before running this script")
        return
    
    logger.info(f"Loading existing index from {existing_index_path}")
    faiss_service.load_index(str(existing_index_path / "faiss_index"))
    existing_chunks = faiss_service.chunks
    logger.info(f"Loaded {len(existing_chunks)} existing chunks")
    
    # Combine old and new chunks
    combined_chunks = existing_chunks + all_chunks
    logger.info(f"Total chunks after adding English: {len(combined_chunks)}")
    
    # Extract embeddings from combined chunks
    logger.info("Extracting embeddings from combined chunks...")
    import numpy as np
    embeddings = np.array([chunk['embedding'] for chunk in combined_chunks])
    logger.info(f"Combined embeddings shape: {embeddings.shape}")
    
    # Build new combined index
    logger.info("Building combined FAISS index...")
    faiss_service.create_index(embeddings, combined_chunks)
    
    # Save combined index
    combined_output = Path(settings.vector_store_dir) / "all-subjects-with-english"
    combined_output.mkdir(parents=True, exist_ok=True)
    faiss_service.save_index(str(combined_output / "faiss_index"))
    logger.success(f"Combined index saved to {combined_output}")
    
    # ========================================================================
    # PROCESSING COMPLETE
    # ========================================================================
    logger.info("="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Previous chunks (Science + Social): {len(existing_chunks)}")
    logger.info(f"New English chunks: {len(all_chunks)}")
    logger.info(f"Total combined chunks: {len(combined_chunks)}")
    logger.info(f"Embedding dimension: {settings.embedding_dimension}")
    logger.info(f"FAISS index type: {settings.faiss_index_type}")
    logger.info(f"Vector store location: {combined_output}")
    logger.info("")
    logger.info("✅ To use the new index:")
    logger.info(f"   1. Copy {combined_output / 'faiss_index.*'} to {Path(settings.vector_store_dir) / 'faiss_index.*'}")
    logger.info("   2. Restart the FastAPI server")
    logger.info("")


if __name__ == "__main__":
    main()
