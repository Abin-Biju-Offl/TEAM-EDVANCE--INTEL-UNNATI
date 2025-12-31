"""
Data Processing Script

Processes Class 10 Science PDFs through complete pipeline:
Phase 1: OCR
Phase 2: Cleaning
Phase 3: Chunking
Phase 4: Embeddings
Phase 5: FAISS Index Building

Run this script once to build the vector store before starting the server.
"""

import sys
from pathlib import Path
from loguru import logger
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.services.ocr_service import ocr_service
from app.services.cleaning_service import cleaning_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service


def main():
    """Run complete data processing pipeline"""
    
    logger.info("="*80)
    logger.info("NCERT RAG System - Data Processing Pipeline")
    logger.info("="*80)
    
    # Define data paths - process ALL subjects
    base_data_dir = Path(settings.data_dir) / "CLASS-X"
    
    # Collect all subject directories
    all_data_dirs = []
    
    # Social Science
    social_science_dir = base_data_dir / "Social-Science-X" / "social-science - English"
    if social_science_dir.exists():
        all_data_dirs.append(social_science_dir)
    
    # Science
    science_dir = base_data_dir / "Science-X" / "Science-X English"
    if science_dir.exists():
        all_data_dirs.append(science_dir)
    
    # English
    english_dir = base_data_dir / "English-X"
    if english_dir.exists():
        # Add all English subdirectories
        for subdir in english_dir.iterdir():
            if subdir.is_dir():
                all_data_dirs.append(subdir)
    
    data_dirs = all_data_dirs
    
    # Use "all-subjects" as the combined subject name
    subject_name = "all-subjects"
    
    processed_dir = Path(settings.processed_data_dir) / subject_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processed data directory: {processed_dir}")
    logger.info(f"Processing subject: {subject_name}")
    
    # Collect all PDF files from all directories
    all_pdf_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            pdfs = list(data_dir.glob("*.pdf"))
            all_pdf_files.extend(pdfs)
            logger.info(f"Found {len(pdfs)} PDFs in {data_dir.name}")
        else:
            logger.warning(f"Directory not found: {data_dir}")
    
    pdf_files = all_pdf_files
    logger.info(f"Total PDF files to process: {len(pdf_files)}")
    
    if len(pdf_files) == 0:
        logger.error("No PDF files found in any data directory")
        return
    
    # ========================================
    # Phase 1: OCR
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: OCR Processing")
    logger.info("="*80)
    
    ocr_cache_file = processed_dir / "ocr_results.json"
    
    if ocr_cache_file.exists():
        logger.info("Loading cached OCR results...")
        with open(ocr_cache_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        logger.success(f"Loaded {len(ocr_results)} cached OCR results")
    else:
        logger.info("Running OCR on PDF files...")
        # Process all directories
        ocr_results = []
        for data_dir in data_dirs:
            if data_dir.exists():
                logger.info(f"Processing PDFs from {data_dir.name}...")
                results = ocr_service.process_directory(str(data_dir))
                ocr_results.extend(results)
        
        # Save OCR results
        with open(ocr_cache_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_results, f, ensure_ascii=False, indent=2)
        logger.success(f"Saved OCR results to {ocr_cache_file}")
    
    # ========================================
    # Phase 2: Text Cleaning
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Text Cleaning & Structure Recovery")
    logger.info("="*80)
    
    cleaning_cache_file = processed_dir / "cleaned_documents.json"
    
    # Always reprocess if cache is metadata-only (missing cleaned_text)
    should_process = True
    if cleaning_cache_file.exists():
        logger.info("Checking cached cleaned documents...")
        with open(cleaning_cache_file, 'r', encoding='utf-8') as f:
            cached_docs = json.load(f)
        # Check if cache has actual cleaned text
        if cached_docs and 'cleaned_text' in cached_docs[0]:
            cleaned_docs = cached_docs
            should_process = False
            logger.success(f"Loaded {len(cleaned_docs)} cached cleaned documents")
    
    if should_process:
        logger.info("Cleaning documents...")
        cleaned_docs = []
        for ocr_result in tqdm(ocr_results, desc="Cleaning"):
            if ocr_result.get('full_text'):
                cleaned_doc = cleaning_service.process_document(ocr_result)
                cleaned_docs.append(cleaned_doc)
        
        # Save complete cleaned documents
        with open(cleaning_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)
        logger.success(f"Processed and saved {len(cleaned_docs)} documents")
    
    # ========================================
    # Phase 3: Chunking
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Semantic Chunking")
    logger.info("="*80)
    
    logger.info("Creating chunks from cleaned documents...")
    all_chunks = []
    for cleaned_doc in tqdm(cleaned_docs, desc="Chunking"):
        if isinstance(cleaned_doc, dict) and 'cleaned_text' in cleaned_doc:
            chunks = chunking_service.process_document(cleaned_doc)
            all_chunks.extend(chunks)
    
    logger.success(f"Created {len(all_chunks)} total chunks")
    
    # ========================================
    # Phase 4: Embeddings
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: Embedding Generation")
    logger.info("="*80)
    
    logger.info("Generating embeddings with SentenceTransformers...")
    embedded_chunks = embedding_service.embed_chunks(all_chunks)
    
    logger.success(f"Generated embeddings for {len(embedded_chunks)} chunks")
    
    # ========================================
    # Phase 5: FAISS Index Building
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: FAISS Index Building")
    logger.info("="*80)
    
    # Extract embeddings as numpy array
    import numpy as np
    embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks])
    
    logger.info(f"Building FAISS index with {embeddings.shape[0]} vectors...")
    faiss_service.create_index(embeddings, embedded_chunks)
    
    # Save index to subject-specific directory
    vector_store_dir = Path(settings.vector_store_dir) / subject_name
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    faiss_service.save_index(str(vector_store_dir / "faiss_index"))
    
    logger.success(f"FAISS index built and saved to {vector_store_dir}!")
    
    # ========================================
    # Summary
    # ========================================
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Subject: {subject_name}")
    logger.info(f"PDF files processed: {len(pdf_files)}")
    logger.info(f"Total chunks created: {len(embedded_chunks)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"FAISS index type: {settings.faiss_index_type}")
    logger.info(f"Vector store location: {vector_store_dir}")
    logger.info("\n✅ System ready! You can now start the FastAPI server.")
    logger.info("   Run: python main.py")


if __name__ == "__main__":
    main()
