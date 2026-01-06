"""
NCERT Textbook Processing System
==================================
Professional OCR, chunking, and embedding pipeline for NCERT textbooks.

Usage:
    python process_textbooks.py              # Interactive menu
    python process_textbooks.py --class 5 --subject English --language en
    python process_textbooks.py --batch      # Process all pending

Author: Intel Unnati Project Team
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.ocr_service import ocr_service
from app.services.cleaning_service import cleaning_service
from app.services.chunking_service import chunking_service
from app.services.embedding_service import embedding_service
from app.services.faiss_service import faiss_service
from loguru import logger


class TextbookProcessor:
    """Comprehensive textbook processing system"""
    
    # Class-specific filename prefixes for validation
    CLASS_PREFIXES = {
        5: {
            "English": "eesa",
            "Hindi": "ehve",
            "Mathematics": "eemm",
            "Physical Education": "eeky",
            "Urdu": "eust"
        },
        10: {
            "English": "jeff",
            "Hindi": "jehp",
            "Mathematics": "jemh",
            "Science": "jesc",
            "Social Science": "jess"
        }
    }
    
    def __init__(self):
        self.data_dir = Path("E:/WORK/intel unnati/data")
        self.output_dir = Path("processed_data")
        self.vector_dir = Path("vector_store")
        
    def get_available_classes(self) -> List[int]:
        """Scan data directory for available classes"""
        classes = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                # Extract class number (CLASS-X, CLASS VI, etc.)
                name = item.name.upper()
                if "CLASS-X" in name or "CLASS-10" in name:
                    classes.append(10)
                elif "CLASS-V" in name or "CLASS 5" in name or "CLASS-5" in name:
                    classes.append(5)
                elif "CLASS-VI" in name or "CLASS 6" in name or "CLASS-6" in name:
                    classes.append(6)
                elif "CLASS-IX" in name or "CLASS 9" in name or "CLASS-9" in name:
                    classes.append(9)
        return sorted(set(classes))
    
    def get_subjects_for_class(self, class_num: int) -> List[Dict[str, str]]:
        """Get available subjects for a class"""
        subjects = []
        
        # Map class number to directory name
        class_dirs = {
            5: ["CLASS V", "CLASS-V", "class-v"],
            6: ["CLASS-VI", "CLASS VI", "class-vi"],
            9: ["CLASS-IX", "CLASS IX", "class-ix"],
            10: ["CLASS-X", "CLASS-10", "class-x"]
        }
        
        for possible_dir in class_dirs.get(class_num, []):
            class_path = self.data_dir / possible_dir
            if class_path.exists():
                for subject_dir in class_path.iterdir():
                    if subject_dir.is_dir():
                        # Parse subject name and language
                        parts = subject_dir.name.split('-')
                        if len(parts) >= 2:
                            subject = parts[0].strip().title()
                            language = parts[-1].strip().title()
                            lang_code = 'hi' if 'Hindi' in language else 'ur' if 'Urdu' in language else 'en'
                            
                            # Count PDFs
                            pdfs = list(subject_dir.glob("*.pdf"))
                            if pdfs:
                                subjects.append({
                                    'subject': subject,
                                    'language': language,
                                    'lang_code': lang_code,
                                    'path': str(subject_dir),
                                    'pdf_count': len(pdfs)
                                })
                break
        
        return subjects
    
    def process_textbook(self, class_num: int, subject: str, language: str, 
                        lang_code: str, pdf_path: Path) -> bool:
        """Process a single textbook through the pipeline"""
        try:
            logger.info(f"Processing Class {class_num} {subject} ({language})")
            
            # Create output directories
            safe_subject = subject.lower().replace(' ', '-')
            output_path = self.output_dir / f"class-{class_num}" / safe_subject
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: OCR
            logger.info("Step 1/4: Running OCR...")
            pdf_files = list(Path(pdf_path).glob("*.pdf"))
            all_pages = []
            
            for pdf_file in pdf_files:
                logger.info(f"  Processing {pdf_file.name}")
                pages = ocr_service.process_pdf(str(pdf_file), lang_code)
                all_pages.extend(pages)
            
            ocr_output = output_path / "ocr_results.json"
            with open(ocr_output, 'w', encoding='utf-8') as f:
                json.dump(all_pages, f, ensure_ascii=False, indent=2)
            
            logger.success(f"  OCR complete: {len(all_pages)} pages")
            
            # Step 2: Cleaning
            logger.info("Step 2/4: Cleaning text...")
            cleaned_docs = []
            for page in all_pages:
                cleaned = cleaning_service.clean_document(
                    page['text'],
                    page['file_path'],
                    page['page_num']
                )
                cleaned_docs.append(cleaned)
            
            cleaned_output = output_path / "cleaned_documents.json"
            with open(cleaned_output, 'w', encoding='utf-8') as f:
                json.dump(cleaned_docs, f, ensure_ascii=False, indent=2)
            
            logger.success(f"  Cleaning complete: {len(cleaned_docs)} documents")
            
            # Step 3: Chunking
            logger.info("Step 3/4: Chunking documents...")
            all_chunks = []
            for doc in cleaned_docs:
                chunks = chunking_service.chunk_document(
                    doc['text'],
                    {
                        'file_path': doc['metadata']['file_path'],
                        'page_num': doc['metadata']['page_num'],
                        'class': str(class_num),
                        'subject': subject
                    }
                )
                all_chunks.extend(chunks)
            
            chunks_output = output_path / "chunks.json"
            with open(chunks_output, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
            logger.success(f"  Chunking complete: {len(all_chunks)} chunks")
            
            # Step 4: Embedding and Indexing
            logger.info("Step 4/4: Creating embeddings and FAISS index...")
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in all_chunks]
            embeddings = embedding_service.embed_batch(texts)
            
            # Create metadata
            metadata_list = [chunk['metadata'] for chunk in all_chunks]
            
            # Create and save index
            index_path = self.vector_dir / f"class-{class_num}" / f"{safe_subject}-{lang_code}"
            faiss_service.create_and_save_index(
                embeddings,
                metadata_list,
                str(index_path)
            )
            
            logger.success(f"  Index saved: {len(embeddings)} vectors")
            logger.success(f"✓ Processing complete for Class {class_num} {subject} ({language})")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Error processing Class {class_num} {subject}: {e}")
            return False
    
    def interactive_mode(self):
        """Interactive menu for processing"""
        print("\n" + "="*70)
        print("NCERT TEXTBOOK PROCESSING SYSTEM")
        print("="*70)
        
        # Get available classes
        classes = self.get_available_classes()
        print(f"\nAvailable Classes: {', '.join(map(str, classes))}")
        
        # Select class
        while True:
            try:
                class_input = input("\nEnter class number (or 'q' to quit): ").strip()
                if class_input.lower() == 'q':
                    return
                class_num = int(class_input)
                if class_num in classes:
                    break
                print(f"Invalid class. Choose from: {classes}")
            except ValueError:
                print("Please enter a valid number")
        
        # Get subjects
        subjects = self.get_subjects_for_class(class_num)
        if not subjects:
            print(f"No subjects found for Class {class_num}")
            return
        
        # Display subjects
        print(f"\nAvailable subjects for Class {class_num}:")
        for i, subj in enumerate(subjects, 1):
            print(f"  {i}. {subj['subject']} ({subj['language']}) - {subj['pdf_count']} PDFs")
        
        # Select subject
        while True:
            try:
                subj_input = input("\nEnter subject number (or 'all' for all, 'q' to quit): ").strip()
                if subj_input.lower() == 'q':
                    return
                
                if subj_input.lower() == 'all':
                    selected = subjects
                    break
                
                subj_idx = int(subj_input) - 1
                if 0 <= subj_idx < len(subjects):
                    selected = [subjects[subj_idx]]
                    break
                print(f"Invalid selection. Choose 1-{len(subjects)}")
            except ValueError:
                print("Please enter a valid number or 'all'")
        
        # Process selected subjects
        print("\n" + "="*70)
        print(f"Processing {len(selected)} subject(s)...")
        print("="*70 + "\n")
        
        results = []
        for subj in selected:
            success = self.process_textbook(
                class_num,
                subj['subject'],
                subj['language'],
                subj['lang_code'],
                Path(subj['path'])
            )
            results.append((subj['subject'], success))
            print()
        
        # Summary
        print("="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        for subject, success in results:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{status}: {subject}")
        print("="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NCERT Textbook Processing System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--class', type=int, dest='class_num',
                       help='Class number (5, 6, 9, 10)')
    parser.add_argument('--subject', type=str,
                       help='Subject name')
    parser.add_argument('--language', type=str,
                       help='Language code (en, hi, ur)')
    parser.add_argument('--batch', action='store_true',
                       help='Process all pending subjects')
    
    args = parser.parse_args()
    
    processor = TextbookProcessor()
    
    # Command line mode
    if args.class_num and args.subject and args.language:
        subjects = processor.get_subjects_for_class(args.class_num)
        matching = [s for s in subjects 
                   if s['subject'].lower() == args.subject.lower() 
                   and s['lang_code'] == args.language.lower()]
        
        if matching:
            subj = matching[0]
            processor.process_textbook(
                args.class_num,
                subj['subject'],
                subj['language'],
                subj['lang_code'],
                Path(subj['path'])
            )
        else:
            print(f"Subject not found: {args.subject}")
            return
    
    # Batch mode
    elif args.batch:
        print("Batch processing not yet implemented")
        return
    
    # Interactive mode
    else:
        processor.interactive_mode()


if __name__ == "__main__":
    main()
