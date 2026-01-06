"""
OCR Service - Phase 1

Handles PDF to text extraction using Tesseract OCR.
Integrates existing OCR pipeline code.
"""

import pytesseract
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
from typing import List, Dict
from loguru import logger
import os
import re
import io

from app.core.config import settings


class OCRService:
    """OCR service for PDF processing"""
    
    def __init__(self):
        """Initialize OCR service with Tesseract configuration"""
        # Set Tesseract path
        if os.path.exists(settings.tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path
            logger.info(f"Tesseract configured at: {settings.tesseract_path}")
        else:
            logger.warning(f"Tesseract not found at: {settings.tesseract_path}")
            logger.info("Attempting to use system Tesseract")
        
        self.dpi = settings.ocr_dpi
        self.language = settings.ocr_language
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF using OCR
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Starting OCR for: {pdf_path}")
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            logger.info(f"PDF has {total_pages} pages")
            
            # Extract text from each page
            pages_text = []
            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    
                    # Convert page to image
                    pix = page.get_pixmap(dpi=self.dpi)
                    img_bytes = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_bytes))
                    
                    # Run OCR on image
                    text = pytesseract.image_to_string(
                        image,
                        lang=self.language
                    )
                    
                    # Extract page number from footer (last line often contains page number)
                    extracted_page_num = self._extract_page_number_from_text(text)
                    
                    pages_text.append({
                        'page_number': page_num + 1,
                        'pdf_page_number': extracted_page_num,  # Actual page number from PDF
                        'text': text,
                        'char_count': len(text)
                    })
                    
                    logger.debug(f"Page {page_num + 1}: Extracted {len(text)} characters")
                    
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num + 1}: {str(e)}")
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': '',
                        'error': str(e)
                    })
            
            pdf_document.close()
            
            # Combine all pages
            full_text = "\n\n".join([p['text'] for p in pages_text if p.get('text')])
            
            # Extract chapter from filename (last 2 digits)
            filename = Path(pdf_path).stem
            chapter = self._extract_chapter_from_filename(filename)
            
            result = {
                'pdf_path': pdf_path,
                'total_pages': total_pages,
                'total_characters': len(full_text),
                'full_text': full_text,
                'pages': pages_text,
                'metadata': {
                    'dpi': self.dpi,
                    'language': self.language,
                    'chapter': chapter,
                    'filename': filename
                }
            }
            
            logger.success(f"OCR completed: {len(pages_text)} pages, {len(full_text)} chars")
            return result
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise
    
    def _extract_page_number_from_text(self, text: str) -> str:
        """
        Extract page number from text (usually in footer)
        
        Args:
            text: Page text
            
        Returns:
            Extracted page number or 'Unknown'
        """
        if not text:
            return 'Unknown'
        
        # Get last few lines where page number usually appears
        lines = text.strip().split('\n')
        last_lines = lines[-3:] if len(lines) > 3 else lines
        
        for line in reversed(last_lines):
            line = line.strip()
            # Look for standalone numbers or page number patterns
            # Common patterns: "25", "Page 25", "25 Science", "Science 25"
            if re.match(r'^\d{1,3}$', line):
                return line
            match = re.search(r'\b(\d{1,3})\b', line)
            if match:
                return match.group(1)
        
        return 'Unknown'
    
    def _extract_chapter_from_filename(self, filename: str) -> str:
        """
        Extract chapter number from filename (last 2 digits)
        
        Args:
            filename: PDF filename without extension
            
        Returns:
            Chapter number as string
        """
        # Extract last 2 digits from filename
        # Example: jesc101.pdf -> chapter 01, jesc110.pdf -> chapter 10
        match = re.search(r'(\d{2})$', filename)
        if match:
            chapter_num = match.group(1)
            # Remove leading zero for display: "01" -> "1"
            return str(int(chapter_num))
        return 'Unknown'
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of OCR results for each PDF
        """
        directory = Path(directory_path)
        pdf_files = list(directory.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        results = []
        for pdf_file in pdf_files:
            try:
                result = self.extract_text_from_pdf(str(pdf_file))
                result['filename'] = pdf_file.name
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                results.append({
                    'filename': pdf_file.name,
                    'error': str(e),
                    'full_text': ''
                })
        
        logger.info(f"Processed {len(results)} PDFs successfully")
        return results
    
    def process_pdf_with_hindi(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from Hindi PDF using OCR with Hindi language model
        
        Args:
            pdf_path: Path to Hindi PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        logger.info(f"Starting Hindi OCR for: {pdf_path}")
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            logger.info(f"Opened PDF with {len(doc)} pages")
            
            # Use lower DPI for faster processing (150 DPI balances speed and quality)
            fast_dpi = 150
            
            # Extract text from each page using Hindi model
            pages_text = []
            for page_num in range(len(doc)):
                try:
                    # Get page
                    page = doc[page_num]
                    
                    # Render page to image at 150 DPI (faster than 300 DPI)
                    zoom = fast_dpi / 72  # 72 DPI is default
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert pixmap to PIL Image (faster method)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Run OCR with Hindi language model
                    text = pytesseract.image_to_string(
                        image,
                        lang='hin'  # Hindi language model
                    )
                    
                    # Extract page number
                    extracted_page_num = self._extract_page_number_from_text(text)
                    
                    pages_text.append({
                        'page_number': page_num + 1,
                        'pdf_page_number': extracted_page_num,
                        'text': text,
                        'char_count': len(text)
                    })
                    
                    logger.debug(f"Page {page_num + 1}: Extracted {len(text)} characters (Hindi)")
                    
                except Exception as e:
                    logger.error(f"Hindi OCR failed for page {page_num + 1}: {str(e)}")
                    pages_text.append({
                        'page_number': page_num + 1,
                        'text': '',
                        'error': str(e)
                    })
            
            # Close document
            doc.close()
            
            # Combine all pages
            full_text = "\n\n".join([p['text'] for p in pages_text if p.get('text')])
            
            # Extract chapter from filename
            filename = Path(pdf_path).stem
            chapter = self._extract_chapter_from_filename(filename)
            
            result = {
                'pdf_path': pdf_path,
                'total_pages': len(pages_text),
                'total_characters': len(full_text),
                'full_text': full_text,
                'pages': pages_text,
                'chapter': chapter,
                'language': 'hindi'
            }
            
            logger.success(f"Hindi OCR completed: {filename} - {len(full_text)} chars from {len(pages_text)} pages")
            
            return result
            
        except Exception as e:
            logger.error(f"Hindi OCR failed for {pdf_path}: {str(e)}")
            raise


# Global OCR service instance
ocr_service = OCRService()

