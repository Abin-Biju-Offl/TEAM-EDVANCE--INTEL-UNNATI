"""
OCR Engine for NCERT Textbook Processing
=========================================

Production-grade OCR implementation with language awareness and accuracy focus.

Tool Selection Rationale:
-------------------------
1. Tesseract OCR 5.x:
   - Best open-source accuracy for English and Hindi
   - Supports 100+ languages including Indian languages
   - LSTM-based neural network engine
   - Actively maintained by Google
   
2. pdf2image (poppler):
   - High-quality PDF to image conversion
   - Preserves layout and quality
   - Handles scanned PDFs effectively

3. Pillow (PIL):
   - Image preprocessing and enhancement
   - Noise removal capabilities
   - Format conversions

4. pytesseract:
   - Python wrapper for Tesseract
   - Provides configuration control
   - Easy language switching
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages for OCR."""
    ENGLISH = "eng"
    HINDI = "hin"
    SANSKRIT = "san"
    URDU = "urd"
    COMBINED_ENG_HIN = "eng+hin"


class ContentType(Enum):
    """Types of content found in textbooks."""
    TEXT = "text"
    EQUATION = "equation"
    TABLE = "table"
    DIAGRAM = "diagram"
    EXERCISE = "exercise"
    MIXED = "mixed"


@dataclass
class PageMetadata:
    """Metadata for each extracted page."""
    class_number: int
    subject: str
    chapter_number: int
    chapter_title: str
    page_number: int
    language: str
    content_type: str
    source_file: str
    ocr_confidence: float
    processing_timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExtractedPage:
    """Complete extracted page with content and metadata."""
    text: str
    metadata: PageMetadata
    bounding_boxes: Optional[List[Dict]] = None
    preprocessing_applied: List[str] = None
    ocr_errors: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'metadata': self.metadata.to_dict(),
            'bounding_boxes': self.bounding_boxes,
            'preprocessing_applied': self.preprocessing_applied or [],
            'ocr_errors': self.ocr_errors or []
        }


class ImagePreprocessor:
    """Handles image preprocessing for improved OCR accuracy."""
    
    def __init__(self):
        self.applied_operations: List[str] = []
    
    def preprocess(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Apply preprocessing pipeline to enhance OCR accuracy.
        
        Args:
            image: Input PIL Image
            aggressive: Apply more aggressive noise removal (for poor scans)
        
        Returns:
            Preprocessed PIL Image
        """
        self.applied_operations = []
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            self.applied_operations.append('rgb_conversion')
        
        # Convert to numpy array for OpenCV operations
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        self.applied_operations.append('grayscale')
        
        # Noise removal using bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        self.applied_operations.append('bilateral_filter')
        
        # Adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        self.applied_operations.append('adaptive_threshold')
        
        if aggressive:
            # Morphological operations to remove noise
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            self.applied_operations.append('morphological_closing')
        
        # Convert back to PIL Image
        processed_image = Image.fromarray(binary)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed_image)
        processed_image = enhancer.enhance(1.5)
        self.applied_operations.append('contrast_enhancement')
        
        # Upscale for better OCR (if image is small)
        width, height = processed_image.size
        if width < 1500 or height < 1500:
            scale_factor = 2
            new_size = (width * scale_factor, height * scale_factor)
            processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
            self.applied_operations.append(f'upscale_{scale_factor}x')
        
        return processed_image
    
    def remove_headers_footers(
        self, 
        image: Image.Image, 
        header_percent: float = 0.08,
        footer_percent: float = 0.08
    ) -> Image.Image:
        """
        Remove header and footer regions from page image.
        
        Args:
            image: Input PIL Image
            header_percent: Percentage of height to crop from top
            footer_percent: Percentage of height to crop from bottom
        
        Returns:
            Cropped PIL Image
        """
        width, height = image.size
        
        # Calculate crop boundaries
        top_crop = int(height * header_percent)
        bottom_crop = int(height * (1 - footer_percent))
        
        # Crop the image
        cropped = image.crop((0, top_crop, width, bottom_crop))
        self.applied_operations.append(f'crop_header_footer')
        
        return cropped


class OCREngine:
    """
    Production-grade OCR engine for NCERT textbook processing.
    
    Prioritizes accuracy and traceability over speed.
    """
    
    def __init__(
        self, 
        tesseract_path: Optional[str] = None,
        default_language: Language = Language.COMBINED_ENG_HIN
    ):
        """
        Initialize OCR engine.
        
        Args:
            tesseract_path: Custom path to Tesseract executable
            default_language: Default language for OCR
        """
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        self.default_language = default_language
        self.preprocessor = ImagePreprocessor()
        
        # Verify Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found or not properly installed: {e}")
            raise RuntimeError("Tesseract OCR is required but not found")
    
    def extract_text_from_image(
        self,
        image: Image.Image,
        language: Optional[Language] = None,
        config: Optional[str] = None
    ) -> Tuple[str, float, List[Dict]]:
        """
        Extract text from image with confidence scores.
        
        Args:
            image: PIL Image object
            language: OCR language (uses default if None)
            config: Custom Tesseract configuration
        
        Returns:
            Tuple of (extracted_text, confidence_score, bounding_boxes)
        """
        lang = language or self.default_language
        
        # Default configuration for best accuracy
        if config is None:
            config = '--psm 3 --oem 3'  # PSM 3: Fully automatic page segmentation
        
        try:
            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=lang.value,
                config=config
            )
            
            # Get detailed data including confidence
            data = pytesseract.image_to_data(
                image,
                lang=lang.value,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate average confidence
            confidences = [
                int(conf) for conf in data['conf'] 
                if conf != '-1' and str(conf).strip()
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract bounding boxes for traceability
            bounding_boxes = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Only include confident detections
                    bounding_boxes.append({
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            logger.info(f"OCR completed with {avg_confidence:.2f}% average confidence")
            return text, avg_confidence, bounding_boxes
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise
    
    def process_pdf_page(
        self,
        pdf_path: Path,
        page_num: int,
        metadata: PageMetadata,
        dpi: int = 300,
        preprocess: bool = True,
        remove_margins: bool = True
    ) -> ExtractedPage:
        """
        Process a single page from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            metadata: Page metadata
            dpi: DPI for PDF to image conversion (higher = better quality)
            preprocess: Apply image preprocessing
            remove_margins: Remove headers and footers
        
        Returns:
            ExtractedPage object with text and metadata
        """
        errors = []
        
        try:
            # Convert PDF page to image
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num + 1,
                last_page=page_num + 1,
                fmt='png'
            )
            
            if not images:
                raise ValueError(f"No image extracted from page {page_num}")
            
            image = images[0]
            logger.info(f"Converted PDF page {page_num} to image at {dpi} DPI")
            
        except Exception as e:
            error_msg = f"PDF to image conversion failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        return self._process_image(
            image, metadata, preprocess, remove_margins, errors
        )
    
    def process_image_file(
        self,
        image_path: Path,
        metadata: PageMetadata,
        preprocess: bool = True,
        remove_margins: bool = True
    ) -> ExtractedPage:
        """
        Process an image file (scanned page).
        
        Args:
            image_path: Path to image file
            metadata: Page metadata
            preprocess: Apply image preprocessing
            remove_margins: Remove headers and footers
        
        Returns:
            ExtractedPage object with text and metadata
        """
        errors = []
        
        try:
            image = Image.open(image_path)
            logger.info(f"Loaded image from {image_path}")
        except Exception as e:
            error_msg = f"Failed to load image: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            raise
        
        return self._process_image(
            image, metadata, preprocess, remove_margins, errors
        )
    
    def _process_image(
        self,
        image: Image.Image,
        metadata: PageMetadata,
        preprocess: bool,
        remove_margins: bool,
        errors: List[str]
    ) -> ExtractedPage:
        """Internal method to process an image with OCR."""
        preprocessing_applied = []
        
        try:
            # Remove headers and footers if requested
            if remove_margins:
                image = self.preprocessor.remove_headers_footers(image)
                preprocessing_applied.extend(self.preprocessor.applied_operations)
            
            # Apply preprocessing if requested
            if preprocess:
                image = self.preprocessor.preprocess(image)
                preprocessing_applied.extend(self.preprocessor.applied_operations)
            
            # Perform OCR
            text, confidence, bounding_boxes = self.extract_text_from_image(
                image,
                language=Language(metadata.language)
            )
            
            # Update metadata with confidence
            metadata.ocr_confidence = confidence
            
            # Validate extraction
            if not text.strip():
                error_msg = f"No text extracted from page {metadata.page_number}"
                logger.warning(error_msg)
                errors.append(error_msg)
            
            if confidence < 50.0:
                error_msg = f"Low OCR confidence: {confidence:.2f}%"
                logger.warning(error_msg)
                errors.append(error_msg)
            
            return ExtractedPage(
                text=text,
                metadata=metadata,
                bounding_boxes=bounding_boxes,
                preprocessing_applied=preprocessing_applied,
                ocr_errors=errors if errors else None
            )
            
        except Exception as e:
            error_msg = f"Image processing failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            # Return partial result with errors
            return ExtractedPage(
                text="",
                metadata=metadata,
                bounding_boxes=None,
                preprocessing_applied=preprocessing_applied,
                ocr_errors=errors
            )
