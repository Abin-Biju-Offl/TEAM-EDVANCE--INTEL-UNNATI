"""
Configuration for NCERT Ingestion Pipeline
==========================================

Centralized configuration management for the OCR and ingestion system.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import json


@dataclass
class OCRConfig:
    """OCR Engine configuration."""
    tesseract_path: Optional[str] = None  # Auto-detect if None
    default_language: str = "eng+hin"  # Combined English + Hindi
    dpi: int = 300  # DPI for PDF conversion
    confidence_threshold: float = 50.0  # Minimum acceptable confidence
    psm_mode: int = 3  # Page segmentation mode (3 = fully automatic)
    oem_mode: int = 3  # OCR Engine mode (3 = LSTM only)


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    enabled: bool = True
    remove_margins: bool = True
    header_crop_percent: float = 0.08  # Remove top 8%
    footer_crop_percent: float = 0.08  # Remove bottom 8%
    apply_denoising: bool = True
    apply_binarization: bool = True
    apply_contrast_enhancement: bool = True
    upscale_small_images: bool = True
    upscale_threshold: int = 1500  # Upscale if dimension < this


@dataclass
class PipelineConfig:
    """Pipeline processing configuration."""
    max_workers: int = 1  # Number of parallel workers
    save_intermediate_images: bool = False  # Save preprocessed images
    enable_progress_bar: bool = True
    batch_size: int = 10  # Pages per batch for memory management
    retry_failed_pages: bool = True
    max_retries: int = 2


@dataclass
class OutputConfig:
    """Output configuration."""
    output_dir: Path = Path("output/extracted")
    save_bounding_boxes: bool = True  # Include bbox data for traceability
    pretty_print_json: bool = True
    include_statistics: bool = True


class ConfigManager:
    """Manages pipeline configuration with defaults and validation."""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.ocr = OCRConfig()
        self.preprocessing = PreprocessingConfig()
        self.pipeline = PipelineConfig()
        self.output = OutputConfig()
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: Path):
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Update OCR config
        if 'ocr' in config_data:
            for key, value in config_data['ocr'].items():
                if hasattr(self.ocr, key):
                    setattr(self.ocr, key, value)
        
        # Update preprocessing config
        if 'preprocessing' in config_data:
            for key, value in config_data['preprocessing'].items():
                if hasattr(self.preprocessing, key):
                    setattr(self.preprocessing, key, value)
        
        # Update pipeline config
        if 'pipeline' in config_data:
            for key, value in config_data['pipeline'].items():
                if hasattr(self.pipeline, key):
                    setattr(self.pipeline, key, value)
        
        # Update output config
        if 'output' in config_data:
            for key, value in config_data['output'].items():
                if hasattr(self.output, key):
                    if key == 'output_dir':
                        setattr(self.output, key, Path(value))
                    else:
                        setattr(self.output, key, value)
    
    def save_to_file(self, config_file: Path):
        """Save current configuration to JSON file."""
        config_data = {
            'ocr': {
                'tesseract_path': self.ocr.tesseract_path,
                'default_language': self.ocr.default_language,
                'dpi': self.ocr.dpi,
                'confidence_threshold': self.ocr.confidence_threshold,
                'psm_mode': self.ocr.psm_mode,
                'oem_mode': self.ocr.oem_mode
            },
            'preprocessing': {
                'enabled': self.preprocessing.enabled,
                'remove_margins': self.preprocessing.remove_margins,
                'header_crop_percent': self.preprocessing.header_crop_percent,
                'footer_crop_percent': self.preprocessing.footer_crop_percent,
                'apply_denoising': self.preprocessing.apply_denoising,
                'apply_binarization': self.preprocessing.apply_binarization,
                'apply_contrast_enhancement': self.preprocessing.apply_contrast_enhancement,
                'upscale_small_images': self.preprocessing.upscale_small_images,
                'upscale_threshold': self.preprocessing.upscale_threshold
            },
            'pipeline': {
                'max_workers': self.pipeline.max_workers,
                'save_intermediate_images': self.pipeline.save_intermediate_images,
                'enable_progress_bar': self.pipeline.enable_progress_bar,
                'batch_size': self.pipeline.batch_size,
                'retry_failed_pages': self.pipeline.retry_failed_pages,
                'max_retries': self.pipeline.max_retries
            },
            'output': {
                'output_dir': str(self.output.output_dir),
                'save_bounding_boxes': self.output.save_bounding_boxes,
                'pretty_print_json': self.output.pretty_print_json,
                'include_statistics': self.output.include_statistics
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration values."""
        # Validate DPI
        if self.ocr.dpi < 150 or self.ocr.dpi > 600:
            raise ValueError(f"DPI must be between 150 and 600, got {self.ocr.dpi}")
        
        # Validate confidence threshold
        if self.ocr.confidence_threshold < 0 or self.ocr.confidence_threshold > 100:
            raise ValueError(
                f"Confidence threshold must be 0-100, got {self.ocr.confidence_threshold}"
            )
        
        # Validate crop percentages
        if not (0 <= self.preprocessing.header_crop_percent <= 0.3):
            raise ValueError("Header crop percent must be between 0 and 0.3")
        if not (0 <= self.preprocessing.footer_crop_percent <= 0.3):
            raise ValueError("Footer crop percent must be between 0 and 0.3")
        
        return True
