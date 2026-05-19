# src/__init__.py
"""
ECG Analyzer Package
Análisis de señales electrocardiográficas con detección de afecciones cardíacas
"""

__version__ = "1.0.0"
__author__ = "ECG Analyzer Team"

from .data_manager import ECGDataManager
from .signal_processor import ECGProcessor
from .feature_extractor import FeatureExtractor
from .report_generator import ReportGenerator
from .ui_styles import setup_page_style, COLORS, render_header, render_metric_card

__all__ = [
    'ECGDataManager',
    'ECGProcessor',
    'FeatureExtractor',
    'ReportGenerator',
    'setup_page_style',
    'COLORS',
    'render_header',
    'render_metric_card'
]