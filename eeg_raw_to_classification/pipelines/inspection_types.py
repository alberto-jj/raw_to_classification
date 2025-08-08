"""Data structures for MEEG inspection pipeline"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np


@dataclass
class InspectionConfig:
    """Pure data class for inspection configuration"""
    output_path: Path
    max_files: Optional[int] = None
    generate_reports: bool = True
    generate_spectra: bool = True
    max_frequency: float = 200.0
    output_format: str = "json"  # json, yaml, or both


@dataclass 
class DatasetInfo:
    """Pure data class for dataset information"""
    label: str
    files: List[Path]
    example_file: Path
    skip: bool = False
    dataset_config: Dict = None


@dataclass
class MEEGInspectionResult:
    """Pure data class for single file inspection result"""
    file_path: Path
    montage: List[str]
    duration: float
    shape: Tuple[int, ...]
    sampling_freq: float
    info_str: str
    relative_path: str = ""
    success: bool = True
    error: Optional[str] = None


@dataclass
class DatasetInspectionResult:
    """Pure data class for dataset-level results"""
    dataset_label: str
    file_results: List[MEEGInspectionResult]
    common_montage: List[str]
    union_montage: List[str]
    duration_stats: Dict[str, float]
    duration_counts: Dict[float, int]
    success_count: int
    total_count: int


@dataclass
class GlobalInspectionResult:
    """Pure data class for cross-dataset results"""
    dataset_results: Dict[str, DatasetInspectionResult]
    global_common_montage: List[str]
    global_union_montage: List[str]
    total_files: int
    total_successful: int