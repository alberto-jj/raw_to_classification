"""Type definitions for the dataset2bids pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable


@dataclass
class BidsificationConfig:
    """Configuration for BIDS conversion"""
    source_path: Path
    bids_path: Path
    rules: Dict[str, Any]
    custom_bidsify_func: Optional[Callable] = None
    overwrite: bool = False


@dataclass
class DatasetBidsResult:
    """Result of BIDS conversion for a single dataset"""
    dataset_label: str
    source_path: Path
    bids_path: Path
    success: bool = True
    error: Optional[str] = None
    files_converted: int = 0
    conversion_summary: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class GlobalBidsResult:
    """Global result for all dataset BIDS conversions"""
    dataset_results: Dict[str, DatasetBidsResult]
    total_datasets: int = 0
    successful_datasets: int = 0
    total_files_converted: int = 0


@dataclass
class Dataset2BidsConfig:
    """Overall configuration for dataset2bids pipeline"""
    pipeline_file: Path
    output_format: str = "json"
    debug: bool = False
    log_dir: Path = Path("./logs")
    overwrite: bool = False
    dry_run: bool = False