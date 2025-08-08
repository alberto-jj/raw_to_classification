"""Pure functions for MEEG inspection (no I/O dependencies)"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from .inspection_types import MEEGInspectionResult, DatasetInspectionResult


def inspect_single_meeg(
    meeg_data,
    file_path: Path,
    dataset_label: str,
    relative_path: str = ""
) -> MEEGInspectionResult:
    """
    Pure function to inspect a single MEEG file.
    
    Args:
        meeg_data: Loaded MNE object (Raw or Epochs)
        file_path: Path to the file
        dataset_label: Dataset identifier
        relative_path: Relative path from dataset root
        
    Returns:
        MEEGInspectionResult with inspection data
    """
    try:
        return MEEGInspectionResult(
            file_path=file_path,
            relative_path=relative_path,
            montage=meeg_data.ch_names,
            duration=meeg_data.times[-1] if len(meeg_data.times) > 0 else 0.0,
            shape=meeg_data.get_data().shape,
            sampling_freq=meeg_data.info['sfreq'],
            info_str=str(meeg_data.info),
            success=True
        )
    except Exception as e:
        return MEEGInspectionResult(
            file_path=file_path,
            relative_path=relative_path,
            montage=[],
            duration=0.0,
            shape=(0,),
            sampling_freq=0.0,
            info_str="",
            success=False,
            error=str(e)
        )


def compute_montage_analysis(file_results: List[MEEGInspectionResult]) -> Tuple[List[str], List[str]]:
    """
    Pure function to compute montage intersections and unions.
    
    Args:
        file_results: List of individual file results
        
    Returns:
        Tuple of (common_montage, union_montage)
    """
    successful_results = [r for r in file_results if r.success and r.montage]
    
    if not successful_results:
        return [], []
    
    # Compute montage intersections/unions
    montages = [set(r.montage) for r in successful_results]
    common_montage = list(set.intersection(*montages)) if montages else []
    union_montage = list(set.union(*montages)) if montages else []
    
    return common_montage, union_montage


def compute_duration_analysis(file_results: List[MEEGInspectionResult]) -> Tuple[Dict[str, float], Dict[float, int]]:
    """
    Pure function to compute duration statistics and counts.
    
    Args:
        file_results: List of individual file results
        
    Returns:
        Tuple of (duration_stats, duration_counts)
    """
    successful_results = [r for r in file_results if r.success]
    
    if not successful_results:
        return {}, {}
    
    # Compute duration statistics
    durations = [r.duration for r in successful_results]
    duration_stats = {
        'mean': float(np.mean(durations)),
        'median': float(np.median(durations)),
        'std': float(np.std(durations)),
        'min': float(np.min(durations)),
        'max': float(np.max(durations))
    }
    
    # Duration counts
    unique_durations, counts = np.unique(durations, return_counts=True)
    duration_counts = {float(dur): int(count) for dur, count in zip(unique_durations, counts)}
    
    return duration_stats, duration_counts


def compute_dataset_summary(
    dataset_label: str,
    file_results: List[MEEGInspectionResult]
) -> DatasetInspectionResult:
    """
    Pure function to compute dataset-level summaries.
    
    Args:
        dataset_label: Label for the dataset
        file_results: List of individual file results
        
    Returns:
        DatasetInspectionResult with complete dataset summary
    """
    # Compute montage analysis
    common_montage, union_montage = compute_montage_analysis(file_results)
    
    # Compute duration analysis  
    duration_stats, duration_counts = compute_duration_analysis(file_results)
    
    # Count successes
    success_count = len([r for r in file_results if r.success])
    total_count = len(file_results)
    
    return DatasetInspectionResult(
        dataset_label=dataset_label,
        file_results=file_results,
        common_montage=common_montage,
        union_montage=union_montage,
        duration_stats=duration_stats,
        duration_counts=duration_counts,
        success_count=success_count,
        total_count=total_count
    )


def compute_global_summary(
    dataset_results: Dict[str, DatasetInspectionResult]
) -> Tuple[List[str], List[str], int, int]:
    """
    Pure function to compute global cross-dataset summaries.
    
    Args:
        dataset_results: Dictionary of dataset results
        
    Returns:
        Tuple of (global_common_montage, global_union_montage, total_files, total_successful)
    """
    if not dataset_results:
        return [], [], 0, 0
    
    # Aggregate all successful file results
    all_montages = []
    total_files = 0
    total_successful = 0
    
    for dataset_result in dataset_results.values():
        total_files += dataset_result.total_count
        total_successful += dataset_result.success_count
        
        successful_files = [r for r in dataset_result.file_results if r.success and r.montage]
        all_montages.extend([set(r.montage) for r in successful_files])
    
    # Compute global montage analysis
    if all_montages:
        global_common_montage = list(set.intersection(*all_montages))
        global_union_montage = list(set.union(*all_montages))
    else:
        global_common_montage = []
        global_union_montage = []
    
    return global_common_montage, global_union_montage, total_files, total_successful


def validate_inspection_config(config: dict) -> List[str]:
    """
    Pure function to validate inspection configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if 'project' not in config:
        errors.append("Missing 'project' in configuration")
    
    if 'datasets_file' not in config:
        errors.append("Missing 'datasets_file' in configuration")
    
    if '0_inspect' not in config:
        errors.append("Missing '0_inspect' section in configuration")
    else:
        inspect_config = config['0_inspect']
        if 'path' not in inspect_config:
            errors.append("Missing 'path' in '0_inspect' configuration")
    
    return errors