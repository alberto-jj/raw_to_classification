"""Pure functions for BIDS conversion logic."""

import importlib
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple, Optional
import traceback

from .dataset2bids_types import BidsificationConfig, DatasetBidsResult, GlobalBidsResult


def validate_bids_config(config: dict) -> List[str]:
    """
    Validate dataset2bids configuration.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Check required top-level fields
    if 'project' not in config:
        errors.append("Missing required field 'project'")
    
    if 'datasets_file' not in config:
        errors.append("Missing required field 'datasets_file'")
    
    # Check 1_dataset2bids section if it exists
    if '1_dataset2bids' in config:
        bids_config = config['1_dataset2bids']
        
        # Validate redefine_bidsify if present
        if 'redefine_bidsify' in bids_config:
            redefine = bids_config['redefine_bidsify']
            if not isinstance(redefine, dict):
                errors.append("'redefine_bidsify' must be a dictionary")
            else:
                if 'from_this' not in redefine:
                    errors.append("Missing 'from_this' in redefine_bidsify")
                if 'import_that' not in redefine:
                    errors.append("Missing 'import_that' in redefine_bidsify")
    
    return errors


def setup_bidsify_function(config: dict) -> Callable:
    """
    Setup the bidsify function based on configuration.
    
    Args:
        config: Dataset2bids configuration
        
    Returns:
        Configured bidsify function
        
    Raises:
        ImportError: If custom module/function cannot be imported
    """
    if 'redefine_bidsify' in config:
        import_dict = config['redefine_bidsify']
        module_name = import_dict['from_this']
        func_name = import_dict['import_that']
        
        try:
            module = importlib.import_module(module_name)
            bidsify_func = getattr(module, func_name)
            return bidsify_func
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import {module_name}.{func_name}: {e}")
    else:
        # Import default function
        from . import dataset2bids
        return dataset2bids.sova_bidsify


def create_bidsification_config(
    dataset_label: str, 
    dataset_config: dict, 
    mount: Optional[str],
    path_resolver: Callable[[str, Optional[str]], str]
) -> Optional[BidsificationConfig]:
    """
    Create BidsificationConfig from dataset configuration.
    
    Args:
        dataset_label: Label for the dataset
        dataset_config: Dataset configuration dictionary
        mount: Mount point for path resolution
        path_resolver: Function to resolve paths
        
    Returns:
        BidsificationConfig or None if no bidsify config
    """
    bidsify_cfg = dataset_config.get('bidsify', None)
    
    if not bidsify_cfg:
        return None
    
    if 'paths' not in bidsify_cfg:
        raise ValueError(f"Missing 'paths' in bidsify config for dataset {dataset_label}")
    
    paths = bidsify_cfg['paths']
    if 'source_path' not in paths or 'bids_path' not in paths:
        raise ValueError(f"Missing source_path or bids_path for dataset {dataset_label}")
    
    source_path = Path(path_resolver(paths['source_path'], mount))
    bids_path = Path(path_resolver(paths['bids_path'], mount))
    rules = bidsify_cfg.get('rules', {})
    
    return BidsificationConfig(
        source_path=source_path,
        bids_path=bids_path,
        rules=rules,
        overwrite=bidsify_cfg.get('overwrite', False)
    )


def convert_dataset_to_bids(
    dataset_label: str,
    bidsify_config: BidsificationConfig,
    bidsify_func: Callable,
    dataset_config: dict,
    pipeline_config: dict,
    dry_run: bool = False
) -> DatasetBidsResult:
    """
    Convert a single dataset to BIDS format.
    
    Args:
        dataset_label: Label for the dataset
        bidsify_config: BIDS conversion configuration
        bidsify_func: Function to perform BIDS conversion
        dataset_config: Full dataset configuration
        pipeline_config: Full pipeline configuration
        dry_run: If True, don't actually convert files
        
    Returns:
        DatasetBidsResult with conversion outcome
    """
    try:
        if dry_run:
            # In dry run mode, just validate paths and return success
            if not bidsify_config.source_path.exists():
                raise FileNotFoundError(f"Source path does not exist: {bidsify_config.source_path}")
            
            return DatasetBidsResult(
                dataset_label=dataset_label,
                source_path=bidsify_config.source_path,
                bids_path=bidsify_config.bids_path,
                success=True,
                files_converted=0,  # Would need actual file counting in real implementation
                conversion_summary={'dry_run': True}
            )
        
        # Perform actual conversion
        result = bidsify_func(
            str(bidsify_config.source_path),
            str(bidsify_config.bids_path),
            dataset_config,
            pipeline_config
        )
        
        return DatasetBidsResult(
            dataset_label=dataset_label,
            source_path=bidsify_config.source_path,
            bids_path=bidsify_config.bids_path,
            success=True,
            files_converted=0,  # This would be filled by the actual bidsify function
            conversion_summary=result if isinstance(result, dict) else {}
        )
        
    except Exception as e:
        return DatasetBidsResult(
            dataset_label=dataset_label,
            source_path=bidsify_config.source_path,
            bids_path=bidsify_config.bids_path,
            success=False,
            error=str(e),
            files_converted=0,
            conversion_summary={'error_trace': traceback.format_exc()}
        )


def compute_global_bids_summary(dataset_results: Dict[str, DatasetBidsResult]) -> GlobalBidsResult:
    """
    Compute global summary of BIDS conversion results.
    
    Args:
        dataset_results: Dictionary of dataset conversion results
        
    Returns:
        GlobalBidsResult with aggregated statistics
    """
    successful_datasets = sum(1 for result in dataset_results.values() if result.success)
    total_files_converted = sum(result.files_converted for result in dataset_results.values())
    
    return GlobalBidsResult(
        dataset_results=dataset_results,
        total_datasets=len(dataset_results),
        successful_datasets=successful_datasets,
        total_files_converted=total_files_converted
    )