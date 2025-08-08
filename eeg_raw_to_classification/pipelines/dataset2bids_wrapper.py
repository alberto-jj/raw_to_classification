"""Wrapper classes for I/O orchestration in dataset2bids pipeline."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from ..utils import load_yaml, get_path
from ..loggers import StructuredLogger
from .dataset2bids_types import Dataset2BidsConfig, DatasetBidsResult, GlobalBidsResult
from .dataset2bids_core import (
    validate_bids_config,
    setup_bidsify_function,
    create_bidsification_config,
    convert_dataset_to_bids,
    compute_global_bids_summary
)
from .output_manager import InspectionOutputManager  # Reuse existing output manager


class Dataset2BidsOutputManager:
    """Manages output for dataset2bids results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset_results(self, result: DatasetBidsResult, format: str = "json"):
        """Save dataset BIDS conversion results"""
        output_file = self.output_dir / f"{result.dataset_label}_bids_conversion"
        
        data = {
            'dataset_label': result.dataset_label,
            'source_path': str(result.source_path),
            'bids_path': str(result.bids_path),
            'success': result.success,
            'error': result.error,
            'files_converted': result.files_converted,
            'conversion_summary': result.conversion_summary
        }
        
        if format in ["json", "both"]:
            import json
            with open(f"{output_file}.json", 'w') as f:
                json.dump(data, f, indent=2)
        
        if format in ["yaml", "both"]:
            import yaml
            with open(f"{output_file}.yaml", 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
    
    def save_global_results(self, result: GlobalBidsResult, format: str = "json"):
        """Save global BIDS conversion results"""
        output_file = self.output_dir / "global_bids_conversion"
        
        data = {
            'total_datasets': result.total_datasets,
            'successful_datasets': result.successful_datasets,
            'total_files_converted': result.total_files_converted,
            'success_rate': result.successful_datasets / result.total_datasets if result.total_datasets > 0 else 0,
            'dataset_summaries': {
                label: {
                    'success': ds_result.success,
                    'files_converted': ds_result.files_converted,
                    'error': ds_result.error
                }
                for label, ds_result in result.dataset_results.items()
            }
        }
        
        if format in ["json", "both"]:
            import json
            with open(f"{output_file}.json", 'w') as f:
                json.dump(data, f, indent=2)
        
        if format in ["yaml", "both"]:
            import yaml
            with open(f"{output_file}.yaml", 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)


class DatasetProcessor:
    """Handles processing of a single dataset for BIDS conversion"""
    
    def __init__(self, config: Dataset2BidsConfig, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        self.output_manager = Dataset2BidsOutputManager(config.log_dir / "bids_outputs")
    
    def process_dataset(
        self, 
        dataset_label: str, 
        dataset_config: dict, 
        bidsify_func: Callable,
        pipeline_config: dict,
        mount: Optional[str]
    ) -> DatasetBidsResult:
        """
        Process a single dataset for BIDS conversion.
        
        Args:
            dataset_label: Label for the dataset
            dataset_config: Dataset configuration
            bidsify_func: Function to perform BIDS conversion
            pipeline_config: Full pipeline configuration
            mount: Mount point for path resolution
            
        Returns:
            DatasetBidsResult with conversion outcome
        """
        with self.logger.timed_operation(f"dataset_bids_{dataset_label}", dataset=dataset_label):
            self.logger.logger.info(f"Starting BIDS conversion for dataset: {dataset_label}")
            
            try:
                # Create bidsification config
                bidsify_config = create_bidsification_config(
                    dataset_label, dataset_config, mount, get_path
                )
                
                if not bidsify_config:
                    self.logger.logger.info(f"No BIDS configuration for dataset {dataset_label}, skipping")
                    return DatasetBidsResult(
                        dataset_label=dataset_label,
                        source_path=Path(),
                        bids_path=Path(),
                        success=True,
                        files_converted=0,
                        conversion_summary={'skipped': 'no_bids_config'}
                    )
                
                self.logger.logger.info(
                    f"Converting {dataset_label}: {bidsify_config.source_path} -> {bidsify_config.bids_path}"
                )
                
                # Perform conversion using pure function
                result = convert_dataset_to_bids(
                    dataset_label,
                    bidsify_config,
                    bidsify_func,
                    dataset_config,
                    pipeline_config,
                    dry_run=self.config.dry_run
                )
                
                # Log metrics
                self.logger.log_metrics({
                    'success': result.success,
                    'files_converted': result.files_converted,
                    'source_path': str(result.source_path),
                    'bids_path': str(result.bids_path)
                }, context=f"dataset_bids_{dataset_label}")
                
                # Save results
                self.output_manager.save_dataset_results(result, self.config.output_format)
                
                if result.success:
                    self.logger.logger.info(
                        f"Successfully converted dataset {dataset_label} "
                        f"({result.files_converted} files)"
                    )
                else:
                    self.logger.logger.error(
                        f"Failed to convert dataset {dataset_label}: {result.error}"
                    )
                
                return result
                
            except Exception as e:
                self.logger.logger.error(f"Error processing dataset {dataset_label}: {e}", exc_info=True)
                return DatasetBidsResult(
                    dataset_label=dataset_label,
                    source_path=Path(),
                    bids_path=Path(),
                    success=False,
                    error=str(e),
                    files_converted=0
                )


class Dataset2BidsPipeline:
    """Main pipeline orchestrator for dataset2bids conversion"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def run_pipeline(
        self,
        pipeline_file: str,
        output_format: str = "json",
        dry_run: bool = False,
        overwrite: bool = False
    ) -> GlobalBidsResult:
        """
        Run the complete dataset2bids pipeline.
        
        Args:
            pipeline_file: Path to pipeline configuration
            output_format: Output format (json, yaml, both)
            dry_run: If True, validate but don't convert files
            overwrite: If True, overwrite existing BIDS data
            
        Returns:
            GlobalBidsResult with all conversion results
        """
        with self.logger.timed_operation("full_dataset2bids_pipeline", pipeline_file=pipeline_file):
            # Load and validate configuration
            cfg = self._load_and_validate_config(pipeline_file)
            
            # Setup configuration
            config = Dataset2BidsConfig(
                pipeline_file=Path(pipeline_file),
                output_format=output_format,
                debug=self.logger.debug,
                log_dir=self.logger.log_dir,
                dry_run=dry_run,
                overwrite=overwrite
            )
            
            # Load datasets
            mount = cfg.get('mount', None)
            datasets = load_yaml(get_path(cfg['datasets_file'], mount))
            
            # Setup bidsify function
            dataset2bids_config = cfg.get('1_dataset2bids', {})
            bidsify_func = setup_bidsify_function(dataset2bids_config)
            
            self.logger.logger.info(f"Using bidsify function: {bidsify_func.__name__}")
            
            # Process datasets
            dataset_results = {}
            processor = DatasetProcessor(config, self.logger)
            
            for dataset_label, dataset_config in datasets.items():
                if dataset_config.get('skip', False):
                    self.logger.logger.info(f"Skipping dataset {dataset_label} as requested")
                    continue
                
                result = processor.process_dataset(
                    dataset_label, dataset_config, bidsify_func, cfg, mount
                )
                dataset_results[dataset_label] = result
            
            # Compute global summary
            global_result = compute_global_bids_summary(dataset_results)
            
            # Save global results
            output_manager = Dataset2BidsOutputManager(config.log_dir / "bids_outputs")
            output_manager.save_global_results(global_result, output_format)
            
            # Log final metrics
            self.logger.log_metrics({
                'total_datasets': global_result.total_datasets,
                'successful_datasets': global_result.successful_datasets,
                'success_rate': global_result.successful_datasets / global_result.total_datasets if global_result.total_datasets > 0 else 0,
                'total_files_converted': global_result.total_files_converted
            }, context='global_bids_summary')
            
            self.logger.logger.info(
                f"BIDS conversion complete: {global_result.successful_datasets}/"
                f"{global_result.total_datasets} datasets successful, "
                f"{global_result.total_files_converted} total files converted"
            )
            
            return global_result
    
    def _load_and_validate_config(self, pipeline_file: str) -> dict:
        """Load and validate pipeline configuration"""
        cfg = load_yaml(pipeline_file)
        
        # Validate configuration
        errors = validate_bids_config(cfg)
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            self.logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return cfg