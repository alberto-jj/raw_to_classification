"""Wrapper classes for I/O orchestration in MEEG inspection"""

import os
import glob
import importlib
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from typing import List, Dict, Optional, Callable
import traceback

from ..utils import load_yaml, get_path, load_meeg, find_minimal_unique_root, save_figs_in_html, save_dict_to_json
from .inspection_types import (
    InspectionConfig, DatasetInfo, MEEGInspectionResult, 
    DatasetInspectionResult, GlobalInspectionResult
)
from .inspection_core import inspect_single_meeg, compute_dataset_summary, compute_global_summary, validate_inspection_config
from .output_manager import InspectionOutputManager


class ReportGenerator:
    """Handles MNE report generation with error handling"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def generate_report(self, meeg_data, file_path: Path, dataset_label: str, relative_path: str, output_dir: Path) -> bool:
        """Generate MNE report for a MEEG file"""
        try:
            output_base = output_dir / Path(relative_path).stem
            output_base.parent.mkdir(parents=True, exist_ok=True)
            
            report = mne.Report(title=f'Inspect {dataset_label} {relative_path}', verbose='error')
            
            if isinstance(meeg_data, mne.io.BaseRaw):
                report.add_raw(meeg_data, title=f'{dataset_label} {relative_path}')
            elif isinstance(meeg_data, mne.BaseEpochs):
                report.add_epochs(meeg_data, title=f'{dataset_label} {relative_path}')
            
            # Add spectrum to report
            report.add_figure(meeg_data.plot_psd(show=False), title=f'{dataset_label} {relative_path} Spectrum')
            
            fmax = meeg_data.info['sfreq'] / 2
            if fmax > 200:
                fmax = 200
            report.add_figure(
                meeg_data.plot_psd(show=False, fmax=fmax), 
                title=f'{dataset_label} {relative_path} Spectrum Below 200Hz'
            )
            
            report.save(str(output_base) + '_report.html', overwrite=True, verbose='error', open_browser=False)
            del report
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating report for {file_path}: {e}")
            save_dict_to_json(str(output_base) + '_problemReport.txt', {'problem': str(e)})
            return False
    
    def generate_spectrum(self, meeg_data, file_path: Path, relative_path: str, output_dir: Path) -> bool:
        """Generate standalone spectrum plot"""
        try:
            output_base = output_dir / Path(relative_path).stem
            output_base.parent.mkdir(parents=True, exist_ok=True)
            
            fig = meeg_data.plot_psd(show=False)
            save_figs_in_html(str(output_base) + '_spectrum.html', [fig])
            plt.close('all')
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating spectrum for {file_path}: {e}")
            return False


class DatasetProcessor:
    """Handles processing of a single dataset"""
    
    def __init__(self, config: InspectionConfig, logger):
        self.config = config
        self.logger = logger
        self.report_generator = ReportGenerator(logger)
        self.output_manager = InspectionOutputManager(config.output_path)
    
    def process_dataset(self, dataset_info: DatasetInfo, loader_func: Callable) -> DatasetInspectionResult:
        """
        Process a complete dataset with logging and error handling.
        
        Args:
            dataset_info: Dataset configuration
            loader_func: Function to load MEEG files
            
        Returns:
            DatasetInspectionResult with all inspection data
        """
        with self.logger.timed_operation(f"dataset_{dataset_info.label}", dataset=dataset_info.label):
            self.logger.logger.info(f"Starting inspection of dataset: {dataset_info.label}")
            
            # Determine files to process
            files_to_process = dataset_info.files[:self.config.max_files] if self.config.max_files else dataset_info.files
            total_files = len(files_to_process)
            
            if total_files == 0:
                self.logger.logger.warning(f"No files found for dataset {dataset_info.label}")
                return DatasetInspectionResult(
                    dataset_label=dataset_info.label,
                    file_results=[],
                    common_montage=[],
                    union_montage=[],
                    duration_stats={},
                    duration_counts={},
                    success_count=0,
                    total_count=0
                )
            
            # Find minimal root for relative paths
            minimal_root = find_minimal_unique_root([str(f) for f in files_to_process])
            self.logger.logger.info(f"[{dataset_info.label}] Minimal root for unique paths: {minimal_root}")
            
            file_results = []
            
            # Process each file
            for i, file_path in enumerate(files_to_process):
                if i % max(1, total_files // 10) == 0:  # Log progress every 10%
                    self.logger.log_progress(i + 1, total_files, "files")
                
                relative_path = os.path.relpath(str(file_path), minimal_root)
                output_dir = self.config.output_path / dataset_info.label
                
                result = self._process_single_file(
                    file_path, dataset_info, loader_func, relative_path, output_dir
                )
                file_results.append(result)
                
                if result.success:
                    self.logger.logger.debug(f"Successfully processed {relative_path}")
                else:
                    self.logger.logger.warning(f"Failed to process {relative_path}: {result.error}")
            
            # Compute dataset summary using pure function
            dataset_result = compute_dataset_summary(dataset_info.label, file_results)
            
            # Log metrics
            self.logger.log_metrics({
                'total_files': dataset_result.total_count,
                'successful_files': dataset_result.success_count,
                'success_rate': dataset_result.success_count / dataset_result.total_count if dataset_result.total_count > 0 else 0,
                'common_channels': len(dataset_result.common_montage),
                'total_channels': len(dataset_result.union_montage)
            }, context=f"dataset_{dataset_info.label}")
            
            # Save results
            self.output_manager.save_dataset_results(dataset_result, self.config.output_format)
            self.output_manager.save_auxiliary_files(dataset_result)
            
            self.logger.logger.info(
                f"Completed inspection of dataset: {dataset_info.label} "
                f"({dataset_result.success_count}/{dataset_result.total_count} successful)"
            )
            
            return dataset_result
    
    def _process_single_file(
        self, file_path: Path, dataset_info: DatasetInfo, 
        loader_func: Callable, relative_path: str, output_dir: Path
    ) -> MEEGInspectionResult:
        """Process a single MEEG file with error handling"""
        try:
            # Load MEEG data
            meeg_data = loader_func(file_path, dataset_info.dataset_config, {'preload': True, 'verbose': 'error'})
            
            # Generate outputs if requested
            if self.config.generate_reports:
                self.report_generator.generate_report(meeg_data, file_path, dataset_info.label, relative_path, output_dir)
            
            if self.config.generate_spectra:
                self.report_generator.generate_spectrum(meeg_data, file_path, relative_path, output_dir)
            
            # Call pure inspection function
            result = inspect_single_meeg(meeg_data, file_path, dataset_info.label, relative_path)
            return result
            
        except Exception as e:
            self.logger.logger.error(f"Error processing {file_path}: {e}", exc_info=True)
            
            # Save error details
            output_base = output_dir / Path(relative_path).stem
            output_base.parent.mkdir(parents=True, exist_ok=True)
            save_dict_to_json(str(output_base) + '_problem.txt', {'problem': traceback.format_exc()})
            
            return MEEGInspectionResult(
                file_path=file_path,
                relative_path=relative_path,
                montage=[], duration=0.0, shape=(0,), 
                sampling_freq=0.0, info_str="",
                success=False, error=str(e)
            )


class MEEGInspectionPipeline:
    """Main pipeline orchestrator class"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def run_pipeline(
        self, 
        pipeline_file: str, 
        max_files: Optional[int] = None,
        output_format: str = "json"
    ) -> GlobalInspectionResult:
        """
        Run the complete inspection pipeline.
        
        Args:
            pipeline_file: Path to pipeline configuration
            max_files: Maximum files to process per dataset
            output_format: Output format (json, yaml, both)
            
        Returns:
            GlobalInspectionResult with all inspection data
        """
        with self.logger.timed_operation("full_inspection_pipeline", pipeline_file=pipeline_file):
            # Load and validate configuration
            cfg = self._load_and_validate_config(pipeline_file)
            
            # Setup paths and configuration
            MOUNT = cfg.get('mount', None)
            PROJECT = cfg['project']
            datasets = load_yaml(get_path(cfg['datasets_file'], MOUNT))
            
            inspect_path = get_path(cfg['0_inspect']['path'], MOUNT).replace('%PROJECT%', PROJECT)
            inspect_path = Path(inspect_path).expanduser()
            
            config = InspectionConfig(
                output_path=inspect_path,
                max_files=max_files,
                generate_reports=True,
                generate_spectra=True,
                output_format=output_format
            )
            
            # Setup loader function
            loader_func = self._setup_loader(cfg.get('0_inspect', {}))
            
            # Process datasets
            dataset_results = {}
            all_montages_for_global = []
            
            for dataset_label, dataset_config in datasets.items():
                if dataset_config.get('skip', False):
                    self.logger.logger.info(f"Skipping dataset {dataset_label} as requested")
                    continue
                
                # Create dataset info
                dataset_info = self._create_dataset_info(dataset_label, dataset_config, MOUNT)
                
                # Process dataset
                processor = DatasetProcessor(config, self.logger)
                result = processor.process_dataset(dataset_info, loader_func)
                
                dataset_results[dataset_label] = result
                
                # Collect montages for global analysis
                successful_files = [r for r in result.file_results if r.success and r.montage]
                all_montages_for_global.extend([r.montage for r in successful_files])
            
            # Compute global summary using pure function
            global_common, global_union, total_files, total_successful = compute_global_summary(dataset_results)
            
            global_result = GlobalInspectionResult(
                dataset_results=dataset_results,
                global_common_montage=global_common,
                global_union_montage=global_union,
                total_files=total_files,
                total_successful=total_successful
            )
            
            # Save global results
            output_manager = InspectionOutputManager(config.output_path)
            output_manager.save_global_results(global_result, output_format)
            output_manager.save_global_auxiliary_files(global_result)
            
            # Log final metrics
            self.logger.log_metrics({
                'total_datasets': len(dataset_results),
                'total_files': total_files,
                'total_successful': total_successful,
                'global_success_rate': total_successful / total_files if total_files > 0 else 0,
                'global_common_channels': len(global_common),
                'global_total_channels': len(global_union)
            }, context='global_summary')
            
            return global_result
    
    def _load_and_validate_config(self, pipeline_file: str) -> dict:
        """Load and validate pipeline configuration"""
        cfg = load_yaml(pipeline_file)
        
        # Validate configuration
        errors = validate_inspection_config(cfg)
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            self.logger.logger.error(error_msg)
            raise ValueError(error_msg)
        
        return cfg
    
    def _setup_loader(self, inspect_config: dict) -> Callable:
        """Setup the MEEG loader function"""
        if 'redefine_loader' in inspect_config:
            import_dict = inspect_config['redefine_loader']
            module_name = import_dict['from_this']
            func_name = import_dict['import_that']
            module = importlib.import_module(module_name)
            loader_func = getattr(module, func_name)
            self.logger.logger.info(f"Using custom loader: {module_name}.{func_name}")
        else:
            loader_func = load_meeg
            self.logger.logger.info("Using default MEEG loader")
        
        return loader_func
    
    def _create_dataset_info(self, dataset_label: str, dataset_config: dict, mount: Optional[str]) -> DatasetInfo:
        """Create DatasetInfo from configuration"""
        exemplar_file = get_path(dataset_config['example_file'], mount)
        exemplar_file = Path(exemplar_file).expanduser()
        
        files = [Path(f) for f in glob.glob(str(exemplar_file), recursive=True)]
        
        return DatasetInfo(
            label=dataset_label,
            files=files,
            example_file=exemplar_file,
            skip=dataset_config.get('skip', False),
            dataset_config=dataset_config
        )