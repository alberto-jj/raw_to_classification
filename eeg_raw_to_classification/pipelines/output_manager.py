"""Output management for inspection results with JSON/YAML support"""

import json
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from .inspection_types import DatasetInspectionResult, GlobalInspectionResult


class InspectionOutputManager:
    """Handles different output formats for inspection results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset_results(self, result: DatasetInspectionResult, format: str = "json"):
        """Save dataset inspection results"""
        
        if format == "json":
            self._save_dataset_as_json(result)
        elif format == "yaml":  
            self._save_dataset_as_yaml(result)
        elif format == "both":
            self._save_dataset_as_json(result)
            self._save_dataset_as_yaml(result)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Always save CSV for compatibility with existing workflows
        self._save_dataset_as_csv(result)
    
    def save_global_results(self, result: GlobalInspectionResult, format: str = "json"):
        """Save global cross-dataset results"""
        
        if format == "json":
            self._save_global_as_json(result)
        elif format == "yaml":
            self._save_global_as_yaml(result)
        elif format == "both":
            self._save_global_as_json(result)
            self._save_global_as_yaml(result)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_dataset_as_json(self, result: DatasetInspectionResult):
        """Save dataset results as JSON (recommended for data)"""
        output_file = self.output_dir / result.dataset_label / f"{result.dataset_label}_inspection.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict for JSON serialization
        data = {
            'dataset_label': result.dataset_label,
            'processed_at': datetime.now().isoformat(),
            'summary': {
                'total_files': result.total_count,
                'successful_files': result.success_count,
                'failed_files': result.total_count - result.success_count,
                'success_rate': result.success_count / result.total_count if result.total_count > 0 else 0,
                'common_montage': result.common_montage,
                'union_montage': result.union_montage,
                'common_channel_count': len(result.common_montage),
                'total_unique_channels': len(result.union_montage),
                'duration_stats': result.duration_stats,
                'duration_counts': result.duration_counts
            },
            'files': [
                {
                    'file_path': str(r.file_path),
                    'relative_path': r.relative_path,
                    'montage': r.montage,
                    'montage_size': len(r.montage),
                    'duration': r.duration,
                    'shape': r.shape,
                    'sampling_freq': r.sampling_freq,
                    'success': r.success,
                    'error': r.error
                } 
                for r in result.file_results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_dataset_as_yaml(self, result: DatasetInspectionResult):
        """Save dataset results as YAML (better for human-readable summaries)"""
        output_file = self.output_dir / result.dataset_label / f"{result.dataset_label}_summary.yaml"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create human-readable summary
        failed_files = [r for r in result.file_results if not r.success]
        
        summary = {
            'dataset_inspection_summary': {
                'dataset': result.dataset_label,
                'processed_at': datetime.now().isoformat(),
                'file_counts': {
                    'total': result.total_count,
                    'successful': result.success_count, 
                    'failed': len(failed_files),
                    'success_rate_percent': round((result.success_count / result.total_count * 100), 1) if result.total_count > 0 else 0
                },
                'montage_analysis': {
                    'common_channels': result.common_montage,
                    'all_channels_union': result.union_montage,
                    'common_channel_count': len(result.common_montage),
                    'total_unique_channels': len(result.union_montage),
                    'channel_coverage_percent': round((len(result.common_montage) / len(result.union_montage) * 100), 1) if result.union_montage else 0
                },
                'duration_analysis': {
                    **result.duration_stats,
                    'unique_durations': len(result.duration_counts)
                },
                'failed_files': [
                    {
                        'file': str(r.file_path),
                        'error': r.error
                    } 
                    for r in failed_files
                ] if failed_files else []
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    def _save_dataset_as_csv(self, result: DatasetInspectionResult):
        """Save dataset results as CSV for compatibility with existing workflows"""
        output_file = self.output_dir / result.dataset_label / f"{result.dataset_label}_inspect.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to pandas DataFrame (similar to original format)
        df_data = []
        for r in result.file_results:
            df_data.append({
                'MEEG': str(r.file_path),
                'relative_path': r.relative_path,
                'montage': r.montage,
                'montage_size': len(r.montage),
                'times': r.duration,
                'shape': r.shape,
                'sfreq': r.sampling_freq,
                'info': r.info_str,
                'success': r.success,
                'error': r.error or ""
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False)
    
    def _save_global_as_json(self, result: GlobalInspectionResult):
        """Save global results as JSON"""
        output_file = self.output_dir / "global_inspection_results.json"
        
        data = {
            'global_summary': {
                'processed_at': datetime.now().isoformat(),
                'total_datasets': len(result.dataset_results),
                'total_files': result.total_files,
                'total_successful': result.total_successful,
                'global_success_rate': result.total_successful / result.total_files if result.total_files > 0 else 0,
                'global_common_montage': result.global_common_montage,
                'global_union_montage': result.global_union_montage,
                'global_common_channels': len(result.global_common_montage),
                'global_total_channels': len(result.global_union_montage)
            },
            'datasets': {
                label: {
                    'total_files': ds.total_count,
                    'successful_files': ds.success_count,
                    'success_rate': ds.success_count / ds.total_count if ds.total_count > 0 else 0,
                    'common_channels': len(ds.common_montage),
                    'total_channels': len(ds.union_montage)
                }
                for label, ds in result.dataset_results.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_global_as_yaml(self, result: GlobalInspectionResult):
        """Save global results as YAML"""
        output_file = self.output_dir / "global_inspection_summary.yaml"
        
        summary = {
            'global_inspection_summary': {
                'processed_at': datetime.now().isoformat(),
                'overview': {
                    'total_datasets': len(result.dataset_results),
                    'total_files': result.total_files,
                    'total_successful': result.total_successful,
                    'global_success_rate_percent': round((result.total_successful / result.total_files * 100), 1) if result.total_files > 0 else 0
                },
                'global_montage_analysis': {
                    'common_channels_across_all': result.global_common_montage,
                    'all_channels_union': result.global_union_montage,
                    'common_channel_count': len(result.global_common_montage),
                    'total_unique_channels': len(result.global_union_montage),
                    'global_coverage_percent': round((len(result.global_common_montage) / len(result.global_union_montage) * 100), 1) if result.global_union_montage else 0
                },
                'dataset_summaries': {
                    label: {
                        'files': f"{ds.success_count}/{ds.total_count}",
                        'success_rate_percent': round((ds.success_count / ds.total_count * 100), 1) if ds.total_count > 0 else 0,
                        'channels': f"{len(ds.common_montage)}/{len(ds.union_montage)}"
                    }
                    for label, ds in result.dataset_results.items()
                }
            }
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    def save_auxiliary_files(self, result: DatasetInspectionResult):
        """Save auxiliary files for backward compatibility"""
        dataset_dir = self.output_dir / result.dataset_label
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Common montage
        with open(dataset_dir / 'common_montage.txt', 'w') as f:
            json.dump({'common_montage': result.common_montage}, f, indent=2)
        
        # Union montage  
        with open(dataset_dir / 'union_montage.txt', 'w') as f:
            json.dump({'union_montage': result.union_montage}, f, indent=2)
        
        # Duration stats
        with open(dataset_dir / 'times_stats.txt', 'w') as f:
            json.dump(result.duration_stats, f, indent=2)
        
        # Duration counts
        with open(dataset_dir / 'times_counts.txt', 'w') as f:
            json.dump({'counts': result.duration_counts}, f, indent=2)
        
        # All durations
        durations = [r.duration for r in result.file_results if r.success]
        with open(dataset_dir / 'times.txt', 'w') as f:
            json.dump({'times': durations}, f, indent=2)
        
        # All shapes
        shapes = [list(r.shape) for r in result.file_results if r.success]
        with open(dataset_dir / 'shapes.txt', 'w') as f:
            json.dump({'shapes': shapes}, f, indent=2)
        
        # All sampling frequencies
        sfreqs = [r.sampling_freq for r in result.file_results if r.success]
        with open(dataset_dir / 'sfreqs.txt', 'w') as f:
            json.dump({'sfreqs': sfreqs}, f, indent=2)
        
        # All info strings
        infos = [r.info_str for r in result.file_results if r.success]
        with open(dataset_dir / 'infos.txt', 'w') as f:
            json.dump({'infos': infos}, f, indent=2)
    
    def save_global_auxiliary_files(self, result: GlobalInspectionResult):
        """Save global auxiliary files for backward compatibility"""
        # Global common montage
        with open(self.output_dir / 'common_montage.txt', 'w') as f:
            json.dump({'common_montage': result.global_common_montage}, f, indent=2)
        
        # Global union montage  
        with open(self.output_dir / 'union_montage.txt', 'w') as f:
            json.dump({'union_montage': result.global_union_montage}, f, indent=2)