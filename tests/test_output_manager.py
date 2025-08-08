"""
Tests for the output manager functionality.

These tests validate output formatting and file generation.
"""

import pytest
import json
import yaml
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from eeg_raw_to_classification.pipelines.inspection_types import (
    MEEGInspectionResult, DatasetInspectionResult, GlobalInspectionResult
)
from eeg_raw_to_classification.pipelines.output_manager import InspectionOutputManager


class TestInspectionOutputManager:
    """Test the output manager functionality"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_dataset_result(self):
        """Create sample dataset result for testing"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"),
                relative_path="sub-01/test1.fif", 
                montage=['Fp1', 'Fp2'],
                duration=5.0,
                shape=(2, 1000),
                sampling_freq=250.0,
                info_str="Mock info",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test2.fif"),
                relative_path="sub-02/test2.fif",
                montage=['Fp1', 'C3'],
                duration=10.0,
                shape=(2, 2000), 
                sampling_freq=250.0,
                info_str="Mock info 2",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test3.fif"),
                relative_path="sub-03/test3.fif",
                montage=[],
                duration=0.0,
                shape=(0,),
                sampling_freq=0.0,
                info_str="",
                success=False,
                error="Load failed"
            ),
        ]
        
        return DatasetInspectionResult(
            dataset_label="test_dataset",
            file_results=file_results,
            common_montage=['Fp1'],
            union_montage=['Fp1', 'Fp2', 'C3'],
            duration_stats={'mean': 7.5, 'min': 5.0, 'max': 10.0, 'std': 2.5, 'median': 7.5},
            duration_counts={5.0: 1, 10.0: 1},
            success_count=2,
            total_count=3
        )
    
    def test_save_dataset_results_json(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results as JSON"""
        manager = InspectionOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="json")
        
        # Check JSON file exists
        json_file = temp_output_dir / "test_dataset" / "test_dataset_inspection.json"
        assert json_file.exists()
        
        # Check JSON content
        with open(json_file) as f:
            data = json.load(f)
        
        assert data['dataset_label'] == 'test_dataset'
        assert data['summary']['total_files'] == 3
        assert data['summary']['successful_files'] == 2
        assert data['summary']['success_rate'] == pytest.approx(2/3, rel=1e-2)
        assert len(data['files']) == 3
        
        # Check CSV file also exists (always generated)
        csv_file = temp_output_dir / "test_dataset" / "test_dataset_inspect.csv"
        assert csv_file.exists()
    
    def test_save_dataset_results_yaml(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results as YAML"""
        manager = InspectionOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="yaml")
        
        # Check YAML file exists
        yaml_file = temp_output_dir / "test_dataset" / "test_dataset_summary.yaml"
        assert yaml_file.exists()
        
        # Check YAML content
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        
        summary = data['dataset_inspection_summary']
        assert summary['dataset'] == 'test_dataset'
        assert summary['file_counts']['total'] == 3
        assert summary['file_counts']['successful'] == 2
        assert summary['file_counts']['failed'] == 1
        assert len(summary['failed_files']) == 1
        assert summary['failed_files'][0]['error'] == "Load failed"
    
    def test_save_dataset_results_both(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results in both formats"""
        manager = InspectionOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="both")
        
        # Check both files exist
        json_file = temp_output_dir / "test_dataset" / "test_dataset_inspection.json"
        yaml_file = temp_output_dir / "test_dataset" / "test_dataset_summary.yaml"
        csv_file = temp_output_dir / "test_dataset" / "test_dataset_inspect.csv"
        
        assert json_file.exists()
        assert yaml_file.exists()
        assert csv_file.exists()
    
    def test_save_auxiliary_files(self, temp_output_dir, sample_dataset_result):
        """Test saving auxiliary files for backward compatibility"""
        manager = InspectionOutputManager(temp_output_dir)
        manager.save_auxiliary_files(sample_dataset_result)
        
        dataset_dir = temp_output_dir / "test_dataset"
        
        # Check all auxiliary files exist
        aux_files = [
            'common_montage.txt',
            'union_montage.txt',
            'times_stats.txt',
            'times_counts.txt',
            'times.txt',
            'shapes.txt',
            'sfreqs.txt',
            'infos.txt'
        ]
        
        for aux_file in aux_files:
            file_path = dataset_dir / aux_file
            assert file_path.exists(), f"Auxiliary file {aux_file} not found"
            
            # Check file contains valid JSON
            with open(file_path) as f:
                data = json.load(f)
                assert isinstance(data, dict)
    
    def test_save_global_results(self, temp_output_dir, sample_dataset_result):
        """Test saving global results"""
        global_result = GlobalInspectionResult(
            dataset_results={'test_dataset': sample_dataset_result},
            global_common_montage=['Fp1'],
            global_union_montage=['Fp1', 'Fp2', 'C3'],
            total_files=3,
            total_successful=2
        )
        
        manager = InspectionOutputManager(temp_output_dir)
        manager.save_global_results(global_result, format="both")
        
        # Check global files exist
        json_file = temp_output_dir / "global_inspection_results.json"
        yaml_file = temp_output_dir / "global_inspection_summary.yaml"
        
        assert json_file.exists()
        assert yaml_file.exists()
        
        # Check JSON content
        with open(json_file) as f:
            data = json.load(f)
        
        assert data['global_summary']['total_datasets'] == 1
        assert data['global_summary']['total_files'] == 3
        assert data['global_summary']['total_successful'] == 2
        
        # Check YAML content  
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        
        summary = data['global_inspection_summary']
        assert summary['overview']['total_datasets'] == 1
        assert summary['overview']['total_files'] == 3
    
    def test_invalid_format_raises_error(self, temp_output_dir, sample_dataset_result):
        """Test that invalid format raises error"""
        manager = InspectionOutputManager(temp_output_dir)
        
        with pytest.raises(ValueError, match="Unsupported format"):
            manager.save_dataset_results(sample_dataset_result, format="invalid")


if __name__ == "__main__":
    pytest.main([__file__])