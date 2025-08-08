"""
Tests for the pure functions in the inspection pipeline.

These tests validate the business logic without I/O dependencies.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, PropertyMock

from eeg_raw_to_classification.pipelines.inspection_types import MEEGInspectionResult
from eeg_raw_to_classification.pipelines.inspection_core import (
    inspect_single_meeg,
    compute_montage_analysis,
    compute_duration_analysis, 
    compute_dataset_summary,
    compute_global_summary,
    validate_inspection_config
)


class TestInspectSingleMEEG:
    """Test the inspect_single_meeg pure function"""
    
    def test_successful_inspection(self):
        """Test successful MEEG inspection"""
        # Mock MEEG data
        mock_meeg = Mock()
        mock_meeg.ch_names = ['Fp1', 'Fp2', 'C3', 'C4']
        mock_meeg.times = np.array([0, 1, 2, 3, 4])
        mock_meeg.get_data.return_value = np.random.rand(4, 5)
        mock_meeg.info = {'sfreq': 250.0}
        
        result = inspect_single_meeg(
            mock_meeg,
            Path("/test/file.fif"),
            "test_dataset",  # dataset_label is passed but not stored in result
            "relative/path.fif"
        )
        
        assert result.success is True
        assert result.file_path == Path("/test/file.fif")
        assert result.relative_path == "relative/path.fif"
        assert result.montage == ['Fp1', 'Fp2', 'C3', 'C4']
        assert result.duration == 4.0
        assert result.shape == (4, 5)
        assert result.sampling_freq == 250.0
        assert result.error is None
    
    def test_failed_inspection(self):
        """Test failed MEEG inspection with error"""
        # Mock MEEG data that raises error when accessing times
        mock_meeg = Mock()
        mock_meeg.ch_names = ['Fp1', 'Fp2']
        # Make times property raise an exception
        type(mock_meeg).times = PropertyMock(side_effect=Exception("Test error"))
        
        result = inspect_single_meeg(
            mock_meeg,
            Path("/test/file.fif"),
            "test_dataset"
        )
        
        assert result.success is False
        assert result.error is not None
        assert "Test error" in result.error
        assert result.montage == []
        assert result.duration == 0.0


class TestMontageAnalysis:
    """Test montage analysis functions"""
    
    def test_compute_montage_analysis_success(self):
        """Test montage analysis with successful files"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"), montage=['Fp1', 'Fp2', 'C3'],
                duration=5.0, shape=(3, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test2.fif"), montage=['Fp1', 'C3', 'C4'], 
                duration=4.0, shape=(3, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test3.fif"), montage=['Fp1', 'Fp2', 'C3', 'C4'],
                duration=6.0, shape=(4, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
        ]
        
        common, union = compute_montage_analysis(file_results)
        
        assert set(common) == {'Fp1', 'C3'}  # Common across all
        assert set(union) == {'Fp1', 'Fp2', 'C3', 'C4'}  # Union of all
    
    def test_compute_montage_analysis_with_failures(self):
        """Test montage analysis with some failed files"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"), montage=['Fp1', 'Fp2'],
                duration=5.0, shape=(2, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test2.fif"), montage=[],
                duration=0.0, shape=(0,), sampling_freq=0.0, info_str="",
                success=False, error="Load failed"
            ),
        ]
        
        common, union = compute_montage_analysis(file_results)
        
        # Should only consider successful files
        assert set(common) == {'Fp1', 'Fp2'}
        assert set(union) == {'Fp1', 'Fp2'}
    
    def test_compute_montage_analysis_empty(self):
        """Test montage analysis with no successful files"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"), montage=[],
                duration=0.0, shape=(0,), sampling_freq=0.0, info_str="",
                success=False, error="Load failed"
            ),
        ]
        
        common, union = compute_montage_analysis(file_results)
        
        assert common == []
        assert union == []


class TestDurationAnalysis:
    """Test duration analysis functions"""
    
    def test_compute_duration_analysis(self):
        """Test duration statistics computation"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"), montage=['Fp1'], 
                duration=5.0, shape=(1, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test2.fif"), montage=['Fp1'],
                duration=10.0, shape=(1, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test3.fif"), montage=['Fp1'],
                duration=5.0, shape=(1, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
        ]
        
        stats, counts = compute_duration_analysis(file_results)
        
        assert stats['mean'] == pytest.approx(6.67, rel=1e-2)
        assert stats['min'] == 5.0
        assert stats['max'] == 10.0
        assert stats['median'] == 5.0
        
        assert counts[5.0] == 2
        assert counts[10.0] == 1


class TestDatasetSummary:
    """Test dataset summary computation"""
    
    def test_compute_dataset_summary(self):
        """Test complete dataset summary computation"""
        file_results = [
            MEEGInspectionResult(
                file_path=Path("/test1.fif"), montage=['Fp1', 'C3'],
                duration=5.0, shape=(2, 100), sampling_freq=250.0, info_str="",
                success=True
            ),
            MEEGInspectionResult(
                file_path=Path("/test2.fif"), montage=['Fp1', 'C4'],
                duration=10.0, shape=(2, 100), sampling_freq=250.0, info_str="",
                success=True  
            ),
            MEEGInspectionResult(
                file_path=Path("/test3.fif"), montage=[],
                duration=0.0, shape=(0,), sampling_freq=0.0, info_str="",
                success=False, error="Failed"
            ),
        ]
        
        result = compute_dataset_summary("test_dataset", file_results)
        
        assert result.dataset_label == "test_dataset"
        assert result.total_count == 3
        assert result.success_count == 2
        assert set(result.common_montage) == {'Fp1'}
        assert set(result.union_montage) == {'Fp1', 'C3', 'C4'}
        assert len(result.duration_stats) > 0
        assert len(result.duration_counts) > 0


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test validation of valid configuration"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml',
            '0_inspect': {
                'path': './inspect'
            }
        }
        
        errors = validate_inspection_config(config)
        assert len(errors) == 0
    
    def test_invalid_config_missing_fields(self):
        """Test validation of invalid configuration"""
        config = {
            'project': 'test_project'
            # Missing datasets_file and 0_inspect
        }
        
        errors = validate_inspection_config(config)
        assert len(errors) == 2
        assert any('datasets_file' in error for error in errors)
        assert any('0_inspect' in error for error in errors)
    
    def test_invalid_config_missing_inspect_path(self):
        """Test validation with missing inspect path"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml',
            '0_inspect': {
                # Missing 'path'
            }
        }
        
        errors = validate_inspection_config(config)
        assert len(errors) == 1
        assert 'path' in errors[0]


if __name__ == "__main__":
    pytest.main([__file__])