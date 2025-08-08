"""
Tests for the dataset2bids output manager.

These tests validate output formatting and file generation.
"""

import pytest
import json
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from eeg_raw_to_classification.pipelines.dataset2bids_types import DatasetBidsResult, GlobalBidsResult
from eeg_raw_to_classification.pipelines.dataset2bids_wrapper import Dataset2BidsOutputManager


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for output testing"""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_dataset_result():
    """Create sample dataset result for testing"""
    return DatasetBidsResult(
        dataset_label="test_dataset",
        source_path=Path("/source/test"),
        bids_path=Path("/bids/test"),
        success=True,
        files_converted=5,
        conversion_summary={"status": "completed", "warnings": 0}
    )


@pytest.fixture
def sample_failed_result():
    """Create sample failed dataset result for testing"""
    return DatasetBidsResult(
        dataset_label="failed_dataset",
        source_path=Path("/source/failed"),
        bids_path=Path("/bids/failed"),
        success=False,
        error="Conversion failed due to missing files",
        files_converted=0,
        conversion_summary={"error_trace": "Traceback details..."}
    )


@pytest.fixture
def sample_global_result(sample_dataset_result, sample_failed_result):
    """Create sample global result for testing"""
    dataset_results = {
        "test_dataset": sample_dataset_result,
        "failed_dataset": sample_failed_result
    }
    
    return GlobalBidsResult(
        dataset_results=dataset_results,
        total_datasets=2,
        successful_datasets=1,
        total_files_converted=5
    )


class TestDataset2BidsOutputManager:
    """Test the dataset2bids output manager"""
    
    def test_save_dataset_results_json(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results in JSON format"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="json")
        
        output_file = temp_output_dir / "test_dataset_bids_conversion.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['dataset_label'] == "test_dataset"
        assert Path(data['source_path']) == Path("/source/test")
        assert Path(data['bids_path']) == Path("/bids/test")
        assert data['success'] is True
        assert data['files_converted'] == 5
        assert data['conversion_summary'] == {"status": "completed", "warnings": 0}
    
    def test_save_dataset_results_yaml(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results in YAML format"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="yaml")
        
        output_file = temp_output_dir / "test_dataset_bids_conversion.yaml"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = yaml.safe_load(f)
        
        assert data['dataset_label'] == "test_dataset"
        assert Path(data['source_path']) == Path("/source/test")
        assert Path(data['bids_path']) == Path("/bids/test")
        assert data['success'] is True
        assert data['files_converted'] == 5
    
    def test_save_dataset_results_both(self, temp_output_dir, sample_dataset_result):
        """Test saving dataset results in both formats"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_dataset_result, format="both")
        
        json_file = temp_output_dir / "test_dataset_bids_conversion.json"
        yaml_file = temp_output_dir / "test_dataset_bids_conversion.yaml"
        
        assert json_file.exists()
        assert yaml_file.exists()
        
        # Verify JSON content
        with open(json_file) as f:
            json_data = json.load(f)
        assert json_data['dataset_label'] == "test_dataset"
        
        # Verify YAML content
        with open(yaml_file) as f:
            yaml_data = yaml.safe_load(f)
        assert yaml_data['dataset_label'] == "test_dataset"
    
    def test_save_failed_dataset_results(self, temp_output_dir, sample_failed_result):
        """Test saving failed dataset results"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_dataset_results(sample_failed_result, format="json")
        
        output_file = temp_output_dir / "failed_dataset_bids_conversion.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['dataset_label'] == "failed_dataset"
        assert data['success'] is False
        assert data['error'] == "Conversion failed due to missing files"
        assert data['files_converted'] == 0
        assert 'error_trace' in data['conversion_summary']
    
    def test_save_global_results_json(self, temp_output_dir, sample_global_result):
        """Test saving global results in JSON format"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_global_results(sample_global_result, format="json")
        
        output_file = temp_output_dir / "global_bids_conversion.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert data['total_datasets'] == 2
        assert data['successful_datasets'] == 1
        assert data['total_files_converted'] == 5
        assert data['success_rate'] == 0.5  # 1/2
        
        # Check dataset summaries
        summaries = data['dataset_summaries']
        assert 'test_dataset' in summaries
        assert 'failed_dataset' in summaries
        assert summaries['test_dataset']['success'] is True
        assert summaries['failed_dataset']['success'] is False
    
    def test_save_global_results_yaml(self, temp_output_dir, sample_global_result):
        """Test saving global results in YAML format"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_global_results(sample_global_result, format="yaml")
        
        output_file = temp_output_dir / "global_bids_conversion.yaml"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = yaml.safe_load(f)
        
        assert data['total_datasets'] == 2
        assert data['successful_datasets'] == 1
        assert data['success_rate'] == 0.5
    
    def test_save_global_results_both(self, temp_output_dir, sample_global_result):
        """Test saving global results in both formats"""
        manager = Dataset2BidsOutputManager(temp_output_dir)
        manager.save_global_results(sample_global_result, format="both")
        
        json_file = temp_output_dir / "global_bids_conversion.json"
        yaml_file = temp_output_dir / "global_bids_conversion.yaml"
        
        assert json_file.exists()
        assert yaml_file.exists()
    
    def test_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "non_existent_dir" / "output"
            
            # Directory shouldn't exist yet
            assert not output_dir.exists()
            
            # Creating manager should create directory
            manager = Dataset2BidsOutputManager(output_dir)
            assert output_dir.exists()
    
    def test_success_rate_calculation_zero_datasets(self):
        """Test success rate calculation with zero datasets"""
        empty_result = GlobalBidsResult(
            dataset_results={},
            total_datasets=0,
            successful_datasets=0,
            total_files_converted=0
        )
        
        with TemporaryDirectory() as temp_dir:
            manager = Dataset2BidsOutputManager(Path(temp_dir))
            manager.save_global_results(empty_result, format="json")
            
            output_file = Path(temp_dir) / "global_bids_conversion.json"
            with open(output_file) as f:
                data = json.load(f)
            
            assert data['success_rate'] == 0


if __name__ == "__main__":
    pytest.main([__file__])