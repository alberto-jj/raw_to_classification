"""
Tests for the pure functions in the dataset2bids pipeline.

These tests validate the business logic without I/O dependencies.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from eeg_raw_to_classification.pipelines.dataset2bids_types import (
    BidsificationConfig, DatasetBidsResult, GlobalBidsResult
)
from eeg_raw_to_classification.pipelines.dataset2bids_core import (
    validate_bids_config,
    setup_bidsify_function,
    create_bidsification_config,
    convert_dataset_to_bids,
    compute_global_bids_summary
)


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test validation of valid configuration"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml',
            '1_dataset2bids': {
                'redefine_bidsify': {
                    'from_this': 'test.module',
                    'import_that': 'test_function'
                }
            }
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 0
    
    def test_valid_config_minimal(self):
        """Test validation of minimal valid configuration"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml'
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 0
    
    def test_invalid_config_missing_fields(self):
        """Test validation of invalid configuration"""
        config = {
            'project': 'test_project'
            # Missing datasets_file
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 1
        assert 'datasets_file' in errors[0]
    
    def test_invalid_config_missing_project(self):
        """Test validation with missing project"""
        config = {
            'datasets_file': 'datasets.yml'
            # Missing project
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 1
        assert 'project' in errors[0]
    
    def test_invalid_redefine_bidsify_structure(self):
        """Test validation of invalid redefine_bidsify structure"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml',
            '1_dataset2bids': {
                'redefine_bidsify': 'invalid_string'  # Should be dict
            }
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 1
        assert 'redefine_bidsify' in errors[0]
        assert 'dictionary' in errors[0]
    
    def test_invalid_redefine_bidsify_missing_fields(self):
        """Test validation of redefine_bidsify missing required fields"""
        config = {
            'project': 'test_project',
            'datasets_file': 'datasets.yml',
            '1_dataset2bids': {
                'redefine_bidsify': {
                    'from_this': 'test.module'
                    # Missing import_that
                }
            }
        }
        
        errors = validate_bids_config(config)
        assert len(errors) == 1
        assert 'import_that' in errors[0]


class TestBidsifyFunctionSetup:
    """Test bidsify function setup"""
    
    def test_setup_default_function(self):
        """Test setup of default bidsify function"""
        config = {}
        
        bidsify_func = setup_bidsify_function(config)
        
        # Should return the default sova_bidsify function
        assert bidsify_func.__name__ == 'sova_bidsify'
    
    @patch('importlib.import_module')
    def test_setup_custom_function(self, mock_import):
        """Test setup of custom bidsify function"""
        config = {
            'redefine_bidsify': {
                'from_this': 'custom.module',
                'import_that': 'custom_bidsify'
            }
        }
        
        # Mock module and function
        mock_module = Mock()
        mock_function = Mock()
        mock_function.__name__ = 'custom_bidsify'
        mock_module.custom_bidsify = mock_function
        mock_import.return_value = mock_module
        
        bidsify_func = setup_bidsify_function(config)
        
        mock_import.assert_called_once_with('custom.module')
        assert bidsify_func == mock_function
    
    @patch('importlib.import_module')
    def test_setup_custom_function_import_error(self, mock_import):
        """Test handling of import errors in custom function setup"""
        config = {
            'redefine_bidsify': {
                'from_this': 'nonexistent.module',
                'import_that': 'nonexistent_function'
            }
        }
        
        mock_import.side_effect = ImportError("Module not found")
        
        with pytest.raises(ImportError, match="Cannot import nonexistent.module.nonexistent_function"):
            setup_bidsify_function(config)


class TestBidsificationConfigCreation:
    """Test bidsification config creation"""
    
    def test_create_config_success(self):
        """Test successful creation of bidsification config"""
        dataset_config = {
            'bidsify': {
                'paths': {
                    'source_path': '/source/path',
                    'bids_path': '/bids/path'
                },
                'rules': {'rule1': 'value1'},
                'overwrite': True
            }
        }
        
        def mock_path_resolver(path, mount):
            return path
        
        config = create_bidsification_config(
            'test_dataset', dataset_config, None, mock_path_resolver
        )
        
        assert config.source_path == Path('/source/path')
        assert config.bids_path == Path('/bids/path')
        assert config.rules == {'rule1': 'value1'}
        assert config.overwrite is True
    
    def test_create_config_no_bidsify(self):
        """Test config creation when no bidsify section exists"""
        dataset_config = {
            'other_config': 'value'
        }
        
        def mock_path_resolver(path, mount):
            return path
        
        config = create_bidsification_config(
            'test_dataset', dataset_config, None, mock_path_resolver
        )
        
        assert config is None
    
    def test_create_config_missing_paths(self):
        """Test config creation with missing paths section"""
        dataset_config = {
            'bidsify': {
                'rules': {'rule1': 'value1'}
                # Missing 'paths'
            }
        }
        
        def mock_path_resolver(path, mount):
            return path
        
        with pytest.raises(ValueError, match="Missing 'paths' in bidsify config"):
            create_bidsification_config(
                'test_dataset', dataset_config, None, mock_path_resolver
            )
    
    def test_create_config_missing_path_fields(self):
        """Test config creation with missing path fields"""
        dataset_config = {
            'bidsify': {
                'paths': {
                    'source_path': '/source/path'
                    # Missing 'bids_path'
                }
            }
        }
        
        def mock_path_resolver(path, mount):
            return path
        
        with pytest.raises(ValueError, match="Missing source_path or bids_path"):
            create_bidsification_config(
                'test_dataset', dataset_config, None, mock_path_resolver
            )


class TestDatasetConversion:
    """Test dataset BIDS conversion"""
    
    def test_convert_dataset_success(self):
        """Test successful dataset conversion"""
        bidsify_config = BidsificationConfig(
            source_path=Path('/source'),
            bids_path=Path('/bids'),
            rules={'rule1': 'value1'}
        )
        
        def mock_bidsify_func(source, bids, dataset_cfg, pipeline_cfg):
            return {'converted_files': 5, 'status': 'success'}
        
        result = convert_dataset_to_bids(
            'test_dataset',
            bidsify_config,
            mock_bidsify_func,
            {'dataset': 'config'},
            {'pipeline': 'config'},
            dry_run=False
        )
        
        assert result.success is True
        assert result.dataset_label == 'test_dataset'
        assert result.source_path == Path('/source')
        assert result.bids_path == Path('/bids')
        assert result.error is None
        assert result.conversion_summary == {'converted_files': 5, 'status': 'success'}
    
    def test_convert_dataset_dry_run(self):
        """Test dataset conversion in dry run mode"""
        bidsify_config = BidsificationConfig(
            source_path=Path(__file__).parent,  # Use existing path
            bids_path=Path('/bids'),
            rules={'rule1': 'value1'}
        )
        
        def mock_bidsify_func(source, bids, dataset_cfg, pipeline_cfg):
            raise Exception("Should not be called in dry run")
        
        result = convert_dataset_to_bids(
            'test_dataset',
            bidsify_config,
            mock_bidsify_func,
            {'dataset': 'config'},
            {'pipeline': 'config'},
            dry_run=True
        )
        
        assert result.success is True
        assert result.dataset_label == 'test_dataset'
        assert result.conversion_summary == {'dry_run': True}
    
    def test_convert_dataset_dry_run_missing_source(self):
        """Test dry run with missing source path"""
        bidsify_config = BidsificationConfig(
            source_path=Path('/nonexistent/path'),
            bids_path=Path('/bids'),
            rules={'rule1': 'value1'}
        )
        
        def mock_bidsify_func(source, bids, dataset_cfg, pipeline_cfg):
            raise Exception("Should not be called")
        
        result = convert_dataset_to_bids(
            'test_dataset',
            bidsify_config,
            mock_bidsify_func,
            {'dataset': 'config'},
            {'pipeline': 'config'},
            dry_run=True
        )
        
        assert result.success is False
        assert 'Source path does not exist' in result.error
    
    def test_convert_dataset_failure(self):
        """Test dataset conversion failure"""
        bidsify_config = BidsificationConfig(
            source_path=Path('/source'),
            bids_path=Path('/bids'),
            rules={'rule1': 'value1'}
        )
        
        def mock_bidsify_func(source, bids, dataset_cfg, pipeline_cfg):
            raise Exception("Conversion failed")
        
        result = convert_dataset_to_bids(
            'test_dataset',
            bidsify_config,
            mock_bidsify_func,
            {'dataset': 'config'},
            {'pipeline': 'config'},
            dry_run=False
        )
        
        assert result.success is False
        assert result.dataset_label == 'test_dataset'
        assert result.error == 'Conversion failed'
        assert 'error_trace' in result.conversion_summary


class TestGlobalSummary:
    """Test global BIDS summary computation"""
    
    def test_compute_global_summary_success(self):
        """Test global summary with all successful datasets"""
        dataset_results = {
            'dataset1': DatasetBidsResult(
                dataset_label='dataset1',
                source_path=Path('/source1'),
                bids_path=Path('/bids1'),
                success=True,
                files_converted=5
            ),
            'dataset2': DatasetBidsResult(
                dataset_label='dataset2',
                source_path=Path('/source2'),
                bids_path=Path('/bids2'),
                success=True,
                files_converted=3
            ),
        }
        
        result = compute_global_bids_summary(dataset_results)
        
        assert result.total_datasets == 2
        assert result.successful_datasets == 2
        assert result.total_files_converted == 8
        assert len(result.dataset_results) == 2
    
    def test_compute_global_summary_mixed(self):
        """Test global summary with mixed success/failure"""
        dataset_results = {
            'dataset1': DatasetBidsResult(
                dataset_label='dataset1',
                source_path=Path('/source1'),
                bids_path=Path('/bids1'),
                success=True,
                files_converted=5
            ),
            'dataset2': DatasetBidsResult(
                dataset_label='dataset2',
                source_path=Path('/source2'),
                bids_path=Path('/bids2'),
                success=False,
                files_converted=0,
                error='Conversion failed'
            ),
        }
        
        result = compute_global_bids_summary(dataset_results)
        
        assert result.total_datasets == 2
        assert result.successful_datasets == 1
        assert result.total_files_converted == 5
    
    def test_compute_global_summary_empty(self):
        """Test global summary with no datasets"""
        dataset_results = {}
        
        result = compute_global_bids_summary(dataset_results)
        
        assert result.total_datasets == 0
        assert result.successful_datasets == 0
        assert result.total_files_converted == 0


if __name__ == "__main__":
    pytest.main([__file__])