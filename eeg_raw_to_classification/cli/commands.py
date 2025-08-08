"""
Command-line interface commands for eeg_raw_to_classification.

This module contains all CLI entry points that will be installed as console commands.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

from ..pipelines.inspect import pipeline_inspect
from ..pipelines.features import pipeline_features
from ..loggers import setup_pipeline_logging


def inspect_main(args: Optional[List[str]] = None):
    """
    Main entry point for meeg-inspect command.
    
    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        prog='meeg-inspect',
        description='Inspect MEEG datasets with enhanced logging and structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use new structured approach with JSON output
  meeg-inspect project_files/dummy_pipeline.yml --max-files 10

  # Use YAML output format for human-readable summaries
  meeg-inspect project_files/dummy_pipeline.yml --output-format yaml

  # Use both JSON and YAML formats
  meeg-inspect project_files/dummy_pipeline.yml --output-format both

  # Enable debug logging
  meeg-inspect project_files/dummy_pipeline.yml --debug

  # Use legacy implementation for backward compatibility
  meeg-inspect project_files/dummy_pipeline.yml --legacy
        """
    )
    
    parser.add_argument(
        'pipeline_file', 
        type=str, 
        help='Path to the pipeline.yml file'
    )
    parser.add_argument(
        '--max-files', 
        type=int, 
        default=None, 
        help='Maximum number of files to process per dataset'
    )
    parser.add_argument(
        '--output-format', 
        choices=['json', 'yaml', 'both'], 
        default='json',
        help='Output format for results (default: json)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )
    parser.add_argument(
        '--legacy', 
        action='store_true',
        help='Use legacy implementation for backward compatibility'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Execute inspection
    try:
        if parsed_args.legacy:
            # Use legacy implementation
            print("Using legacy implementation for backward compatibility")
            pipeline_inspect(
                parsed_args.pipeline_file,
                max_files=parsed_args.max_files,
                use_legacy=True
            )
        else:
            # Use new structured implementation
            logger = setup_pipeline_logging('inspect_pipeline', Path(parsed_args.log_dir), parsed_args.debug)
            
            with logger.timed_operation('full_inspection_pipeline', 
                                       pipeline_file=parsed_args.pipeline_file,
                                       max_files=parsed_args.max_files):
                
                results = pipeline_inspect(
                    pipeline_file=parsed_args.pipeline_file,
                    max_files=parsed_args.max_files,
                    output_format=parsed_args.output_format,
                    logger=logger.logger
                )
                
                if results:
                    # Log summary metrics
                    logger.log_metrics({
                        'total_datasets': len(results.dataset_results),
                        'total_files': results.total_files,
                        'successful_files': results.total_successful,
                        'success_rate': results.total_successful / results.total_files if results.total_files > 0 else 0,
                        'global_common_channels': len(results.global_common_montage),
                        'global_total_channels': len(results.global_union_montage)
                    }, context='pipeline_summary')
                    
                    print(f"\n Inspection completed successfully!")
                    print(f"   Datasets processed: {len(results.dataset_results)}")
                    print(f"   Files processed: {results.total_successful}/{results.total_files}")
                    print(f"   Success rate: {(results.total_successful/results.total_files*100):.1f}%" if results.total_files > 0 else "   📊 Success rate: 0%")
                    print(f"   Output format: {parsed_args.output_format}")
                    print(f"   Results saved to inspection output directory")
                    print(f"   Logs saved to: {parsed_args.log_dir}")
                else:
                    print("⚠️ No results returned from inspection pipeline")
                    
    except KeyboardInterrupt:
        print("\n⚠️ Inspection interrupted by user")
        sys.exit(1)
    except Exception as e:
        if not parsed_args.legacy:
            logger.logger.error(f"Pipeline failed: {e}", exc_info=True)
            print(f"\n Pipeline failed: {e}")
            print(f"   Check logs in: {parsed_args.log_dir}")
        else:
            print(f"\n Pipeline failed: {e}")
        sys.exit(1)


def features_main(args: Optional[List[str]] = None):
    """
    Main entry point for meeg-features command.
    
    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        prog='meeg-features',
        description='Extract features from MEEG datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract features with default settings
  meeg-features project_files/dummy_pipeline.yml

  # Use parallel processing with 4 jobs
  meeg-features project_files/dummy_pipeline.yml --external-jobs 4

  # Enable debug mode
  meeg-features project_files/dummy_pipeline.yml --raise-on-error

  # Process only specific file index
  meeg-features project_files/dummy_pipeline.yml --index 42

  # Inspect only (check what would be processed)
  meeg-features project_files/dummy_pipeline.yml --inspect-only
        """
    )
    
    parser.add_argument(
        'pipeline_file', 
        type=str, 
        help='Path to the pipeline YAML file'
    )
    parser.add_argument(
        '--external-jobs', 
        type=int, 
        default=1, 
        help='Number of external jobs for parallel processing (default: 1)'
    )
    parser.add_argument(
        '--raise-on-error', 
        action='store_true', 
        help='Raise on error if set (useful for debugging)'
    )
    parser.add_argument(
        '--retry-errors', 
        action='store_true', 
        help='Retry files that had errors'
    )
    parser.add_argument(
        '--index', 
        type=int, 
        default=None, 
        help='Index of the file to process (for batch processing)'
    )
    parser.add_argument(
        '--only-total', 
        action='store_true', 
        help='Just get the total number of files without processing'
    )
    parser.add_argument(
        '--inspect-only', 
        action='store_true', 
        help='Just get the status of the features for each file'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Execute feature extraction
    try:
        inspect_list = pipeline_features(
            parsed_args.pipeline_file, 
            parsed_args.external_jobs, 
            parsed_args.raise_on_error, 
            parsed_args.external_jobs > 1,  # parallelize if external_jobs > 1
            parsed_args.retry_errors, 
            parsed_args.index, 
            parsed_args.only_total, 
            parsed_args.inspect_only
        )
        
        if parsed_args.only_total:
            print(f"Total files to process: {inspect_list}")
        elif parsed_args.inspect_only:
            processed_count = len([item for item in inspect_list if item.get('status', False)])
            total_count = len(inspect_list)
            print(f"\n Feature inspection completed!")
            print(f"   Features ready: {processed_count}/{total_count}")
            print(f"   Completion rate: {(processed_count/total_count*100):.1f}%" if total_count > 0 else "   📊 Completion rate: 0%")
        else:
            print(f"\n Feature extraction completed!")
            if inspect_list:
                print(f"   Features processed: {len(inspect_list)} items")
                
    except KeyboardInterrupt:
        print("\n Feature extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Feature extraction failed: {e}")
        if parsed_args.raise_on_error:
            raise
        sys.exit(1)


def dataset2bids_main(args: Optional[List[str]] = None):
    """
    Main entry point for meeg-dataset2bids command.
    
    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        prog='meeg-dataset2bids',
        description='Convert MEEG datasets to BIDS format with enhanced logging and structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert datasets to BIDS format with JSON output
  meeg-dataset2bids project_files/dummy_pipeline.yml

  # Use YAML output format for human-readable summaries
  meeg-dataset2bids project_files/dummy_pipeline.yml --output-format yaml

  # Use both JSON and YAML formats
  meeg-dataset2bids project_files/dummy_pipeline.yml --output-format both

  # Enable debug logging
  meeg-dataset2bids project_files/dummy_pipeline.yml --debug --log-dir ./my_logs

  # Dry run mode (validate without converting)
  meeg-dataset2bids project_files/dummy_pipeline.yml --dry-run

  # Allow overwriting existing BIDS data
  meeg-dataset2bids project_files/dummy_pipeline.yml --overwrite

  # Use legacy implementation for backward compatibility
  meeg-dataset2bids project_files/dummy_pipeline.yml --legacy
        """
    )
    
    parser.add_argument(
        'pipeline_file', 
        type=str, 
        help='Path to the pipeline.yml file'
    )
    parser.add_argument(
        '--output-format', 
        choices=['json', 'yaml', 'both'], 
        default='json',
        help='Output format for results (default: json)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug logging'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./logs',
        help='Directory for log files (default: ./logs)'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Validate configuration and paths without converting files'
    )
    parser.add_argument(
        '--overwrite', 
        action='store_true',
        help='Allow overwriting existing BIDS data'
    )
    parser.add_argument(
        '--legacy', 
        action='store_true',
        help='Use legacy implementation for backward compatibility'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Execute dataset2bids conversion
    try:
        if parsed_args.legacy:
            # Use legacy implementation
            print("Using legacy implementation for backward compatibility")
            from ..pipelines.dataset2bids import pipeline_dataset2bids
            pipeline_dataset2bids(parsed_args.pipeline_file)
        else:
            # Use new structured implementation
            from ..loggers import StructuredLogger
            from ..pipelines.dataset2bids_wrapper import Dataset2BidsPipeline
            
            # Setup logger
            logger = StructuredLogger(
                name="dataset2bids",
                log_dir=Path(parsed_args.log_dir),
                debug=parsed_args.debug
            )
            
            # Create and run pipeline
            pipeline = Dataset2BidsPipeline(logger)
            
            result = pipeline.run_pipeline(
                pipeline_file=parsed_args.pipeline_file,
                output_format=parsed_args.output_format,
                dry_run=parsed_args.dry_run,
                overwrite=parsed_args.overwrite
            )
            
            print(f"\n Dataset to BIDS conversion completed!")
            print(f"   Datasets converted: {result.successful_datasets}/{result.total_datasets}")
            print(f"   Total files converted: {result.total_files_converted}")
            print(f"   Success rate: {(result.successful_datasets/result.total_datasets*100):.1f}%" if result.total_datasets > 0 else "   📊 Success rate: 0%")
            print(f"   Output format: {parsed_args.output_format}")
            
            if parsed_args.dry_run:
                print("   Mode: Dry run (no files were actually converted)")
            
            print(f"   Results saved to BIDS output directory")
            print(f"   Logs saved to: {parsed_args.log_dir}")
            
            # Exit with error code if any datasets failed
            if result.successful_datasets < result.total_datasets:
                print(f"\n⚠️ Some datasets failed conversion. Check logs in {parsed_args.log_dir}")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n Dataset to BIDS conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Dataset to BIDS conversion failed: {e}")
        if parsed_args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def preprocessing_main(args: Optional[List[str]] = None):
    """
    Main entry point for meeg-preprocess command (placeholder for future implementation).
    
    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        prog='meeg-preprocess',
        description='Preprocess MEEG datasets.',
    )
    
    parser.add_argument(
        'pipeline_file', 
        type=str, 
        help='Path to the pipeline YAML file'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    parsed_args = parser.parse_args(args)
    
    print("🚧 Preprocessing pipeline not yet implemented in new CLI structure")
    print("   Use the legacy scripts/3_preprocess.py for now")
    sys.exit(1)


def aggregate_main(args: Optional[List[str]] = None):
    """
    Main entry point for meeg-aggregate command (placeholder for future implementation).
    
    Args:
        args: Optional list of arguments (for testing). If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        prog='meeg-aggregate',
        description='Aggregate MEEG features across datasets.',
    )
    
    parser.add_argument(
        'pipeline_file', 
        type=str, 
        help='Path to the pipeline YAML file'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    parsed_args = parser.parse_args(args)
    
    print("🚧 Aggregation pipeline not yet implemented in new CLI structure")
    print("   Use the legacy scripts/5_aggregate.py for now")
    sys.exit(1)


# Convenience function for backward compatibility
def main():
    """
    General entry point that dispatches to appropriate command based on script name.
    This maintains some backward compatibility.
    """
    import os
    script_name = os.path.basename(sys.argv[0])
    
    if 'inspect' in script_name or script_name == '0_inspect.py':
        inspect_main()
    elif 'dataset2bids' in script_name or script_name == '1_dataset2bids.py':
        dataset2bids_main()
    elif 'features' in script_name or script_name == '4_features.py':
        features_main()
    elif 'preprocess' in script_name or script_name == '3_preprocess.py':
        preprocessing_main()
    elif 'aggregate' in script_name or script_name == '5_aggregate.py':
        aggregate_main()
    else:
        print("Available commands:")
        print("  meeg-inspect        - Inspect MEEG datasets")
        print("  meeg-dataset2bids   - Convert datasets to BIDS format")
        print("  meeg-features       - Extract features from MEEG datasets")
        print("  meeg-preprocess     - Preprocess MEEG datasets (coming soon)")
        print("  meeg-aggregate      - Aggregate features (coming soon)")
        sys.exit(1)


if __name__ == "__main__":
    main()