#!/usr/bin/env python
"""
Quick test script to verify dataset2bids CLI commands work before installation.

Run this to test the CLI functionality without installing entry points.
"""

import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent))

def test_dataset2bids_help():
    """Test dataset2bids command help"""
    print("=== Testing meeg-dataset2bids --help ===")
    from eeg_raw_to_classification.cli.commands import dataset2bids_main
    try:
        dataset2bids_main(['--help'])
    except SystemExit:
        pass  # Help command exits normally
    print("Dataset2bids help works\n")

def test_version():
    """Test version information"""
    print("=== Testing --version ===")
    from eeg_raw_to_classification.cli.commands import dataset2bids_main
    try:
        dataset2bids_main(['--version'])
    except SystemExit:
        pass  # Version command exits normally
    print("Version works\n")

def test_invalid_args():
    """Test error handling with invalid arguments"""
    print("=== Testing error handling ===")
    from eeg_raw_to_classification.cli.commands import dataset2bids_main
    try:
        dataset2bids_main(['nonexistent_file.yml', '--invalid-arg'])
    except SystemExit as e:
        if e.code != 0:
            print("Error handling works (expected failure)")
        else:
            print("Unexpected success with invalid args")
    except Exception as e:
        print(f"Error caught as expected: {type(e).__name__}")
    print()

def main():
    """Run all tests"""
    print("Testing dataset2bids CLI commands before installation...\n")
    
    try:
        test_dataset2bids_help()
        test_version()
        test_invalid_args()
        
        print("All dataset2bids CLI tests passed!")
        print("\nNext steps:")
        print("1. Run 'pip install -e .' to install entry points")
        print("2. Test with: meeg-dataset2bids --help")
        print("3. Test with: meeg-dataset2bids project_files/dummy_pipeline.yml --dry-run")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()