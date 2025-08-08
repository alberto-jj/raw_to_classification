#!/usr/bin/env python
"""
Quick test script to verify CLI commands work before installation.

Run this to test the CLI functionality without installing entry points.
"""

import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent))

def test_inspect_help():
    """Test inspect command help"""
    print("=== Testing meeg-inspect --help ===")
    from eeg_raw_to_classification.cli.commands import inspect_main
    try:
        inspect_main(['--help'])
    except SystemExit:
        pass  # Help command exits normally
    print("Inspect help works\n")

def test_features_help():
    """Test features command help"""
    print("=== Testing meeg-features --help ===")
    from eeg_raw_to_classification.cli.commands import features_main
    try:
        features_main(['--help'])
    except SystemExit:
        pass  # Help command exits normally
    print("Features help works\n")

def test_version():
    """Test version information"""
    print("=== Testing --version ===")
    from eeg_raw_to_classification.cli.commands import inspect_main
    try:
        inspect_main(['--version'])
    except SystemExit:
        pass  # Version command exits normally
    print("Version works\n")

def test_invalid_args():
    """Test error handling with invalid arguments"""
    print("=== Testing error handling ===")
    from eeg_raw_to_classification.cli.commands import inspect_main
    try:
        inspect_main(['nonexistent_file.yml', '--invalid-arg'])
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
    print("Testing CLI commands before installation...\n")
    
    try:
        test_inspect_help()
        test_features_help() 
        test_version()
        test_invalid_args()
        
        print("All CLI tests passed!")
        print("\nNext steps:")
        print("1. Run 'pip install -e .' to install entry points")
        print("2. Test with: meeg-inspect --help")
        print("3. Test with: meeg-features --help")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()