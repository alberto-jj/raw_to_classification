"""
DEPRECATED: This script is deprecated in favor of the new CLI entry points.

After installing the package with 'pip install -e .', use the new commands:

  meeg-dataset2bids project_files/dummy_pipeline.yml
  meeg-dataset2bids project_files/dummy_pipeline.yml --output-format yaml
  meeg-dataset2bids project_files/dummy_pipeline.yml --dry-run
  
For backward compatibility, this script still works but will redirect to the new CLI.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "scripts/1_dataset2bids.py is deprecated. Use 'meeg-dataset2bids' command instead. "
    "Install with 'pip install -e .' to get the new CLI commands.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect to new CLI
from eeg_raw_to_classification.cli.commands import dataset2bids_main

if __name__ == "__main__":
    print("⚠️  DEPRECATED: This script is deprecated.")
    print("   After 'pip install -e .', use: meeg-dataset2bids <args>")
    print("   Redirecting to new CLI...\n")
    
    dataset2bids_main()