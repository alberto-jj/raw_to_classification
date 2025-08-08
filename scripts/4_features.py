"""
DEPRECATED: This script is deprecated in favor of the new CLI entry points.

After installing the package with 'pip install -e .', use the new commands:

  meeg-features project_files/dummy_pipeline.yml --external-jobs 4
  meeg-features project_files/dummy_pipeline.yml --retry-errors
  meeg-features project_files/dummy_pipeline.yml --inspect-only
  
For backward compatibility, this script still works but will redirect to the new CLI.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "scripts/4_features.py is deprecated. Use 'meeg-features' command instead. "
    "Install with 'pip install -e .' to get the new CLI commands.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect to new CLI
from eeg_raw_to_classification.cli.commands import features_main

if __name__ == "__main__":
    print("⚠️  DEPRECATED: This script is deprecated.")
    print("   After 'pip install -e .', use: meeg-features <args>")
    print("   Redirecting to new CLI...\n")
    
    features_main()

