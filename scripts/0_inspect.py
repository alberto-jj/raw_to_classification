"""
DEPRECATED: This script is deprecated in favor of the new CLI entry points.

After installing the package with 'pip install -e .', use the new commands:

  meeg-inspect project_files/dummy_pipeline.yml --max-files 10
  meeg-inspect project_files/dummy_pipeline.yml --output-format yaml
  meeg-inspect project_files/dummy_pipeline.yml --debug
  
For backward compatibility, this script still works but will redirect to the new CLI.
"""

import sys
import warnings

# Show deprecation warning
warnings.warn(
    "scripts/0_inspect.py is deprecated. Use 'meeg-inspect' command instead. "
    "Install with 'pip install -e .' to get the new CLI commands.",
    DeprecationWarning,
    stacklevel=2
)

# Redirect to new CLI
from eeg_raw_to_classification.cli.commands import inspect_main

if __name__ == "__main__":
    print("⚠️  DEPRECATED: This script is deprecated.")
    print("   After 'pip install -e .', use: meeg-inspect <args>")
    print("   Redirecting to new CLI...\n")
    
    inspect_main()