"""
Command-line interface for eeg_raw_to_classification package.

This module provides unified CLI commands for all pipeline operations.
"""

from .commands import inspect_main, features_main

__all__ = ['inspect_main', 'features_main']