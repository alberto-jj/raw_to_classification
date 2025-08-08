# CLI Migration Guide

This document explains the new command-line interface and how to migrate from the old scripts.

## Installation

After making changes, reinstall the package to get the new CLI commands:

```bash
pip install -e .
```

This will create the following console commands:
- `meeg-inspect`
- `meeg-dataset2bids`
- `meeg-features` 
- `meeg-preprocess` (placeholder)
- `meeg-aggregate` (placeholder)

## New CLI Commands

### MEEG Inspect

**Old way:**
```bash
python scripts/0_inspect.py project_files/dummy_pipeline.yml --max_files 10
```

**New way:**
```bash
meeg-inspect project_files/dummy_pipeline.yml --max-files 10
```

**New features:**
```bash
# YAML output for human-readable summaries
meeg-inspect project_files/dummy_pipeline.yml --output-format yaml

# Both JSON and YAML
meeg-inspect project_files/dummy_pipeline.yml --output-format both

# Debug logging
meeg-inspect project_files/dummy_pipeline.yml --debug --log-dir ./my_logs

# Legacy mode (same as old script)
meeg-inspect project_files/dummy_pipeline.yml --legacy
```

### MEEG Dataset2BIDS

**Old way:**
```bash
python scripts/1_dataset2bids.py project_files/dummy_pipeline.yml
```

**New way:**
```bash
meeg-dataset2bids project_files/dummy_pipeline.yml
```

**New features:**
```bash
# YAML output for human-readable summaries
meeg-dataset2bids project_files/dummy_pipeline.yml --output-format yaml

# Both JSON and YAML
meeg-dataset2bids project_files/dummy_pipeline.yml --output-format both

# Debug logging
meeg-dataset2bids project_files/dummy_pipeline.yml --debug --log-dir ./my_logs

# Dry run mode (validate without converting)
meeg-dataset2bids project_files/dummy_pipeline.yml --dry-run

# Allow overwriting existing BIDS data
meeg-dataset2bids project_files/dummy_pipeline.yml --overwrite

# Legacy mode (same as old script)
meeg-dataset2bids project_files/dummy_pipeline.yml --legacy
```

### MEEG Features

**Old way:**
```bash
python scripts/4_features.py project_files/dummy_pipeline.yml --external_jobs 4
```

**New way:**
```bash
meeg-features project_files/dummy_pipeline.yml --external-jobs 4
```

**New features:**
```bash
# Inspect what would be processed
meeg-features project_files/dummy_pipeline.yml --inspect-only

# Retry failed files
meeg-features project_files/dummy_pipeline.yml --retry-errors

# Process specific file by index
meeg-features project_files/dummy_pipeline.yml --index 42

# Get total file count
meeg-features project_files/dummy_pipeline.yml --only-total
```

## Benefits of New CLI

### 1. **Professional Installation**
- Commands available system-wide after `pip install`
- No need to remember script paths
- Follows Python packaging best practices

### 2. **Consistent Interface**
- All commands use similar argument patterns
- Better help messages with examples
- Version information with `--version`

### 3. **Enhanced Features**
- Structured logging with JSON metrics
- Progress tracking and timing
- Multiple output formats (JSON/YAML)
- Better error handling and reporting

### 4. **Future-Proof**
- Easy to add new commands
- Centralized CLI logic
- Extensible architecture

## Backward Compatibility

The old scripts still work and will redirect to the new CLI with a deprecation warning:

```bash
$ python scripts/0_inspect.py project_files/dummy_pipeline.yml
⚠️ DEPRECATED: This script is deprecated.
   After 'pip install -e .', use: meeg-inspect <args>
   Redirecting to new CLI...

$ python scripts/1_dataset2bids.py project_files/dummy_pipeline.yml
⚠️ DEPRECATED: This script is deprecated.
   After 'pip install -e .', use: meeg-dataset2bids <args>
   Redirecting to new CLI...
```

## Migration Checklist

- [ ] Run `pip install -e .` to install entry points
- [ ] Test new commands: `meeg-inspect --help`, `meeg-dataset2bids --help`
- [ ] Update any batch scripts or documentation
- [ ] Try new features like `--output-format yaml` and `--dry-run`
- [ ] Gradually phase out old script usage

## Troubleshooting

**Commands not found?**
```bash
# Reinstall in development mode
pip install -e .

# Check if commands are available
meeg-inspect --version
```

**Import errors?**
```bash
# Make sure you're in the right environment
pip list | grep eeg-raw-to-classification

# Reinstall if needed
pip uninstall eeg_raw_to_classification
pip install -e .
```

**Old behavior needed?**
```bash
# Use legacy flag
meeg-inspect project_files/dummy_pipeline.yml --legacy

# Or use old scripts (with deprecation warning)
python scripts/0_inspect.py project_files/dummy_pipeline.yml
```