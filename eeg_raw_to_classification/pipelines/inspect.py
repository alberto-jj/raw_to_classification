"""
Refactored MEEG inspection pipeline with improved architecture.

This module provides both the new structured approach and backward compatibility
with the original pipeline_inspect function.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import new structured components
from .inspection_wrapper import MEEGInspectionPipeline
from .inspection_types import GlobalInspectionResult
from ..loggers import setup_pipeline_logging

# Legacy imports for backward compatibility
import mne
import os
import glob
from eeg_raw_to_classification.utils import load_yaml, save_dict_to_json, save_figs_in_html,get_path, load_meeg, find_minimal_unique_root
import numpy as np
import pandas as pd
import traceback
import pathlib
import importlib
import matplotlib.pyplot as plt


def pipeline_inspect(
    pipeline_file: str, 
    max_files: Optional[int] = None,
    output_format: str = "json",
    logger: Optional[logging.Logger] = None,
    use_legacy: bool = False
) -> Optional[GlobalInspectionResult]:
    """
    Main pipeline inspection function with new structured approach.
    
    Args:
        pipeline_file: Path to pipeline configuration file
        max_files: Maximum files to process per dataset
        output_format: Output format ('json', 'yaml', 'both')
        logger: Optional logger instance
        use_legacy: Use legacy implementation for backward compatibility
        
    Returns:
        GlobalInspectionResult if using new implementation, None if legacy
    """
    if use_legacy:
        # Use original implementation for backward compatibility
        return _pipeline_inspect_legacy(pipeline_file, max_files)
    
    # Use new structured implementation
    if logger is None:
        # Create a simple logger if none provided
        structured_logger = setup_pipeline_logging(
            "inspect_pipeline", 
            Path("./logs"), 
            debug=False
        )
        logger = structured_logger
    elif not hasattr(logger, 'timed_operation'):
        # Wrap regular logger in structured logger
        structured_logger = setup_pipeline_logging(
            "inspect_pipeline", 
            Path("./logs"), 
            debug=False
        )
        logger = structured_logger
    
    # Run new pipeline
    pipeline = MEEGInspectionPipeline(logger)
    return pipeline.run_pipeline(pipeline_file, max_files, output_format)


def _pipeline_inspect_legacy(pipeline_file: str, max_files: Optional[int] = None):
    """
    Legacy implementation for backward compatibility.
    Preserves original behavior exactly.
    """
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    PROJECT = cfg['project']
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))
    #load_yaml(cfg['datasets_file'])
    inspect_path = get_path(cfg['0_inspect']['path'],MOUNT).replace('%PROJECT%', PROJECT)
    inspect_path = pathlib.Path(inspect_path).expanduser().as_posix()  # Ensure path is expanded and in POSIX format
    os.makedirs(inspect_path, exist_ok=True)
    this_inspect = cfg['0_inspect']

    if 'redefine_loader' in this_inspect:
        import_dict = this_inspect['redefine_loader']
        module_name = import_dict['from_this']
        func_name = import_dict['import_that']
        module = importlib.import_module(module_name)
        load_meeg = getattr(module, func_name)
    else:
        from eeg_raw_to_classification.utils import load_meeg

    MONTAGES_ALL_DATASETS = []

    for dslabel, DATASET in datasets.items():
        if DATASET.get('skip', False):
            continue
        exemplar_file = get_path(DATASET['example_file'], MOUNT)
        exemplar_file = pathlib.Path(exemplar_file).expanduser().as_posix()  # Ensure path is expanded and in POSIX format
        #DATASET['example_file']
        meegs = glob.glob(exemplar_file, recursive=True)
        MONTAGES = []
        TIMES = []
        MEEGs = []
        SHAPES = []
        SFREQS = []
        INFOS = []

        minimal_root = find_minimal_unique_root(meegs)
        print(f"[{dslabel}] Minimal root for unique paths: {minimal_root}")
        for i, meeg_file in enumerate(meegs):

            rel_path = os.path.relpath(meeg_file, minimal_root)
            output_dir = os.path.join(inspect_path, dslabel, os.path.dirname(rel_path))
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(meeg_file)

            name_no_ext = os.path.splitext(filename)[0]
            output_base = os.path.join(output_dir, name_no_ext)
            print(f"Writing to: {output_base}")
            os.makedirs(os.path.dirname(output_base), exist_ok=True)


            try:

                
                meeg = load_meeg(meeg_file, DATASET, kwargs={'preload': True, 'verbose': 'error'})

                try:
                    report = mne.Report(title=f'Inspect {dslabel} {rel_path}', verbose='error')

                    if isinstance(meeg, mne.io.BaseRaw):
                        report.add_raw(meeg, title=f'{dslabel} {rel_path}')
                    elif isinstance(meeg, mne.BaseEpochs):
                        report.add_epochs(meeg, title=f'{dslabel} {rel_path}')

                    ## add spectrum to report
                    report.add_figure(meeg.plot_psd(show=False), title=f'{dslabel} {rel_path} Spectrum')
                    fmax = meeg.info['sfreq'] / 2
                    if fmax > 200:
                        fmax = 200
                        report.add_figure(meeg.plot_psd(show=False, fmax=fmax), title=f'{dslabel} {rel_path} Spectrum Below 200Hz')

                    report.save(output_base + '_report.html', overwrite=True, verbose='error', open_browser=False)
                    del report
                except Exception as e:
                    print(f"Error adding to report: {e}")
                    save_dict_to_json(output_base + '_problemReport.txt', {'problem': str(e)})

                TIMES.append(meeg.times[-1])
                ch_names = meeg.ch_names
                MONTAGES.append(ch_names)
                MEEGs.append(meeg_file)
                SHAPES.append(meeg.get_data().shape)
                SFREQS.append(meeg.info['sfreq'])
                INFOS.append(meeg.info.__str__())
                ALL_MONTAGES = MONTAGES_ALL_DATASETS.append(ch_names)
                
                if True: #i == 0:
                    fig = meeg.plot_psd(show=False)
                    save_figs_in_html(output_base + '_spectrum.html', [fig])
                plt.close('all')
            except:
                save_dict_to_json(output_base + '_problem.txt', {'problem': traceback.format_exc()})
                print(traceback.format_exc())
            if max_files and i > max_files:
                break
        common = set(MONTAGES[0])
        union_montage = set(MONTAGES[0])
        for montage in MONTAGES[1:]:
            common = common.intersection(set(montage))
            union_montage = union_montage.union(set(montage))

        save_dict_to_json(os.path.join(inspect_path, dslabel, 'common_montage.txt'), {'common_montage': list(common)})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'union_montage.txt'), {'union_montage': list(union_montage)})

        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times.txt'), {'times': TIMES})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times_stats.txt'), {'mean': np.mean(TIMES), 'max': np.max(TIMES), 'min': np.min(TIMES), 'median': np.median(TIMES), 'std': np.std(TIMES)})
        
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'shapes.txt'), {'shapes': SHAPES})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'sfreqs.txt'), {'sfreqs': SFREQS})
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'infos.txt'), {'infos': INFOS})

        counts_dict = dict(zip(*np.unique(TIMES, return_counts=True)))
        counts_dict = {int(k): int(v) for k, v in counts_dict.items()}  # Convert keys and values to Python int
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times_counts.txt'), {'counts': counts_dict})

        df = pd.DataFrame({'MEEG': MEEGs, 'montage': MONTAGES, 'times': TIMES, 'shape': SHAPES, 'sfreq': SFREQS, 'info': INFOS})
        df.to_csv(os.path.join(inspect_path, dslabel, f'{dslabel}_inspect.csv'))
        print(df)

    common = set(MONTAGES_ALL_DATASETS[0])
    union_montage = set(MONTAGES_ALL_DATASETS[0])
    for montage in MONTAGES_ALL_DATASETS[1:]:
        common = common.intersection(set(montage))
        union_montage = union_montage.union(set(montage))
    save_dict_to_json(os.path.join(inspect_path, 'common_montage.txt'), {'common_montage': list(common)})
    save_dict_to_json(os.path.join(inspect_path, 'union_montage.txt'), {'union_montage': list(union_montage)})


# CLI interface moved to eeg_raw_to_classification.cli.commands
# Use 'meeg-inspect' command after pip installation