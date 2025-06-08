import mne
import os
import glob
import argparse
from eeg_raw_to_classification.utils import load_yaml, save_dict_to_json, save_figs_in_html,get_path, load_meeg, find_minimal_unique_root
import numpy as np
import pandas as pd
import traceback
import pathlib
import importlib
def main(pipeline_file, max_files=None):
    cfg = load_yaml(pipeline_file)
    MOUNT = cfg.get('mount', None)
    PROJECT = cfg['project']
    datasets = load_yaml(get_path(cfg['datasets_file'],MOUNT))
    #load_yaml(cfg['datasets_file'])
    inspect_path = get_path(cfg['inspect']['path'],MOUNT).replace('%PROJECT%', PROJECT)
    inspect_path = pathlib.Path(inspect_path).expanduser().as_posix()  # Ensure path is expanded and in POSIX format
    os.makedirs(inspect_path, exist_ok=True)
    this_inspect = cfg['inspect']

    if 'redefine_loader' in this_inspect:
        import_dict = this_inspect['redefine_loader']
        module_name = import_dict['from_this']
        func_name = import_dict['import_that']
        module = importlib.import_module(module_name)
        load_meeg = getattr(module, func_name)
    else:
        from eeg_raw_to_classification.utils import load_meeg

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

                
                meeg = load_meeg(meeg_file, kwargs={'preload': True, 'verbose': 'error'})

                try:
                    report = mne.Report(title=f'Inspect {dslabel} {rel_path}', verbose='error')

                    if isinstance(meeg, mne.io.BaseRaw):
                        report.add_raw(meeg, title=f'{dslabel} {rel_path}')
                    elif isinstance(meeg, mne.BaseEpochs):
                        report.add_epochs(meeg, title=f'{dslabel} {rel_path}')

                    ## add spectrum to report
                    report.add_figure(meeg.plot_psd(show=False), title=f'{dslabel} {rel_path} Spectrum')
                    report.add_figure(meeg.plot_psd(show=False, fmax=200), title=f'{dslabel} {rel_path} Spectrum')

                    report.save(output_base + '_report.html', overwrite=True, verbose='error')
                except Exception as e:
                    print(f"Error adding to report: {e}")
                    save_dict_to_json(output_base + '_problemReport.txt', {'problem': str(e)})

                TIMES.append(meeg.times[-1])
                ch_names = meeg.ch_names
                MONTAGES.append(ch_names)
                MEEGs.append(meeg_file)
                if True: #i == 0:
                    fig = meeg.plot_psd(show=False)
                    save_figs_in_html(output_base + '_spectrum.html', [fig])
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
        
        counts_dict = dict(zip(*np.unique(TIMES, return_counts=True)))
        counts_dict = {int(k): int(v) for k, v in counts_dict.items()}  # Convert keys and values to Python int
        save_dict_to_json(os.path.join(inspect_path, dslabel, 'times_counts.txt'), {'counts': counts_dict})

        df = pd.DataFrame({'EEG': MEEGs, 'montage': MONTAGES, 'times': TIMES})
        df.to_csv(os.path.join(inspect_path, dslabel, f'{dslabel}_inspect.csv'))
        print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect EEG datasets.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process per dataset')
    args = parser.parse_args()
    main(args.pipeline_file, max_files=args.max_files)