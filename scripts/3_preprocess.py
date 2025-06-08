import argparse
import os
import traceback
from joblib import delayed, Parallel
import psutil
import bids
from eeg_raw_to_classification.utils import load_yaml
from eeg_raw_to_classification.utils import get_derivative_path,get_path
import time
import pathlib
import importlib


def foo(meeg_file, this_prep, DATASET, preprocessed_path, DEBUG, internal_njobs=1, retry_errors=False):
    # imports here to avoid problems with joblib
    import os


    njobs = internal_njobs #internal jobs #len(psutil.Process().cpu_affinity())
    print('Internal NJOBS:', njobs)
    print(meeg_file)
    fifname = os.path.basename(preprocessed_path)
    fifpath = os.path.dirname(preprocessed_path)

    if not os.path.isfile(preprocessed_path) or this_prep['overwrite']:
        if os.path.isfile(preprocessed_path.replace('.fif', '_problem.txt')) and not retry_errors:
            print(f'Error file exists: {preprocessed_path.replace(".fif", "_problem.txt")}, skipping')
            return
        import matplotlib.pyplot as plt

        os.makedirs(fifpath, exist_ok=True)
        try:
            from eeg_raw_to_classification.utils import save_figs_in_html, save_dict_to_json

            if 'redefine_prepare' in this_prep:
                import_dict = this_prep['redefine_prepare']
                module_name = import_dict['from_this']
                func_name = import_dict['import_that']
                module = importlib.import_module(module_name)
                prepare = getattr(module, func_name)
            else:
                from eeg_raw_to_classification.preprocessing import prepare

            processed_meeg, info, figures, report = prepare(filename=meeg_file, dataset_cfg=DATASET, njobs=njobs, **this_prep['prepare'])
            figs_path = preprocessed_path.replace('reject_epo.fif', '_prepareFigs.html')
            info_path = preprocessed_path.replace('reject_epo.fif', '_prepareInfo.txt')

            save_figs_in_html(figs_path, figures)
            save_dict_to_json(info_path, info)

            processed_meeg.save(fifpath + '/' + fifname, split_naming='bids', overwrite=True)

            if report is not None:
                report.save_as_html(preprocessed_path.replace('.fif', '_prepareReport.html'), overwrite=True)
            plt.close('all')
        except Exception:
            print(traceback.format_exc())
            if DEBUG:
                raise
            else:
                save_dict_to_json(preprocessed_path.replace('.fif', '_problem.txt'), {'file': meeg_file, 'problem': traceback.format_exc()})
    else:
        print(f'Already Exists: {preprocessed_path} or overwrite is False')

def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG data.')
    parser.add_argument('pipeline_yml', type=str, help='Path to the pipeline YAML file.')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process.')
    parser.add_argument('--external_jobs', type=int, default=1, help='Number of external jobs.')
    parser.add_argument('--internal_jobs', type=int, default=1, help='Number of internal jobs.')
    parser.add_argument('--retry_errors', action='store_true', help='Retry files that had errors.')
    parser.add_argument('--raise_on_error', action='store_true', help='Enable raise_on_error mode.')
    parser.add_argument('--index', type=int, default=None, help='Index of the file to process. Total index taking into account the dataset outer loop. Only works with external_jobs=1.')
    parser.add_argument('--only_total', action='store_true', help='Just get the total number of files. Only works with external_jobs=1.')



    args = parser.parse_args()

    cfg = load_yaml(args.pipeline_yml)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'], MOUNT))
    #datasets = load_yaml(cfg['datasets_file'])

    PROJECT = cfg['project']
    MAX_FILES = args.max_files
    external_njobs = args.external_jobs
    DEBUG = args.raise_on_error
    PARALLELIZE = external_njobs > 1
    internal_njobs = args.internal_jobs
    only_total = args.only_total
    single_index = args.index



    if (only_total or single_index) and external_njobs > 1:
        raise ValueError('Cannot get total number of files or process single file with external_jobs > 1')
    ALL_MEEGS = []
    for preplabel in cfg['preprocess']['prep_list']:
        overall_index = 0
        # you may try to do this loop outisde (with inner eeg loop) as in 4_features.py,
        # but notice that foo depends on some loop-state variables,
        # so its a bit more complicated, and perhaps not that worth it
        for dslabel, DATASET in datasets.items(): 
            if DATASET.get('skip', False):
                continue

            this_prep = cfg['3_preprocess']['prep_cfg'][preplabel]

            print(f'PREPROCESSING {dslabel} with {preplabel} pipeline')
            file_filter = DATASET.get('bids_layout', None)

            start_time = time.time()
            bids_root = DATASET.get('bids_root', None)
            bids_root = get_path(bids_root, MOUNT)
            layout = bids.BIDSLayout(bids_root,validate=False)
            # how to make this faster, it takes too long...
            meegs = layout.get(**file_filter)
            end_time = time.time()
            print(f'Time taken to get EEG files from layout: {end_time - start_time} seconds for dataset {dslabel}')

            if MAX_FILES:
                if MAX_FILES > len(meegs):
                    limit = len(meegs)
                else:
                    limit = MAX_FILES
                meegs = meegs[:limit]
            meegs = [pathlib.Path(x).as_posix() for x in meegs]
            print(len(meegs), meegs)
            

            derivatives_root = DATASET.get('derivatives_root', None)
            
            if derivatives_root is not None:
                derivatives_root = get_path(derivatives_root, MOUNT)
                derivatives_root = os.path.join(derivatives_root, f'{preplabel}/')
            else:
                derivatives_root = os.path.join(layout.root, f'derivatives/{preplabel}/')
            
            get_derivative = lambda x: pathlib.Path(get_derivative_path(layout, x, 'None', 'epo', '.fif', bids_root, derivatives_root)).as_posix()

            if PARALLELIZE:
                Parallel(n_jobs=external_njobs)(delayed(foo)(x, this_prep, DATASET, get_derivative(x), DEBUG,internal_njobs, args.retry_errors ) for x in meegs)
            else:
                for meeg_file in meegs:
                    ALL_MEEGS.append(meeg_file)

                    if only_total:
                        overall_index+=1
                        continue
                    if single_index is not None and overall_index != single_index:
                        overall_index+=1
                        continue
                    foo(meeg_file, this_prep, DATASET, get_derivative(meeg_file), DEBUG, internal_njobs, args.retry_errors)
                    overall_index+=1
        if only_total:
            print(f'Total number of files: {len(ALL_MEEGS)}')
            for count,eeg in enumerate(ALL_MEEGS):
                print(count,eeg)
            print(f'Total number of files: {len(ALL_MEEGS)}')
            return len(ALL_MEEGS)
if __name__ == '__main__':
    main()
