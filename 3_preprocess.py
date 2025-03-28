import argparse
import os
import traceback
from joblib import delayed, Parallel
import psutil
import bids
from eeg_raw_to_classification.utils import load_yaml
from eeg_raw_to_classification.utils import get_derivative_path,get_path
import time

def foo(eeg_file, this_prep, DATASET, reject_path, DEBUG, internal_njobs=1, retry_errors=False):
    # imports here to avoid problems with joblib
    import os


    njobs = internal_njobs #internal jobs #len(psutil.Process().cpu_affinity())
    print('Internal NJOBS:', njobs)
    print(eeg_file)
    fifname = os.path.basename(reject_path)
    fifpath = os.path.dirname(reject_path)

    if not os.path.isfile(reject_path) or this_prep['overwrite']:
        if os.path.isfile(reject_path.replace('.fif', '_problem.txt')) and not retry_errors:
            print(f'Error file exists: {reject_path.replace(".fif", "_problem.txt")}, skipping')
            return
        from eeg_raw_to_classification.utils import save_figs_in_html, save_dict_to_json
        from eeg_raw_to_classification.preprocessing import prepare
        import matplotlib.pyplot as plt

        line_noise = DATASET['PowerLineFrequency']
        os.makedirs(fifpath, exist_ok=True)
        try:
            if DATASET.get('precode', None):
                print('Running precode')
                scope = {}
                scope['eeg_file'] = eeg_file
                scope['DATASET'] = DATASET
                scope['this_prep'] = this_prep
                exec(DATASET['precode'], None, scope)
                raw_file = scope['raw_file']
            else:
                raw_file = eeg_file

            reject_eeg, info, figures = prepare(filename=raw_file, keep_chans=DATASET['ch_names'], line_noise=line_noise, njobs=njobs, **this_prep['prepare'])
            figs_path = reject_path.replace('reject_epo.fif', 'reject_figs.html')
            info_path = reject_path.replace('reject_epo.fif', 'reject_info.txt')

            save_figs_in_html(figs_path, figures)
            save_dict_to_json(info_path, info)

            if DATASET.get('postcode', None):
                print('Running postcode')
                exec(DATASET['postcode'])

            reject_eeg.save(fifpath + '/' + fifname, split_naming='bids', overwrite=True)
            plt.close('all')
        except Exception:
            print(traceback.format_exc())
            if DEBUG:
                raise
            else:
                save_dict_to_json(reject_path.replace('.fif', '_problem.txt'), {'file': eeg_file, 'problem': traceback.format_exc()})
    else:
        print(f'Already Exists: {reject_path} or overwrite is False')

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
    ALL_EEGS = []
    for preplabel in cfg['preprocess']['prep_list']:
        overall_index = 0
        # you may try to do this loop outisde (with inner eeg loop) as in 4_features.py,
        # but notice that foo depends on some loop-state variables,
        # so its a bit more complicated, and perhaps not that worth it
        for dslabel, DATASET in datasets.items(): 
            if DATASET.get('skip', False):
                continue

            this_prep = cfg['preprocess']['prep_cfg'][preplabel]

            print(f'PREPROCESSING {dslabel} with {preplabel} pipeline')
            file_filter = DATASET.get('raw_layout', None)

            start_time = time.time()
            bids_root = DATASET.get('bids_root', None)
            bids_root = get_path(bids_root, MOUNT)
            layout = bids.BIDSLayout(bids_root)
            # how to make this faster, it takes too long...
            eegs = layout.get(**file_filter)
            end_time = time.time()
            print(f'Time taken to get EEG files from layout: {end_time - start_time} seconds for dataset {dslabel}')

            if MAX_FILES:
                if MAX_FILES > len(eegs):
                    limit = len(eegs)
                else:
                    limit = MAX_FILES
                eegs = eegs[:limit]
            eegs = [x.replace('\\', '/') for x in eegs]
            print(len(eegs), eegs)
            derivatives_root = os.path.join(layout.root, f'derivatives/{preplabel}/')
            
            get_derivative = lambda x: get_derivative_path(layout, x, 'reject', 'epo', '.fif', bids_root, derivatives_root).replace('\\', '/')

            if PARALLELIZE:
                Parallel(n_jobs=external_njobs)(delayed(foo)(eeg_file, this_prep, DATASET, get_derivative(eeg_file), DEBUG,internal_njobs, args.retry_errors ) for eeg_file in eegs)
            else:
                for eeg_file in eegs:
                    ALL_EEGS.append(eeg_file)

                    if only_total:
                        overall_index+=1
                        continue
                    if single_index is not None and overall_index != single_index:
                        overall_index+=1
                        continue
                    foo(eeg_file, this_prep, DATASET, get_derivative(eeg_file), DEBUG, internal_njobs, args.retry_errors)
                    overall_index+=1
        if only_total:
            print(f'Total number of files: {len(ALL_EEGS)}')
            for count,eeg in enumerate(ALL_EEGS):
                print(count,eeg)
            print(f'Total number of files: {len(ALL_EEGS)}')
            return len(ALL_EEGS)
if __name__ == '__main__':
    main()
