import argparse
import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from toposort import toposort
from eeg_raw_to_classification.utils import load_yaml
from joblib import delayed, Parallel
import itertools
from graphlib import TopologicalSorter

def get_dependencies(feature, FEATURE_CFG):
    dependencies = []
    depends_on = [f['feature'] for f in FEATURE_CFG[feature]['chain'] if 'feature' in f]
    dependencies += depends_on + list(itertools.chain(*[get_dependencies(f, FEATURE_CFG) for f in depends_on]))
    return dependencies

def foo(eeg_file, DOWNSAMPLE, keep_channels, featurepipelineCFG, FEATURE_CFG, feature, pipeline_name, prep_pipeline, DEBUG=False,retry_errors=False):
    from mne.datasets.eegbci import standardize
    import mne
    import os
    from eeg_raw_to_classification import features as feat
    from eeg_raw_to_classification.utils import load_yaml, save_dict_to_json
    import traceback

    dirname = os.path.dirname(eeg_file)
    finame = os.path.basename(eeg_file)
    derifile = eeg_file.replace(prep_pipeline, pipeline_name)
    errorfile = os.path.join(dirname, f'file-{finame}_feature-{feature}_featureError.txt')
        # inspect first if the file is already processed
    inspect_vector = feat.process_feature(None, derifile, FEATURE_CFG, feature, pipeline_name, inspect_only=True)
    inspect_ = np.all(inspect_vector)
    inspect_error = os.path.isfile(errorfile)

    if inspect_:
        print(f'{feature} already processed for {finame}, skipping')
        return
    if inspect_error and not retry_errors:
        print(f'{feature} had an error, skipping according to retry_errors={retry_errors}')
        return
    try:
        epochs = mne.read_epochs(eeg_file, preload=True)
        standardize(epochs)
        epochs = epochs.filter(**featurepipelineCFG['prefilter'])
        epochs = epochs.resample(DOWNSAMPLE)
        if keep_channels:
                        # We picked the common channels between datasets for simplicity
            epochs = epochs.reorder_channels(keep_channels)

        try:
            print(f'Processing {feature}')
            output = feat.process_feature(epochs, derifile, FEATURE_CFG, feature, pipeline_name)
        except:
            if DEBUG:
                raise
            else:
                save_dict_to_json(os.path.join(dirname, f'file-{finame}_feature-{feature}_featureError.txt'), {'error': traceback.format_exc()})
    except:
        if DEBUG:
            raise
        else:
            save_dict_to_json(os.path.join(dirname, f'file-{finame}_feature-{feature}_featureError.txt'), {'error': traceback.format_exc()})

def main(pipeline_file, external_jobs, debug, parallelize, retry_errors, single_index=None, only_total=False):
    PIPELINE = load_yaml(pipeline_file)

    datasets = load_yaml(PIPELINE['datasets_file'])
    PROJECT = PIPELINE['project']

    for featurepipeline in PIPELINE['features']['feature_pipeline_list']:
        print(f'Feature Pipeline: {featurepipeline}')

        featurepipelineCFG = PIPELINE['features']['feature_pipeline_cfg'][featurepipeline]
        CFG = PIPELINE['features']
        pipeline_name = featurepipeline
        prep_pipeline = featurepipelineCFG['prep_pipeline']
        FEATURE_CFG = CFG['feature_cfg']

    # see which of these features depend on the calculation of other features to avoid redundancy and parallelize (avoid race conditions)
        chains_of_features = {}
        for feature in featurepipelineCFG['feature_list']:
            chains_of_features[feature] = get_dependencies(feature, FEATURE_CFG)
        # assume that if a feature depends on more than one feature, the list is in reverse order of calculation
        # so that means that the last feature does not depend on anything
        # create new keys with this order

        new_chains_of_features = {}
        for feature in chains_of_features:
            if len(chains_of_features[feature]) > 0:
                for ixf, feature_2 in enumerate(chains_of_features[feature][:-1]):
                    new_chains_of_features[feature_2] = [chains_of_features[feature][ixf + 1]]
                new_chains_of_features[feature] = [chains_of_features[feature][0]]
            else:
                new_chains_of_features[feature] = []
        # based on chains of feature how to create levels of parallelization that dont race
        levels = list(toposort(new_chains_of_features))

        DOWNSAMPLE = featurepipelineCFG['downsample']
        keep_channels = featurepipelineCFG['keep_channels']

        all_EEGS = []

        for dslabel, DATASET in datasets.items():
            if DATASET.get('skip', False):
                continue

            pattern = os.path.join(DATASET.get('bids_root', None), 'derivatives', prep_pipeline, '**/*_epo.fif').replace('\\', '/')
            eegs = glob.glob(pattern, recursive=True)
            os.makedirs(os.path.join(DATASET.get('bids_root', None), 'derivatives', pipeline_name), exist_ok=True)

        # we can now parallelize the levels
            for eeg_file in eegs:
                all_EEGS.append(eeg_file)

        if only_total:
            print(f'Total number of files: {len(all_EEGS)}')
            for i,eeg in enumerate(all_EEGS):
                print(i,eeg)
            print(f'Total number of files: {len(all_EEGS)}')
            return len(all_EEGS)
        
        if single_index is not None:
            all_EEGS = [all_EEGS[single_index]]
        if parallelize:
            for level in levels:
                Parallel(n_jobs=external_jobs)(delayed(foo)(eeg_file, DOWNSAMPLE, keep_channels, featurepipelineCFG, FEATURE_CFG, feature, pipeline_name, prep_pipeline, debug, ) for eeg_file in all_EEGS for feature in level)
        else:
            for eeg_file in enumerate(all_EEGS):
                for level in levels:
                    for feature in level:
                        foo(eeg_file, DOWNSAMPLE, keep_channels, featurepipelineCFG, FEATURE_CFG, feature, pipeline_name, prep_pipeline, debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run EEG feature extraction pipeline.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline YAML file.')
    parser.add_argument('--external_jobs', type=int, default=1, help='Number of external jobs for parallel processing.')
    parser.add_argument('--raise_on_error', action='store_true', help='Raise on error if set.')
    parser.add_argument('--retry_errors', action='store_true', help='Retry files that had errors.')
    parser.add_argument('--index', type=int, default=None, help='Index of the file to process. Total index taking into account the dataset outer loop.')
    parser.add_argument('--only_total', action='store_true', help='Just get the total number of files.')

    args = parser.parse_args()
    main(args.pipeline_file, args.external_jobs, args.raise_on_error, args.external_jobs > 1, args.retry_errors, args.index, args.only_total)
