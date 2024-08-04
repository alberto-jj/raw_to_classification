# use sova environment
import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from toposort import toposort
from eeg_raw_to_classification.utils import load_yaml
from joblib import delayed, Parallel
external_njobs = 10 # len(psutil.Process().cpu_affinity())
datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
DEBUG = True
import itertools
def get_dependencies(feature):
    dependencies=[]
    depends_on = [f['feature'] for f in FEATURE_CFG[feature]['chain'] if 'feature' in f]
    dependencies+=depends_on+list(itertools.chain(*[get_dependencies(f) for f in depends_on]))
    #dependencies.reverse()
    return dependencies

def foo(eeg_file,DOWNSAMPLE,keep_channels,featurepipelineCFG,FEATURE_CFG,feature,pipeline_name,DEBUG=False):
    from mne.datasets.eegbci import standardize
    import mne
    import os
    from eeg_raw_to_classification import features as feat
    from eeg_raw_to_classification.utils import load_yaml,save_dict_to_json
    import traceback

    dirname= os.path.dirname(eeg_file)
    finame= os.path.basename(eeg_file)

    try:
        epochs = mne.read_epochs(eeg_file, preload = True)
        standardize(epochs) #standardize ch_names
        epochs = epochs.filter(**featurepipelineCFG['prefilter']) #bandpassing 1-30Hz, TODO: Should we remove 1Hz as it was previously done?
        epochs = epochs.resample(DOWNSAMPLE)
        if keep_channels:
            # We picked the common channels between datasets for simplicity
            epochs = epochs.reorder_channels(keep_channels)

        try:
            print(f'Processing {feature}')
            derifile = eeg_file.replace(prep_pipeline,pipeline_name)
            output = feat.process_feature(epochs,derifile,FEATURE_CFG,feature,pipeline_name)
        except:
            if DEBUG:
                raise
            else:
                save_dict_to_json(os.path.join(dirname,f'file-{finame}_feature-{feature}_featureError.txt'),{'error':traceback.format_exc()})
    except:
        if DEBUG:
            raise
        else:
            save_dict_to_json(os.path.join(dirname,f'file-{finame}_feature-{feature}_featureError.txt'),{'error':traceback.format_exc()})

from graphlib import TopologicalSorter
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
        chains_of_features[feature] = get_dependencies(feature)
    # assume that if a feature depends on more than one feature, the list is in reverse order of calculation
    # so that means that the last feature does not depend on anything
    # create new keys with this order
    new_chains_of_features = {}
    for feature in chains_of_features:
        if len(chains_of_features[feature])>0:
            for ixf,feature_2 in enumerate(chains_of_features[feature][:-1]):
                new_chains_of_features[feature_2]=[chains_of_features[feature][ixf+1]]
            new_chains_of_features[feature]=[chains_of_features[feature][0]]
    
    # based on chains of feature how to create levels of parallelization that dont race
    levels = list(toposort(new_chains_of_features))

    DOWNSAMPLE = featurepipelineCFG['downsample']
    keep_channels = featurepipelineCFG['keep_channels']     # We picked the common channels between datasets for simplicity TODO: Do this in aggregate script

    all_EEGS=[]

    for dslabel, DATASET in datasets.items():
        pattern = os.path.join(DATASET.get('bids_root', None),'derivatives',prep_pipeline,'**/*_epo.fif').replace('\\','/')
        eegs = glob.glob(pattern,recursive=True)
        os.makedirs(os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name),exist_ok=True)

        # we can now parallelize the levels
        for eeg_file in eegs:
            all_EEGS.append(eeg_file)
    # begin parallelization
    for level in levels:
        pairs=[(eeg_file,feature) for eeg_file in all_EEGS for feature in level]
        Parallel(n_jobs=external_njobs)(delayed(foo)(eeg_file,DOWNSAMPLE,keep_channels,featurepipelineCFG,FEATURE_CFG,feature,pipeline_name,DEBUG) for eeg_file in all_EEGS for feature in level)