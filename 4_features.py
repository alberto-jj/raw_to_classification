import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from mne.datasets.eegbci import standardize
import mne
from eeg_raw_to_classification import features as feat
from eeg_raw_to_classification.utils import load_yaml,save_dict_to_json
import traceback
datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
DEBUG = True

for featurepipeline in PIPELINE['features']['feature_pipeline_list']:
    print(f'Feature Pipeline: {featurepipeline}')
    for dslabel, DATASET in datasets.items():

        featurepipelineCFG = PIPELINE['features']['feature_pipeline_cfg'][featurepipeline]
        CFG = PIPELINE['features']
        pipeline_name = featurepipeline
        prep_pipeline = featurepipelineCFG['prep_pipeline']
        FEATURE_CFG = CFG['feature_cfg']

        pattern = os.path.join(DATASET.get('bids_root', None),'derivatives',prep_pipeline,'**/*_epo.fif').replace('\\','/')
        eegs = glob.glob(pattern,recursive=True)
        os.makedirs(os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name),exist_ok=True)

        DOWNSAMPLE = featurepipelineCFG['downsample']
        keep_channels = featurepipelineCFG['keep_channels']     # We picked the common channels between datasets for simplicity TODO: Do this in aggregate script

        for eeg_file in eegs:
            print(eeg_file)
            try:
                epochs = mne.read_epochs(eeg_file, preload = True)
                standardize(epochs) #standardize ch_names
                epochs = epochs.filter(**featurepipelineCFG['prefilter']) #bandpassing 1-30Hz, TODO: Should we remove 1Hz as it was previously done?
                epochs = epochs.resample(DOWNSAMPLE)
                if keep_channels:
                    # We picked the common channels between datasets for simplicity
                    epochs = epochs.reorder_channels(keep_channels)

                for feature in featurepipelineCFG['feature_list']:
                    try:
                        print(f'Processing {feature}')
                        derifile = eeg_file.replace(prep_pipeline,pipeline_name)
                        output = feat.process_feature(epochs,derifile,FEATURE_CFG,feature,pipeline_name)
                    except:
                        if DEBUG:
                            raise
                        else:
                            save_dict_to_json(os.path.join(os.path.dirname(eeg_file),f'{feature}_featureserror.txt'),{'error':traceback.format_exc()})
            except:
                if DEBUG:
                    raise
                else:
                    save_dict_to_json(os.path.join(os.path.dirname(eeg_file),f'{feature}_featureserror.txt'),{'error':traceback.format_exc()})
