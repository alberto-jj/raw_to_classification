import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from mne.datasets.eegbci import standardize
import mne
from eeg_raw_to_classification import features as feat
from eeg_raw_to_classification.utils import load_yaml

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')

for dslabel, DATASET in datasets.items():

    CFG = PIPELINE['features']

    pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','prepare','**/*_epo.fif').replace('\\','/')
    eegs = glob.glob(pattern,recursive=True)
    pipeline_name = 'features'
    os.makedirs(os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name),exist_ok=True)

    NUM_EPOCHS = CFG['num_epochs']
    DOWNSAMPLE = CFG['downsample']
    keep_channels = CFG['keep_channels']     # We picked the common channels between datasets for simplicity TODO: Do this in aggregate script

    for eeg_file in eegs:
        print(eeg_file)

        epochs = mne.read_epochs(eeg_file, preload = True)
        standardize(epochs) #standardize ch_names
        epochs = epochs.filter(**CFG['prefilter']) #bandpassing 1-30Hz, TODO: Should we remove 1Hz as it was previously done?
        epochs = epochs.resample(DOWNSAMPLE)
        if NUM_EPOCHS is None:
            epochs_c = epochs.copy()
        else:
            epochs_c = mne.concatenate_epochs(epochs_list = [epochs[:][range(NUM_EPOCHS)]], add_offset=False, on_mismatch='raise', verbose=None) # 18 WAS THE MINIMAL NUMBER OF EPOCHS ACROSS SUBJECTS
        epochs = epochs_c
        if keep_channels:
            # We picked the common channels between datasets for simplicity
            epochs = epochs.reorder_channels(keep_channels)

        for feature,featdict in CFG['feature_list'].items():
            suffix = feature
            outputfile = eeg_file.replace('_epo.fif',f'_{suffix}.npy').replace('prepare',pipeline_name)

            if not os.path.isfile(outputfile) or featdict['overwrite']:
                fun = eval(f"feat.{featdict['function']}")
                output = fun(epochs,**featdict['args'])
                os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                np.save(outputfile,output)
            else:
                print(f'Already Exists:{outputfile}')
