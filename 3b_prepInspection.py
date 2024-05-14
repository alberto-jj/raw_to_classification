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
import pandas as pd
datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')


pipeline_name = 'prep-inspection'
outputfolder=PIPELINE['prep_inspection']['path']
## Get the number of epochs
if True:
    SHAPES = []
    EEGS = []
    for dslabel, DATASET in datasets.items():
        CFG = PIPELINE['features']
        pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','prepare','**/*_epo.fif').replace('\\','/')
        eegs = glob.glob(pattern,recursive=True)
        for eeg_file in eegs:
            print(eeg_file)
            epochs = mne.read_epochs(eeg_file, preload = True)
            EEGS.append(eeg_file)
            SHAPES.append(epochs.get_data().shape[0])
    NUM_EPOCHS = np.min(SHAPES)
    epoch_info_source={'shapes':SHAPES,
                       'file':EEGS}
print(f'Using {NUM_EPOCHS} epochs')

dfepochs =pd.DataFrame(epoch_info_source)
outputpath=os.path.join(outputfolder,'epochs_info.csv')
os.makedirs(outputfolder,exist_ok=True)
dfepochs.to_csv(outputpath)

dfepochs.describe().to_csv(os.path.join(outputfolder,'epochs_info_summary.csv'))