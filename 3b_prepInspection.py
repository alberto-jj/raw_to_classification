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

PIPELINE = load_yaml(f'pipeline.yml')
datasets = load_yaml(PIPELINE['datasets_file'])

PROJECT = PIPELINE['project']

for preplabel in PIPELINE['prep_inspection']['prep_list']:
    outputfolder=PIPELINE['prep_inspection']['path'].replace('%PROJECT%,PROJECT')
    ## Get the number of epochs
    if 'epochs' in PIPELINE['prep_inspection']['checks']:
        SHAPES = []
        EEGS = []
        DATASETS=[]
        for dslabel, DATASET in datasets.items():
            if DATASET.get('skip',False):
                continue

            CFG = PIPELINE['features']
            pattern = os.path.join(DATASET.get('bids_root', None),'derivatives',preplabel,'**/*_epo.fif').replace('\\','/')
            eegs = glob.glob(pattern,recursive=True)
            for eeg_file in eegs:
                print(eeg_file)
                epochs = mne.read_epochs(eeg_file, preload = True)
                DATASETS.append(dslabel)
                EEGS.append(eeg_file)
                SHAPES.append(epochs.get_data().shape[0])
        if len(SHAPES)==0:
            print('No epochs found')
        else:
            NUM_EPOCHS = np.min(SHAPES)
            epoch_info_source={'shapes':SHAPES,
                            'file':EEGS,
                            'dataset':DATASETS,}
            print(f'Using {NUM_EPOCHS} epochs')

            dfepochs =pd.DataFrame(epoch_info_source)
            outputpath=os.path.join(outputfolder,preplabel)
            outputfile = os.path.join(outputpath,'epochs_info.csv')
            os.makedirs(outputpath,exist_ok=True)
            dfepochs.to_csv(outputfile)

            dfepochs.describe().to_csv(os.path.join(outputpath,'epochs_info_summary.csv'))
            print('Epochs info saved to:',outputpath)