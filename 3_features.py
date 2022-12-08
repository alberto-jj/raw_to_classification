import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.features import relative_power
from eeg_raw_to_classification.utils import load_yaml

datasets = load_yaml('datasets.yml')
DATASET = datasets['FINLAND']

pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','prepare','**/*_epo.fif').replace('\\','/')
eegs = glob.glob(pattern,recursive=True)
pipeline_name = 'features'
os.makedirs(os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name),exist_ok=True)

NUM_EPOCHS = 18
for eeg_file in eegs:
    # We picked the common channels between datasets for simplicity
    keep_channels = None #["Fp1", "Fp2", "O1", "O2", "FC2", "Fz", "F4", "P3", "Oz", "F7", "CP6", "FC1", "P8", "T7", "Cz", "F3",  "CP2", "FC6",  "C3", "FC5", "CP5", "AF4", "F8", "P4",  "CP1", "C4", "AF3", "T8",  "P7"]
    output = relative_power(eeg_file,downsample=500,num_epochs=NUM_EPOCHS,keep_channels=keep_channels)
    outputfile = eeg_file.replace('_epo.fif','_relativePower.npy').replace('prepare',pipeline_name)
    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
    np.save(outputfile,output)