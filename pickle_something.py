import pickle

path=r"Y:\code\raw_to_classification\data\scalingAndFolding\FeaturesChannels@30@prep-defaultprep\noScaling_all\folds-noScaling@all.pkl"
pickle_file = open(path, 'rb')
data = pickle.load(pickle_file)
pickle_file.close()

data.keys()
data[-1].keys()


import numpy as np

path=r"Y:\datasets\Iowa\Dataset\IowaDataset\bids\derivatives\features30@ROI@prep-defaultprep\sub-Control1021\ses-01\eeg\sub-Control1021_ses-01_task-rest_desc-reject_higuchiROI.npy"
data=np.load(path,allow_pickle=True)

data

import mne

path=r"Y:\datasets\Iowa\Dataset\IowaDataset\bids\derivatives\defaultprep\sub-Control1201\ses-01\eeg\sub-Control1201_ses-01_task-rest_desc-reject_epo.fif"
eeg=mne.read_epochs(path,preload=True)

path=r"C:\datasets\Data and Code\Dataset\IowaDataset\Raw data\Control1081.vhdr"
path=r"Y:\datasets\ds004796-download\sub-03\eeg\sub-03_task-rest_eeg.vhdr"
eeg=mne.io.read_raw(path,preload=True)

eeg.annotations
