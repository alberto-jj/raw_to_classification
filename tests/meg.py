import os
import scipy.io as sio
import mat73
import mne
import numpy as np
import glob
import scipy



path =r"Y:\computecanada\MEG_tiagabine\MEG_tiagabine\meg_data\TGB_PLA_030709_2.mat"
def loadmat(x,kwargs={}):
    try:
        print(f"Loading {x} with scipy.io.loadmat")
        return sio.loadmat(x,**kwargs)
    except:
        print(f"Loading {x} with mat73.loadmat")
        return mat73.loadmat(x,**kwargs)

meg=loadmat(path,dict(simplify_cells=True))

meg.keys()

type(meg['data'])

meg['data'].keys()

meg['data']['hdr'].keys()

meg['data']['label'].shape

meg['data']['time'].shape

loaded_mat = meg
loaded_mat.keys()
from pprint import pprint

# use file stream for pprinting

with open('meg_data.txt', 'w') as f:
    pprint(meg, stream=f)

pprint(meg)


import mne
import numpy as np

data_dict = loaded_mat['data']  # loaded from scipy.io.loadmat(...)

# 1. Extract data
fsample = data_dict['fsample']  # e.g., 1200
labels = data_dict['label']     # list of channel names
data = data_dict['trial']    # assuming continuous → one array of shape (n_channels, n_samples)
times = data_dict['time']    # should be 1D array of length n_samples

# 2. Create info
info = mne.create_info(
    ch_names=labels.tolist(),  # convert to list if it's not already
    sfreq=fsample,
    ch_types='grad'  # or 'mag' / 'eeg' / etc., depending on the sensor type
)

# 3. Create Raw object
raw = mne.io.RawArray(data, info)

data_dict['trial'].shape

raw.plot(duration=10, scalings='auto', show_scrollbars=True, block=True)


event = data_dict['cfg']['event']  # Dict with keys: 'duration', 'offset', 'sample', 'type', 'value'
trl = data_dict['cfg']['trl']      # Usually a numpy array with shape (n_trials, 3)


import glob

megs=glob.glob("Y:/computecanada/MEG_tiagabine/MEG_tiagabine/meg_data/*.mat")

for meg in megs:
    print(meg)
    loaded_mat = loadmat(meg, dict(simplify_cells=True))
    data_dict = loaded_mat['data']
    
    # fsample = data_dict['fsample']
    # labels = data_dict['label']
    # data = data_dict['trial']
    # times = data_dict['time']
    
    # info = mne.create_info(
    #     ch_names=labels.tolist(),
    #     sfreq=fsample,
    #     ch_types='grad'
    # )
    
    # raw = mne.io.RawArray(data, info)
    
    # raw.plot(duration=10, scalings='auto', show_scrollbars=True, block=True)

    event = data_dict['cfg']['event']  # Dict with keys: 'duration', 'offset', 'sample', 'type', 'value'
    trl = data_dict['cfg']['trl']      # Usually a numpy array with shape (n_trials, 3)
    print(event)
    print(trl)

