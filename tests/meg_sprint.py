#import mat73
#import mne
import numpy as np
import glob
import scipy.io as sio
import traceback
import os
from pprint import pprint


def loadmat(x,kwargs={}):
    print(f"Loading {x} with scipy.io.loadmat")
    return sio.loadmat(x,**kwargs)


def get_metadata(meg, filepath=None):
    try:
        #kwargs={'squeeze_me': True, 'struct_as_record': False})
        #squeeze_me=True, struct_as_record=False)
        data_dict = meg['data']  # Assuming 'data' is the key for the main data structure
        # 1. Extract data
        fsample = data_dict['fsample']  # e.g., 1200
        labels = data_dict['label']     # list of channel names
        data_shape = data_dict['trial'].shape    # assuming continuous → one array of shape (n_channels, n_samples)
        times = data_dict['time']    # should be 1D array of length n_samples
        event = data_dict['cfg']['event']  # Dict with keys: 'duration', 'offset', 'sample', 'type', 'value'
        trl = data_dict['cfg']['trl']      # Usually a numpy array with shape (n_trials, 3)

        # make dictionary with this information

        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'fsample': fsample,
            'labels': labels,
            'data_shape': data_shape,
            'times': times,
            'event': event,
            'trl': trl
        }
        return metadata
    except Exception as e:
        metadata = {
            'filepath': filepath if filepath else 'not provided',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return metadata


def inspect_meg_data(mats, file='meg_data.txt'):
    amount = len(mats)
    for i, meg_path in enumerate(mats):
        print(f"Processing {i+1}/{amount}: {meg_path}")
        meg_data = loadmat(meg_path, kwargs=dict(simplify_cells=True))
        metadata = get_metadata(meg_data, meg_path)
        with open(file, 'a') as f:
            pprint(metadata, stream=f)



datasets = {
    'MEG_ketamine': '/home/yorguin/scratch/data/MEG_ketamine/meg_data',
    'MEG_perampanel': '/home/yorguin/scratch/data/MEG_perampanel/meg_data',
    'MEG_psilocybin':'/home/yorguin/scratch/data/MEG_psilocybin/meg_data',
    'MEG_tiagabine': '/home/yorguin/scratch/data/MEG_tiagabine/meg_data',
}


FILES_PER_DATASET = 3  # Number of files to inspect per dataset
for dataset_name, dataset_path in datasets.items():
    print(f"Inspecting dataset: {dataset_name} at {dataset_path}")
    mats = glob.glob(os.path.join(dataset_path, '**', '*.mat'), recursive=True)

    print(f"Found {len(mats)} .mat files in {dataset_path}, limiting to {FILES_PER_DATASET} files for inspection.")

    if FILES_PER_DATASET is not None:
        mats = mats[:FILES_PER_DATASET]  if len(mats) > FILES_PER_DATASET else mats

    inspect_meg_data(mats, file=f'{dataset_name}_meg_data.txt')
    print(f"Inspection results saved to {dataset_name}_meg_data.txt")
# Note: This script assumes that the .mat files are structured in a way that allows for the extraction of the 'data' key.
# If the structure is different, you may need to adjust the key names accordingly.