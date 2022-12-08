import bids
from eeg_raw_to_classification.preprocessing import prepare
from eeg_raw_to_classification.utils import get_derivative_path,save_figs_in_html,save_dict_to_json
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np


FINLAND = {
    'layout':{
        'extension':'.vhdr', 
        'suffix':'eeg', 
        'return_type':'filename',
        'task':'eyesClosed'},
    'ch_names':['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'IZ', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FPZ', 'CPZ', 'POZ', 'OZ'],
    'bids_root':"Y:/datasets/HenryRailo/bids/"
}

DATASET = FINLAND #DEFINE DATASET
MAX_FILES = 1
file_filter = DATASET.get('layout', None)
layout = bids.BIDSLayout(DATASET.get('bids_root', None))
eegs = layout.get(**file_filter)[:MAX_FILES]

print(len(eegs), eegs)
derivatives_root = os.path.join(layout.root,'derivatives/prepare/')
description = layout.get_dataset_description()

for eeg_file in eegs:
    njobs = len(psutil.Process().cpu_affinity())
    print('NJOBS:',njobs)
    reject_eeg,info,figures = prepare(filename = eeg_file, epoch_length = 2, keep_chans = DATASET['ch_names'], downsample = 500, line_noise = 50,ica_method='fastica',njobs=njobs)
    eeg_file_correct = eeg_file.replace('\\','/')
    reject_path = get_derivative_path(layout,eeg_file_correct,'reject','epo','.fif',DATASET['bids_root'],derivatives_root)
    reject_path = reject_path.replace('\\','/')
    figs_path = reject_path.replace('reject_epo.fif','reject_figs.html')
    info_path = reject_path.replace('reject_epo.fif','reject_info.txt')
    fifname = os.path.basename(reject_path)
    fifpath = os.path.dirname(reject_path)
    os.makedirs(fifpath,exist_ok=True)

    # Save prepare info
    save_figs_in_html(figs_path, figures)
    save_dict_to_json(info_path,info)

    # Export preprocessed data

    reject_eeg.save(fifpath + '/' + fifname, split_naming='bids', overwrite=True)


