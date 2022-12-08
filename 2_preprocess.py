import bids
from eeg_raw_to_classification.preprocessing import prepare
from eeg_raw_to_classification.utils import get_derivative_path,save_figs_in_html,save_dict_to_json
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from eeg_raw_to_classification.utils import load_yaml

datasets = load_yaml('datasets.yml')
DATASET = datasets['FINLAND']
MAX_FILES = 3
file_filter = DATASET.get('raw_layout', None)
layout = bids.BIDSLayout(DATASET.get('bids_root', None))
eegs = layout.get(**file_filter)[:MAX_FILES]

print(len(eegs), eegs)
derivatives_root = os.path.join(layout.root,'derivatives/prepare/')
description = layout.get_dataset_description()

for eeg_file in eegs:
    njobs = len(psutil.Process().cpu_affinity())
    print('NJOBS:',njobs)
    reject_eeg,info,figures = prepare(filename = eeg_file, epoch_length = 2, keep_chans = DATASET['ch_names'], downsample = 500, line_noise = 50,ica_method='fastica',njobs=njobs,skip_prep=False)
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
    plt.close('all')


