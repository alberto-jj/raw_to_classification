import bids
from eeg_raw_to_classification.preprocessing import prepare
from eeg_raw_to_classification.utils import get_derivative_path,save_figs_in_html,save_dict_to_json
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from eeg_raw_to_classification.utils import load_yaml

datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
#MAX_FILES = 3 #TODO: erase this when ready

for dslabel, DATASET in datasets.items():
    print(f'PREPROCESSING {dslabel}')
    line_noise = DATASET['PowerLineFrequency']
    file_filter = DATASET.get('raw_layout', None)
    layout = bids.BIDSLayout(DATASET.get('bids_root', None))
    eegs = layout.get(**file_filter)#[:MAX_FILES]
    eegs = [x.replace('\\','/') for x in eegs]
    print(len(eegs), eegs)
    derivatives_root = os.path.join(layout.root,'derivatives/prepare/')
    description = layout.get_dataset_description()

    for eeg_file in eegs:

        njobs = len(psutil.Process().cpu_affinity())
        print('NJOBS:',njobs)
        reject_path = get_derivative_path(layout,eeg_file,'reject','epo','.fif',DATASET['bids_root'],derivatives_root).replace('\\','/')
        print(eeg_file)
        if not os.path.isfile(reject_path) or cfg['preprocess']['overwrite']:

            reject_eeg,info,figures = prepare(filename = eeg_file,keep_chans = DATASET['ch_names'],line_noise = line_noise,njobs=njobs,**cfg['preprocess']['prepare'])
            # prepare should standarize name
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
        else:
            print(f'Already Exists:{reject_path}')



