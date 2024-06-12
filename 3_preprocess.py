import bids
from eeg_raw_to_classification.preprocessing import prepare
from eeg_raw_to_classification.utils import get_derivative_path,save_figs_in_html,save_dict_to_json
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from eeg_raw_to_classification.utils import load_yaml
import traceback
from joblib import delayed, Parallel
datasets = load_yaml('datasets.yml')
cfg = load_yaml('pipeline.yml')
MAX_FILES = 3 #3 #TODO: erase this when ready
external_njobs = 10 # len(psutil.Process().cpu_affinity())

def foo(eeg_file,this_prep,DATASET):
    from eeg_raw_to_classification.utils import get_derivative_path,save_figs_in_html,save_dict_to_json
    import traceback
    import os
    from eeg_raw_to_classification.preprocessing import prepare
    import copy

    layout = bids.BIDSLayout(DATASET.get('bids_root', None))
    njobs = 1 #internal jobs #len(psutil.Process().cpu_affinity())
    print('Internal NJOBS:',njobs)
    reject_path = get_derivative_path(layout,eeg_file,'reject','epo','.fif',DATASET['bids_root'],derivatives_root).replace('\\','/')
    print(eeg_file)
    fifname = os.path.basename(reject_path)
    fifpath = os.path.dirname(reject_path)

    line_noise = DATASET['PowerLineFrequency']
    os.makedirs(fifpath,exist_ok=True)



    if not os.path.isfile(reject_path) or this_prep['overwrite']:
        try:
            reject_eeg,info,figures = prepare(filename = eeg_file,keep_chans = DATASET['ch_names'],line_noise = line_noise,njobs=njobs,**this_prep['prepare'])
            # prepare should standarize name
            figs_path = reject_path.replace('reject_epo.fif','reject_figs.html')
            info_path = reject_path.replace('reject_epo.fif','reject_info.txt')

            # Save prepare info
            save_figs_in_html(figs_path, figures)
            save_dict_to_json(info_path,info)

            # Export preprocessed data

            reject_eeg.save(fifpath + '/' + fifname, split_naming='bids', overwrite=True)
            plt.close('all')
        except Exception:
            print(traceback.format_exc())
            save_dict_to_json(reject_path.replace('.fif','_problem.txt'),{'file':eeg_file,'problem':traceback.format_exc()})
    else:
        print(f'Already Exists:{reject_path}')


for preplabel in cfg['preprocess']['prep_list']:

    for dslabel, DATASET in datasets.items():

        this_prep = cfg['preprocess']['prep_cfg'][preplabel]

        print(f'PREPROCESSING {dslabel} with {preplabel} pipeline')
        file_filter = DATASET.get('raw_layout', None)
        layout = bids.BIDSLayout(DATASET.get('bids_root', None))
        eegs = layout.get(**file_filter)[:MAX_FILES]

        if MAX_FILES > len(eegs):
            limit = len(eegs)
        else:
            limit = MAX_FILES
        eegs = eegs[:limit]
        eegs = [x.replace('\\','/') for x in eegs]
        print(len(eegs), eegs)
        derivatives_root = os.path.join(layout.root,f'derivatives/{preplabel}/')
        #description = layout.get_dataset_description()

        Parallel(n_jobs=external_njobs)(delayed(foo)(eeg_file,this_prep,DATASET) for eeg_file in eegs)
        break