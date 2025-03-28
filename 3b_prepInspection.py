import argparse
import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import mne
from eeg_raw_to_classification import features as feat
from eeg_raw_to_classification.utils import load_yaml, get_path
import traceback
import pandas as pd

def main(pipeline_file):
    PIPELINE = load_yaml(pipeline_file)
    MOUNT = PIPELINE.get('mount', None)
    datasets = load_yaml(get_path(PIPELINE['datasets_file'], MOUNT))
    #load_yaml(PIPELINE['datasets_file'])

    PROJECT = PIPELINE['project']

    for preplabel in PIPELINE['prep_inspection']['prep_list']:
        outputfolder = PIPELINE['prep_inspection']['path']
        outputfolder = get_path(outputfolder, MOUNT).replace('%PROJECT%', PROJECT)

        ## Get the number of epochs
        if 'epochs' in PIPELINE['prep_inspection']['checks']:
            SHAPES = []
            EEGS = []
            DATASETS = []
            for dslabel, DATASET in datasets.items():
                if DATASET.get('skip', False):
                    print(f'Skipping {dslabel} because it is marked as skip')
                    continue

                CFG = PIPELINE['features']
                bids_root = DATASET.get('bids_root', None)
                bids_root = get_path(bids_root, MOUNT)
                pattern = os.path.join(bids_root, 'derivatives', preplabel, '**/*_epo.fif').replace('\\', '/')
                eegs = glob.glob(pattern, recursive=True)
                for eeg_file in eegs:
                    print(eeg_file)
                    epochs = mne.read_epochs(eeg_file, preload=True)
                    DATASETS.append(dslabel)
                    EEGS.append(eeg_file)
                    SHAPES.append(epochs.get_data().shape[0])
            if len(SHAPES) == 0:
                print('No epochs found')
            else:
                NUM_EPOCHS = np.min(SHAPES)
                epoch_info_source = {'shapes': SHAPES,
                                     'file': EEGS,
                                     'dataset': DATASETS,}
                print(f'Using {NUM_EPOCHS} epochs')

                dfepochs = pd.DataFrame(epoch_info_source)
                outputpath = os.path.join(outputfolder, preplabel)
                outputfile = os.path.join(outputpath, 'epochs_info.csv')
                os.makedirs(outputpath, exist_ok=True)
                dfepochs.to_csv(outputfile)

                dfepochs.describe().to_csv(os.path.join(outputpath, 'epochs_info_summary.csv'))
                print('Epochs info saved to:', outputpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the EEG preprocessing inspection.')
    parser.add_argument('pipeline_file', type=str, help='Path to the pipeline.yml file')
    args = parser.parse_args()
    main(args.pipeline_file)