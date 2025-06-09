from mne.io import read_raw
from mne import read_epochs
import mne
import numpy as np
import scipy

def hello(name='World'):
    return f"Hello, {name}!"

def load_meeg(meeg_file, dataset, kwargs={}):
    """Load a MEEG file using MNE-Python."""
    print('Example redefinition of load_meeg function')
    hello('MEEG World')
    print(f"Dataset: {dataset}")
    # You could add special handling based on dataset if needed
    try:
        meeg = read_raw(meeg_file, **kwargs)
    except Exception as e:
        # try to load as epochs
        try:
            meeg = read_epochs(meeg_file, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not load MEEG file {meeg_file}. Error: {e}")
    return meeg

from mne_bids import BIDSPath, write_raw_bids
from sovabids.parsers import parse_from_placeholder

def bidsify(source_path, bids_path, DATASET_CFG, pipeline_cfg=None):
    """
    Convert source_path to BIDS format and save to bids_path.
    
    Parameters
    ----------
    source_path : str
        Path to the source data.
    bids_path : str
        Path where the BIDS dataset will be saved.
    DATASET_CFG : dict
        Configuration dictionary for the dataset.
    """
    print(f"Converting {source_path} to BIDS format at {bids_path}")
    
    # Example of how you might use DATASET_CFG
    rule = DATASET_CFG.get('bidsify', {}).get('pattern', None)
    
    import glob, os, pathlib

    if DATASET_CFG.get('dataset_label','') == 'eegbci': # You could add per dataset handling here
        rule = DATASET_CFG.get('bidsify', {}).get('pattern', 'sub-{subject}_ses-{session}_task-{task}_eeg.edf')
        pattern = os.path.join(source_path, '**', '*.edf')
        files = glob.glob(pattern, recursive=True)
        # Parse the source path for BIDS entities



        for f in files:
            this_file = pathlib.Path(f).as_posix()
            entities = parse_from_placeholder(this_file, pattern=rule)
            # see https://mne.tools/stable/generated/mne.datasets.eegbci.load_data.html#mne.datasets.eegbci.load_data
            """
            1 Baseline, eyes open
            2 Baseline, eyes closed
            3, 7, 11 Motor execution: left vs right hand
            4, 8, 12 Motor imagery: left vs right hand
            5, 9, 13 Motor execution: hands vs feet
            6, 10, 14 Motor imagery: hands vs feet
            """
            run_to_task = {
                'None': 'unknownTask',
                1: 'baselineOpen',
                2: 'baselineClosed',
                3: 'leftHand',
                4: 'rightHand',
                5: 'leftFoot',
                6: 'rightFoot',
                7: 'leftHandImagery',
                8: 'rightHandImagery',
                9: 'leftFootImagery',
                10: 'rightFootImagery'
            }

            subject = entities.get('subject', '')
            task = run_to_task.get(int(entities.get('run', 'None')), 'unknownTask')
            run = entities.get('run', 'None')

            bidsTree = BIDSPath(subject=subject,task=task,run=run, root=bids_path)
            if not os.path.isfile(bidsTree.fpath):
                meeg = load_meeg(this_file, DATASET_CFG)
                write_raw_bids(meeg, bids_path=bidsTree, overwrite=False, format="BrainVision", allow_preload=True)
            else:
                print(f"File {bidsTree.fpath} already exists, skipping.")

    print(f"BIDS conversion complete. Data saved at {bids_path}")


def prepare(filename, dataset_cfg=None, njobs=1, downsample = 500, normalization = False, filter_args=None, epoch_config={}):
    """
    njobs: For the moment ignored, only used to keep the same signature as the original function
    """
    info = {}
    figures = []

    info['filename'] = filename
    info['dataset_cfg'] = dataset_cfg
    info['downsample'] = downsample
    info['normalization'] = normalization
    info['filter_args'] = filter_args
    info['njobs'] = njobs
    info['epoch_config'] = epoch_config

    eegpath = filename
    print('PREPARE FUNCTION OVERRIDENED')


    raw = mne.io.read_raw(eegpath,verbose=False,preload=True)

    if normalization:
        # It is debatable where to normalize the data. Here we do it after PyPREP.
        # Sanity check, the argmin of the zscored data should be the same as the argmin of the raw data
        assert np.argmin(raw.get_data()[0,:])==np.argmin(scipy.stats.zscore(raw.get_data(),axis=1)[0,:])
        raw._data = scipy.stats.zscore(raw.get_data(),axis=1)
        print('AMPLITUDE NORMALIZATION DONE')

    # Filter the data
    if filter_args is not None:
        raw = raw.filter(**filter_args,verbose=False)
        print('FILTERED',end=' ')

    # Extract epochs
    print('EPOCH SEGMENTATION')
    epochs = mne.make_fixed_length_epochs(raw,preload=True,**epoch_config)

    if downsample is not None:
        epochs = epochs.resample(downsample)

    report =  None
    # If you want to generate a MNE report, you can add it here
    return epochs,info,figures, report
