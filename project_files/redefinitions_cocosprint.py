import os
import scipy.io as sio
import mne
import numpy as np
import glob
import scipy


def parse_bids(bidsname):
    name = os.path.basename(bidsname)
    entities=name.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    if not '-' in suffix:
        d['suffix']=suffix
    else:
        a,b=suffix.split('-')
        d[a]=b
    return d

# signature prepare(filename=raw_file, keep_chans=DATASET['ch_names'], line_noise=line_noise, njobs=njobs, **this_prep['prepare'])
def prepare(filename, line_noise=None, keep_chans=None, downsample = 500, normalization = False, filter_args=None,njobs=1, epoch_config={}):
    """
    keep_chans: is ignored, only used to keep the same signature as the original function
    line_noise: is ignored, only used to keep the same signature as the original function
    njobs: is ignored, only used to keep the same signature as the original function
    """
    info = {}
    figures = []

    info['filename'] = filename
    info['line_noise'] = line_noise
    info['keep_chans'] = keep_chans
    info['downsample'] = downsample
    info['normalization'] = normalization
    info['filter_args'] = filter_args
    info['njobs'] = njobs
    info['epoch_config'] = epoch_config

    eegpath = filename
    print('PREPARE FUNCTION OVERRIDENED')

    if '.fif' in eegpath:
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


    return epochs,info,figures
