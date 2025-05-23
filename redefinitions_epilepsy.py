import os
import scipy.io as sio
import mat73
import mne
import numpy as np
import glob
import scipy

def loadmat(x,kwargs={}):
    try:
        return sio.loadmat(x,**kwargs)
    except:
        return mat73.loadmat(x,**kwargs)

def get_subdict_from_path(x):
    bidspath = x[:x.find('sub-')]
    acq = parse_bids(x)['acq']
    subinfo = loadmat(os.path.join(bidspath,'subinfo.mat',),dict(simplify_cells=True))['SubInfo'][acq]
    subs = [d['Name'] for d in subinfo]
    sub_idx = subs.index(parse_bids(x)['sub'])
    subdict = subinfo[sub_idx]
    return subdict


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


def get_rundict_from_path(x):
    subdict = get_subdict_from_path(x)
    run = parse_bids(x)['run']
    if isinstance(subdict['SZ'],list):
        subruns =[sz['Name'].replace('_','') for sz in subdict['SZ']]
    else:
        if isinstance(subdict['SZ'],dict):
            subruns = [subdict['SZ']['Name'].replace('_','')]
            subdict['SZ'] = [subdict['SZ']]
        else:
            raise Exception
    idx_run = subruns.index(run)
    rundict = subdict['SZ'][idx_run]
    assert rundict['Name'].replace('_','') == run
    return rundict

def loader_iEEG(x,sfreq=None):
    data = loadmat(x)['F']
    subdict=get_subdict_from_path(x)
    rundict = get_rundict_from_path(x)
    if sfreq is None:
        sfreq1 = subdict['sfreq_orig']
        sfreq = loadmat(x)['Time']
        Ts = sfreq[1]-sfreq[0]
        sfreq = 1/Ts
    ch_types = 'eeg'
    ch_names=rundict['Channel']['SEEG']['Name'].tolist()
    assert data.shape[0] == len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data, info,verbose=False)
    return raw

def get_suffix_from_path(x,suffixext):
    subrundir = os.path.dirname(x)
    files = glob.glob(os.path.join(subrundir,f'*_{suffixext}'))
    files = [x for x in files if os.path.isfile(x)]
    query = parse_bids(x)
    for f in files:
        candidate = parse_bids(f)
        winner = True
        if 'suffix' in candidate:
            del candidate['suffix']
        
        for key,val in candidate.items():
            if key in query:
                if query[key]!=candidate[key]:
                    winner = False
                if not winner:
                    break
        if winner:
            return f
    return None


# signature prepare(filename=raw_file, keep_chans=DATASET['ch_names'], line_noise=line_noise, njobs=njobs, **this_prep['prepare'])
def prepare(filename, line_noise=None, keep_chans=None, downsample = 500, normalization = False, filter_args=None,njobs=1, epoch_config={}):
    """
    keep_chans: is ignored, only used to keep the same signature as the original function
    line_noise: is ignored, only used to keep the same signature as the original function
    njobs: is ignored, only used to keep the same signature as the original function
    """
    print('PREPARE FUNCTION OVERRIDENED')
    eegpath = filename
    if '.mat' in eegpath:
        bads = get_suffix_from_path(eegpath,'flag.mat')
        bads = [bool(aux) for aux in np.squeeze(loadmat(bads)['BadChannel']).tolist()]
        labelresect = get_suffix_from_path(eegpath,'labelresect.mat')
        labelresect = [bool(aux) for aux in np.squeeze(loadmat(labelresect)['LabelResect'].tolist()).tolist()]
        raw = loader_iEEG(eegpath)

    elif '.fif' in eegpath:
        bads = get_suffix_from_path(eegpath,'flag.npy')
        bads = [bool(aux) for aux in np.squeeze(np.load(bads)).tolist()]
        labelresect = get_suffix_from_path(eegpath,'labelresect.npy')
        labelresect = [bool(aux) for aux in np.squeeze(np.load(labelresect).tolist()).tolist()]
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
    epochs = epochs.resample(downsample)

    info = {}
    figures = []

    return epochs,info,figures
