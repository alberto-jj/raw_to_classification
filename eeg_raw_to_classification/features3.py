import scipy as sp
import neurokit2 as nk2
import numpy as np

def spectrum_multitaper(epochs,multitaper={}):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']

    kwargs = copy.deepcopy(multitaper)
    for k,v in kwargs.items():
        if isinstance(v,str) and 'eval%' in v:
            expression = v.replace('eval%','')
            kwargs[k] = eval(expression)

    space_names = epochs.info['ch_names']

    psd,freqs = psd_array_multitaper(epochs.get_data(), sf, **kwargs)
    fullpsd = psd
    # I think that having the mean here at the last position is confusing
    #psd_mean = np.mean(psd,axis=0,keepdims=True) #epochs, spaces,freqs
    #fullpsd = np.concatenate([psd,psd_mean])
    #assert np.all(fullpsd[-1,:,:]==psd_mean) # Last Epoch is the mean
    epochs_labels = [x for x in range(fullpsd.shape[0])]
    #epochs_labels[-1] = 'EPOCHS-MEAN'
    output = {}
    output['metadata'] = {'type':'PowerSpectrum'}
    output['metadata']['axes']={'epochs':epochs_labels,'spaces':space_names,'frequencies':freqs}
    output['metadata']['order']=('epochs','spaces','frequencies')
    output['values'] = fullpsd
    output['metadata']['times']=epochs.times #TODO: times is not standarized across all features
    return output

filepath = '/home/yorguin/scratch/data/MEG_LSDV2/derivatives/prepDur30Ov15/sub-S1TW/ses-lsd/meg/sub-S1TW_ses-lsd_task-Closed1_desc-None_epo.fif'


def dfa_feature(epochs):
    import scipy.signal
    import scipy.stats as sp_stats
    import neurokit2 as nk2

    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    n_epochs, n_spaces, n_times = data.shape
    dfa_values = np.empty((n_epochs, n_spaces))
    
    for epoch_idx in range(n_epochs):
        for space_idx in range(n_spaces):
            time_series = data[epoch_idx, space_idx, :]
            envelope = np.abs(sp.signal.hilbert(sp_stats.zscore(time_series)))
            dfa, _ = nk2.fractal_dfa(envelope)
            dfa_values[epoch_idx, space_idx] = dfa
    
    epochs_labels = [x for x in range(n_epochs)]
    
    output = {}
    output['metadata'] = {'type': 'DFAFeature'}
    output['metadata']['axes'] = {
        'epochs': epochs_labels,
        'spaces': space_names
    }
    output['metadata']['order'] = ('epochs', 'spaces')
    output['metadata']['times'] = times
    output['values'] = dfa_values
    
    return output

import mne
meg = mne.read_epochs(filepath,preload=True)

dfa = dfa_feature(meg)
breakpoint()
print('end')

