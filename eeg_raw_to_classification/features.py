from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from mne.datasets.eegbci import standardize
import numpy as np
import mne
import pandas as pd
from eeg_raw_to_classification.utils import parse_bids


def spectrum(data, sf, method='multitaper_average', window_sec=None):
    """Compute the spectrum of the signal x.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 2d-array ()
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'multitaper_epochs' or 'multitaper_average'. Default is 'multitaper_average'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    """

    # Compute PSD using multitaper average psd vectors over all epochs, or return the psd vector for each epoch
    if method == 'multitaper_average':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True, low_bias = True,
                                          normalization='full', verbose=0)
        psds_mean = psd.mean(0)
        psd = psds_mean

    elif method == 'multitaper_epochs':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True, low_bias = True,
                                          normalization='full', verbose=0)

    return psd,freqs

def bandpower(psd,freqs,band,relative=True):
    """ 
    Return
    ------
    bp : float
      Absolute or relative/absolute band power after averagind the PSD vectors of each epoch by channel (default). 
      Can also return the relative/absolute bandpower for each epoch if "multitaper_epochs" is selected.
    """
    band = np.asarray(band)
    low, high = band

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

BANDS = {
    'delta' : (1, 4),
    'theta' : (4, 8),
    'alpha' : (8, 13),
    'beta' : (13, 30),
    'pre_alpha' : (5.5, 8),
    'slow_theta' : (4, 5.5),
}

BANDS2 ={
    'alpha1' : (8.5, 10.5),
    'alpha2' : (10.5, 12.5),
    'beta1' : (12.5, 18.5),
    'beta2' : (18.5, 21),
    'beta3' : (21, 30),
}

def spectrum(epochs,multitaper={}):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']

    space_names = epochs.info['ch_names']

    psd,freqs = psd_array_multitaper(epochs.get_data(), sf, **multitaper)

    psd_mean = np.mean(psd,axis=0,keepdims=True) #epochs, spaces,freqs
    fullpsd = np.concatenate([psd,psd_mean])
    assert np.all(fullpsd[-1,:,:]==psd_mean) # Last Epoch is the mean
    epochs_labels = [x for x in range(fullpsd.shape[0])]
    epochs_labels[-1] = 'EPOCHS-MEAN'
    output = {}
    output['metadata'] = {'type':'PowerSpectrum'}
    output['metadata']['axes']={'epochs':epochs_labels,'spaces':space_names,'frequencies':freqs}
    output['metadata']['order']=('epochs','spaces','frequencies')
    output['values'] = fullpsd
    return output

def relative_bandpower(data,bands=BANDS,multitaper={}):
    if isinstance(data,dict):
      spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
      spectra = spectrum(data,multitaper)
    # Only the mean
    psd = spectra['values'][-1,:,:] # last epoch is mean
    spaces =  spectra['metadata']['axes']['spaces']
    freqs =   spectra['metadata']['axes']['frequencies']
    output = {}
    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),len(spaces)))
    output['metadata'] = {'type':'RelativeBandPower','kwargs':{'bands':bands,'multitaper':multitaper}}
    output['metadata']['axes']={'bands':bands_list,'spaces':spaces}
    output['metadata']['order']=('bands','spaces')
    for space in spaces:
        space_idx = spaces.index(space)
        for blabel,brange in bands.items():
            band_idx = bands_list.index(blabel)
            values[band_idx,space_idx]= bandpower(psd[space_idx,:],freqs,brange,True)
    output['values'] = values
    return output
