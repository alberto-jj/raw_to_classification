from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from mne.datasets.eegbci import standardize
import numpy as np
import mne
import pandas as pd
from eeg_raw_to_classification.utils import parse_bids
import os
from fooof import FOOOF
import copy

def process_feature(epochs,relevantpath,CFG,feature,pipeline_name):
    featdict = CFG[feature]
    overwrite = featdict['overwrite']
    output = epochs.copy()

    for i_f,stage in enumerate(featdict['chain']):
        input_data = output
        if 'feature' in stage.keys():
            # is a feature that is saved or should be saved
            suffix = stage['feature']
            outputfile = relevantpath.replace('_epo.fif',f'_{suffix}.npy')
            inner_featdict = CFG[suffix]
            if not os.path.isfile(outputfile) or inner_featdict['overwrite']:
                print(f'Feature {suffix} not found')
                output = process_feature(input_data,relevantpath,CFG,suffix,pipeline_name)
                os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                np.save(outputfile,output)
            else:
                print(f'Already Exists:{outputfile}')
                output = np.load(outputfile,allow_pickle=True).item()

        if 'function' in stage.keys():
            inner_featdict = stage
            if i_f == len(featdict['chain'])-1:
                # Last stage, assume we want to save it with the feature name
                suffix = feature
                outputfile = relevantpath.replace('_epo.fif',f'_{suffix}.npy')
                if not os.path.isfile(outputfile) or overwrite:
                    fun = eval(f"{inner_featdict['function']}")
                    output = fun(input_data,**inner_featdict['args'])
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    np.save(outputfile,output)
                else:
                    print(f'Already Exists:{outputfile}')
                    output = np.load(outputfile,allow_pickle=True).item()
            else:
                output = eval(f"inner_featdict['function']")(input_data,**inner_featdict['args'])
        input_data = output
    return output


def agg_numpy(x,numpyfun,axisname='epochs',max_numitem=None): # or give a more complex indexing for items
    x = copy.deepcopy(x)
    # input is the dict from np.load
    # a function like this could help for rois
    # spaces gets mapped to rois for example, and you modify the metadata appropiately
    axis= x['metadata']['order']
    axis = axis.index(axisname)

    if max_numitem is not None:
        x['values'] = np.take(x['values'], indices=range(max_numitem), axis=axis)
    # handle metadata appropriately
    x['values'] = numpyfun(x['values'],axis=axis)
    order = list(x['metadata']['order'])
    order.remove(axisname)
    x['metadata']['order'] = tuple(order)
    del x['metadata']['axes'][axisname]
    return x


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

def spectrum_multitaper(epochs,multitaper={}):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']

    space_names = epochs.info['ch_names']

    psd,freqs = psd_array_multitaper(epochs.get_data(), sf, **multitaper)
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
    return output

def single_fooof(freqs, psds, internal_kwargs={'FOOOF':{},'fit':{}}):
    kwargs = copy.deepcopy(internal_kwargs)
    for key,val in kwargs.items():
        for k,v in val.items():
            if isinstance(v,str) and 'eval%' in v:
                expression = v.replace('eval%','')
                kwargs[key][k] = eval(expression)

    fm = FOOOF(verbose=False,**kwargs['FOOOF'])
    fm.fit(freqs, psds,**kwargs['fit']) # correct, if we used add_data it would ignore for examle freq_range
    return fm

def fooof_from_average(data,agg_fun=None,internal_kwargs={'FOOOF':{},'fit':{}}):
    #data,internal_kwargs={'compute_psd':{},'single_fooof':{}},extra_metadata={},n_jobs=1):
    # we can view this as a new feature or as an aggregate
    if isinstance(data,dict):
      spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
      raise ValueError('Only dict input supported')
    kwargs = copy.deepcopy(internal_kwargs)
    for key,val in kwargs.items():
        for k,v in val.items():
            if isinstance(v,str) and 'eval%' in v:
                expression = v.replace('eval%','')
                kwargs[key][k] = eval(expression)

    if agg_fun is None:
      psd = data['values'].mean(axis=axes.index('epochs'))
    else:
      agg_fun = eval(agg_fun.replace('eval%',''))
      spectra = agg_fun(data)
      psd = spectra['values']

    spaces =  spectra['metadata']['axes']['spaces']
    freqs =   spectra['metadata']['axes']['frequencies']
    # Only the mean
    axes= spectra['metadata']['order']

    output = {}

    values = np.empty(len(spaces),dtype=object)
    output['metadata'] = {'type':'fooofFromAverageSpectrum','kwargs':{'internal_kwargs':internal_kwargs},'freqs':freqs}
    output['metadata']['axes']={'spaces':spaces}
    output['metadata']['order']=('spaces')
    for space in spaces:
        space_idx = spaces.index(space)
        thispsd = np.take(psd,indices=space_idx,axis=axes.index('spaces'))
        fm = single_fooof(freqs, thispsd,kwargs)
        values[space_idx]= fm
    output['values'] = values
    return output

def relative_bandpower(data,bands=BANDS,multitaper={},agg_fun=None):
    # we can view this as a new feature or as an aggregate
    if isinstance(data,dict):
      spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
      spectra = spectrum_multitaper(data,multitaper)
    # Only the mean
    axes= spectra['metadata']['order']

    if agg_fun is None:
      psd = spectra['values'].mean(axis=axes.index('epochs'))
    else:
      agg_fun = eval(agg_fun.replace('eval%',''))
      spectra = agg_fun(spectra)
      psd = spectra['values']
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
