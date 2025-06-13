
import scipy as sp
import neurokit2 as nk2
import numpy as np
import copy
from mne.time_frequency import psd_array_multitaper
import time

from scipy.integrate import simpson as simps
from mne.time_frequency import psd_array_multitaper,psd_array_welch
from mne.datasets.eegbci import standardize
import numpy as np
import mne
import pandas as pd
from eeg_raw_to_classification.utils import parse_bids,extract_item,agg_numpy
import os
from fooof import FOOOF
import copy
from antropy import detrended_fluctuation,lziv_complexity,sample_entropy,spectral_entropy,app_entropy,hjorth_params,num_zerocross,perm_entropy,svd_entropy,higuchi_fd,katz_fd,petrosian_fd
from neurokit2 import entropy_multiscale
import copy
import neurokit2 as nk2

DEBUG = False
def process_feature(epochs,relevantpath,CFG,feature,pipeline_name,inspect_only=False):
    featdict = CFG[feature]
    overwrite = featdict['overwrite']
    if not inspect_only:
        if epochs is None:
            output = None # this should only happen if we have already computed the next to last part of the chain
        else:
            output = epochs.copy()
    else:
        output = None
    inspect_only_output = []
    #breakpoint()
    for i_f,stage in enumerate(featdict['chain']):
        input_data = output
        if 'feature' in stage.keys():
            # is a feature that is saved or should be saved
            suffix = stage['feature']
            outputfile = relevantpath.replace('_epo.fif',f'_{suffix}.npy')
            inner_featdict = CFG[suffix]
            if not os.path.isfile(outputfile) or inner_featdict['overwrite']:
                if inspect_only:
                    inspect_only_output.append({'status':False, 'feature_file':outputfile, 'feature':suffix})
                    continue
                print(f'Feature {suffix} not found')
                output = process_feature(input_data,relevantpath,CFG,suffix,pipeline_name,inspect_only)
                os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                np.save(outputfile,output)
            else:
                if inspect_only:
                    inspect_only_output.append({'status':True, 'feature_file':outputfile, 'feature':suffix})
                    continue
                print(f'Already Exists:{outputfile}')
                output = np.load(outputfile,allow_pickle=True).item()
                inspect_only_output.append({'status':True, 'feature_file':outputfile, 'feature':suffix})

        if 'function' in stage.keys():
            inner_featdict = stage
            if i_f == len(featdict['chain'])-1:
                # Last stage, assume we want to save it with the feature name
                suffix = feature
                outputfile = relevantpath.replace('_epo.fif',f'_{suffix}.npy')
                if not os.path.isfile(outputfile) or overwrite:
                    if inspect_only:
                        inspect_only_output.append({'status':False, 'feature_file':outputfile, 'feature':suffix})
                        continue
                    fun = eval(f"{inner_featdict['function']}")
                    if isinstance(fun,str):
                        fun=eval(fun)
                    output = fun(input_data,**inner_featdict['args'])
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    np.save(outputfile,output)
                    
                else:
                    if inspect_only:
                        inspect_only_output.append({'status':True, 'feature_file':outputfile, 'feature':suffix})
                        continue
                    print(f'Already Exists:{outputfile}')
                    output = np.load(outputfile,allow_pickle=True).item()
            else:
                if inspect_only:
                    continue
                innerfun=eval(f"inner_featdict['function']")
                innerfun=eval(innerfun)
                output = innerfun(input_data,**inner_featdict['args'])
        input_data = output
    if inspect_only:
        output=inspect_only_output
    return output

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

    kwargs = copy.deepcopy(multitaper)
    for k,v in kwargs.items():
        if isinstance(v,str) and 'eval%' in v:
            expression = v.replace('eval%','')
            kwargs[k] = eval(expression)

    space_names = epochs.info['ch_names']

    start = time.time()

    psd,freqs = psd_array_multitaper(epochs.get_data(), sf, **kwargs)
    end = time.time()
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
    output['metadata']['timings'] = {end - start}
    return output

def spectrum_welch(epochs,welch={}):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']

    space_names = epochs.info['ch_names']
    #breakpoint()
    kwargs = copy.deepcopy(welch)
    for k,v in kwargs.items():
        if isinstance(v,str) and 'eval%' in v:
            expression = v.replace('eval%','')
            kwargs[k] = eval(expression)

    psd,freqs = psd_array_welch(epochs.get_data(), sf, **kwargs)
    fullpsd = psd
    # I think that having the mean here at the last position is confusing
    #psd_mean = np.mean(psd,axis=0,keepdims=True) #epochs, spaces,freqs
    #fullpsd = np.concatenate([psd,psd_mean])
    #assert np.all(fullpsd[-1,:,:]==psd_mean) # Last Epoch is the mean
    epochs_labels = [x for x in range(fullpsd.shape[0])]
    #epochs_labels[-1] = 'EPOCHS-MEAN'
    output = {}
    output['metadata'] = {'type':'PowerSpectrumWelch'}
    output['metadata']['axes']={'epochs':epochs_labels,'spaces':space_names,'frequencies':freqs}
    output['metadata']['order']=('epochs','spaces','frequencies')
    output['values'] = fullpsd
    output['metadata']['times']=epochs.times #TODO: times is not standarized across all features
    return output

    

def single_fooof(freqs, psds, internal_kwargs={'FOOOF':{},'fit':{}}):
    kwargs = copy.deepcopy(internal_kwargs)
    for key,val in kwargs.items():
        if isinstance(val,dict):
            for k,v in val.items():
                if isinstance(v,str) and 'eval%' in v:
                    expression = v.replace('eval%','')
                    kwargs[key][k] = eval(expression)

    fm = FOOOF(verbose=False,**kwargs['FOOOF'])
    fm.fit(freqs, psds,**kwargs['fit']) # correct, if we used add_data it would ignore for examle freq_range
    return fm

import time
import math

def fooof_from_average(data,internal_kwargs={'FOOOF':{},'fit':{}, 'freq_res':None}):
    #data,internal_kwargs={'compute_psd':{},'single_fooof':{}},extra_metadata={},n_jobs=1):
    # we can view this as a new feature or as an aggregate
    if isinstance(data,dict):
      spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
      raise ValueError('Only dict input supported')
    kwargs = copy.deepcopy(internal_kwargs)
    for key,val in kwargs.items():
        if isinstance(val,dict):
            for k,v in val.items():
                if isinstance(v,str) and 'eval%' in v:
                    expression = v.replace('eval%','')
                    kwargs[key][k] = eval(expression)

    spaces =  spectra['metadata']['axes']['spaces']
    freqs =   spectra['metadata']['axes']['frequencies']
    # Only the mean
    axes= spectra['metadata']['order']

    output = {}

    values = np.empty(len(spaces),dtype=object)
    timings = np.empty(len(spaces),dtype=float) * np.nan
    output['metadata'] = {'type':'fooofFromAverageSpectrum','kwargs':{'internal_kwargs':internal_kwargs},'freqs':freqs}
    output['metadata']['axes']={'spaces':spaces}
    output['metadata']['order']=('spaces')
    psd = spectra['values']
    if 'freq_res' in internal_kwargs:
        freq_res = internal_kwargs['freq_res']
    if freq_res is not None:
        # undersample the spectra
        current_res = freqs[1] - freqs[0]
        assert current_res <= freq_res
        print(f"Current resolution: {current_res}, Desired resolution: {freq_res}")
        step = max(1, math.ceil(freq_res / current_res))  # use ceil to ensure >= desired res
        print(f"Downsampling step: {step}, to achieve resolution {freq_res} Hz")
    else:
        step = 1
    for space in spaces:
        space_idx = spaces.index(space)
        thispsd = np.take(psd,indices=space_idx,axis=axes.index('spaces'))
        # Downsample
        freqs_downsampled = freqs[::step]
        thispsd_downsampled = thispsd[::step]
        start = time.time()
        fm = single_fooof(freqs_downsampled, thispsd_downsampled, kwargs)
        end = time.time()
        timings[space_idx] = end - start
        values[space_idx]= fm
        if DEBUG:
            print(f"Processed {space} in {timings[space_idx]} seconds")
    output['values'] = values
    output['metadata']['timings'] = timings
    output['metadata']['total_time'] = np.nansum(timings)
    return output

def roi_mapping_alberto(x):
    anterior = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"]
    central = ["T7", "T8", "C3", "C4", "Cz"]
    posterior = ["P3", "P4", "P7", "P8", "O1", "O2"]
    if x in anterior:
        return 'anterior'
    elif x in central:
        return 'central'
    elif x in posterior:
        return 'posterior'
    else:
        return 'none'

def roi_mapping_yorguin(x):
    if x[0] == 'F' and x[1].isdigit():
        return 'frontal'
    elif x[0] == 'C' and x[1].isdigit():
        return 'central'
    elif x[0]== 'P' and x[1].isdigit():
        return 'parietal'
    elif x[0]== 'O' and x[1].isdigit():
        return 'occipital'
    else:
        return 'none'

default_mapping=roi_mapping_alberto


def roi_aggregator(data,mapping=default_mapping,numpyfun=None,axisname='spaces',ignore=['none']):
    # maybe a similar strategy could also work when the epochs are inhomoegeneous (different tasks?)
    # numpy fun should be a function that takes data,axis and keepdims
    # we can view this as a new feature or as an aggregate
    # assume dict output format from other features

    # first map spaces to rois
    spaces =  data['metadata']['axes'][axisname]
    scope = {}
    if isinstance(mapping,str):
        exec(mapping, scope)
    else:
        scope['roi_mapping'] = mapping
    roi_mapping = scope['roi_mapping']  # Access the function from the scope
    rois = [roi_mapping(x) for x in spaces]
    rois = list(set(rois) - set(ignore))

    if numpyfun is None:
        numpyfun = np.mean
    elif isinstance(numpyfun,str):
        numpyfun = eval(numpyfun.replace('eval%',''))
    # Do aggregation for each roi

    new_values=[]
    new_spaces=[]
    axes= data['metadata']['order']
    for roi in rois:
        roi_idx = [i for i,x in enumerate(spaces) if roi_mapping(x)==roi]
        thisdata = np.take(data['values'],indices=roi_idx,axis=axes.index(axisname))
        new_values.append(numpyfun(thisdata,axis=axes.index(axisname),keepdims=True))

    new_data = np.concatenate(new_values,axis=axes.index(axisname))
    data['values'] = new_data
    data['metadata']['axes'][axisname] = rois

    return data

def relative_bandpower(data,bands=BANDS):
    # we can view this as a new feature or as an aggregate
    if isinstance(data,dict):
        spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
        raise ValueError('Only dict input supported')
    # Only the mean
    axes= spectra['metadata']['order']

    psd = spectra['values']
    spaces =  spectra['metadata']['axes']['spaces']
    freqs =   spectra['metadata']['axes']['frequencies']
    output = {}
    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),len(spaces)))
    output['metadata'] = {'type':'RelativeBandPower','kwargs':{'bands':bands}}
    output['metadata']['axes']={'bands':bands_list,'spaces':spaces}
    output['metadata']['order']=('bands','spaces')
    for space in spaces:
        space_idx = spaces.index(space)
        for blabel,brange in bands.items():
            band_idx = bands_list.index(blabel)
            values[band_idx,space_idx]= bandpower(psd[space_idx,:],freqs,brange,True)
    output['values'] = values
    return output


def relative_bandpower_from_fooof(data,bands=BANDS):
    # we can view this as a new feature or as an aggregate
    if isinstance(data,dict):
        spectra = data # Assume we have the output of spectrum() if input is dict
    else: # Else assume epochs mne object
        raise ValueError('Only dict input supported')
    # Only the mean
    axes= spectra['metadata']['order']

    psd = spectra['values']
    spaces =  spectra['metadata']['axes']['spaces']
    output = {}
    bands_list = list(bands.keys())
    values = np.empty((len(bands_list),len(spaces)))
    output['metadata'] = {'type':'RelativeBandPower','kwargs':{'bands':bands}}
    output['metadata']['axes']={'bands':bands_list,'spaces':spaces}
    output['metadata']['order']=('bands','spaces')
    for space in spaces:
        space_idx = spaces.index(space)
        for blabel,brange in bands.items():
            band_idx = bands_list.index(blabel)
            fo = psd[space_idx]
            x = np.power(10,fo.power_spectrum)-np.power(10,fo._ap_fit)
            freqs = fo.freqs
            values[band_idx,space_idx]= bandpower(x,freqs,brange,True)
    output['values'] = values
    return output

def band_ratios(data):
    ## Assume output of bandpower
    import itertools
    if isinstance(data,dict):
        band_data = data # Assume we have the output of bandpower() if input is dict
    else: # Else assume epochs mne object
        raise ValueError('Only dict input supported')

    axes = band_data['metadata']['order']
    spaces = band_data['metadata']['axes']['spaces']
    bands_list = band_data['metadata']['axes']['bands']
    bands_list = list(bands_list)

    # get all permutations of bands (order matters)
    band_combinations = list(itertools.permutations(bands_list, 2))

    output = {}
    values = np.empty((len(band_combinations),len(spaces)))
    output['metadata'] = {'type':'BandsRatio','kwargs':{'bands':bands_list}}
    output['metadata']['axes']={'bands_pairs':band_combinations,'spaces':spaces}
    output['metadata']['order']=('bands_pairs','spaces')

    for space in spaces:
        space_idx = spaces.index(space)
        for btop,bbottom in band_combinations:
            band_idx_top = bands_list.index(btop)
            band_idx_bottom = bands_list.index(bbottom)

            if band_data['values'][band_idx_bottom,space_idx] == 0:
                values[band_idx_top,space_idx] = np.nan
            else:
                values[band_idx_top,space_idx] = band_data['values'][band_idx_top,space_idx] / band_data['values'][band_idx_bottom,space_idx]

    output['values'] = values
    return output

#%% Antropy Features

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + ''.join(i.capitalize() for i in s[1:])

funs = ['detrended_fluctuation','lziv_complexity','sample_entropy','spectral_entropy','app_entropy','hjorth_params','num_zerocross','perm_entropy','svd_entropy','higuchi_fd','katz_fd','petrosian_fd','entropy_multiscale']#,'chaos_pipeline']
labels = [to_camel_case(x) for x in funs]



fun_template="""
def compute_%label%(eeg, suffix='%label%',internal_kwargs=dict(),extra_metadata={},prefoo=lambda x: x):

    if isinstance(prefoo,str) and 'eval%' in prefoo:
        prefoo=eval(prefoo.replace('eval%',''))

    kwargs = copy.deepcopy(internal_kwargs)
    for key,val in kwargs.items():
        for k,v in val.items():
            if isinstance(v,str) and 'eval%' in v:
                expression = v.replace('eval%','')
                kwargs[key][k] = eval(expression)

    if len(eeg.get_data().shape)==3:
        nepochs = eeg.get_data().shape[0]
        data = eeg.get_data()
    else:
        nepochs = 1
        data = eeg.get_data()[None,:,:]

    epochs = []
    values = np.empty((nepochs,len(eeg.ch_names)),dtype=object)
    timings = np.empty((nepochs,len(eeg.ch_names)),dtype=float)
    for e in range(nepochs):
        for i in range(len(eeg.ch_names)):
            start = time.time()
            result = %fun%(prefoo(data[e,i,:]),**kwargs['%fun%'])
            end = time.time()
            if isinstance(result,tuple):
                result = {i:v for i,v in enumerate(result)}
            values[e,i] = result
            timings[e,i] = end - start
            if DEBUG:
                print(f"Processed epoch {e}, channel {i} '{eeg.ch_names[i]}' in {timings[e,i]} seconds")

    total_time = np.nansum(timings)
    if len(eeg.get_data().shape)==3:
        axes = {'epochs':list(range(eeg.get_data().shape[0])),'spaces':eeg.info['ch_names']}
        order = ('epochs','spaces')
    else:
        axes = {'spaces':eeg.info['ch_names']}
        order = ('spaces')
        values = np.squeeze(values)

    output = {}
    output['metadata']={'type':suffix}
    output['metadata']['axes']=axes
    output['metadata']['order']=order
    output['metadata']['times']=eeg.times
    output['metadata']['timings'] = timings
    output['metadata']['total_time'] = total_time
    output['values']= values
    output['metadata'].update(extra_metadata)
    return output
"""
antropy_definitions = [fun_template.replace('%label%',label).replace('%fun%',fun) for label,fun in zip(labels,funs)]
for foo in antropy_definitions:
    exec(foo)



from copy import deepcopy
import numpy as np
from mne import read_epochs
from phyid.calculate import calc_PhiID
from mne import BaseEpochs


INFORMATION_DYNAMICS_METRICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "DownwardCausation": ["sty", "stx", "str"],
    "UpwardCausation": ["xts", "yts", "rts"],
}

IIT_METRICS = {
    "InformationStorage": ["xtx", "yty", "rtr", "sts"],
    "TransferEntropy": ["xty", "xtr", "str", "sty"],
    "CausalDensity": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "IntegratedInformation": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


{
"PhiID": {
        "tau": dict(FloatParam=(5, 1, 100, dict(doc="Time lag for the PhiID algorithm"))),
        "kind": dict(StringParam=(
            "gaussian",
            dict(options=["gaussian", "discrete"]),
            dict(doc="Kind of data (continuous Gaussian or discrete-binarized)"),
        )),
        "redudancy": dict(StringParam=("MMI", dict(options=["MMI", "CCS"]), dict(doc="Redundancy measure to use"))),
    }
}
def single_atoms(epochs, tau=5,redundancy='MMI', kind='gaussian', channel_labels=None):


    if isinstance(epochs, BaseEpochs):
        #breakpoint()
        # drop channels not starting with 'M'
        # custom code for cocosprint cocodelics project
        idx_to_keep = [i for i,ch in enumerate(epochs.ch_names) if ch.startswith('M')]
        chans_to_keep = [ch for i,ch in enumerate(epochs.ch_names) if ch.startswith('M')]

        matrix = epochs.get_data()
        matrix = matrix[:, idx_to_keep, :]  # shape (n_epochs, n_channels, n_time)
        channel_labels = chans_to_keep

    else:
        # Assume input is a 3D numpy array: epochs x channels x timepoints
        matrix = np.asarray(epochs, dtype=float)
        if matrix.ndim != 3:
            raise ValueError("Input must be a 3D numpy array (epochs x channels x timepoints).")

    n_epochs, n_channels, n_time = matrix.shape

    # If channel_labels is not provided, create default labels
    if channel_labels is None:
        channel_labels = [f"ch{i}" for i in range(n_channels)]


    # List of atom names in fixed order
    atom_names = [
        "rtr",
        "rtx",
        "rty",
        "rts",
        "xtr",
        "xtx",
        "xty",
        "xts",
        "ytr",
        "ytx",
        "yty",
        "yts",
        "str",
        "stx",
        "sty",
        "sts",
    ]
    n_atoms = len(atom_names)

    atoms_vals = np.zeros((n_epochs, n_channels, n_atoms, n_time - tau), dtype=np.float64)

    timings = np.empty((n_epochs, n_channels), dtype=float) * np.nan

    # Compute PhiID for each channel vs. the mean of all other channels
    for e in range(n_epochs):
        data = matrix[e, :, :]  # shape (n_channels, n_time)
        for i in range(n_channels):
            start = time.time()
            src = data[i]
            if n_channels > 1:
                # target is average of all other channels
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                # only one channel: create trg as the timelagged version of src
                #trg = np.roll(src, tau)
                # 
                trg = src #Antoine
                #raise ValueError("Only one channel found, cannot compute PhiID with a single channel (TODO).")

            # Run the PhiID calculation
            try:
                atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
                #assert [k for k in atoms_res.keys()] == atom_names
                atoms_res['rtr'].shape
                for key in atoms_res.keys():
                    vals = atoms_res[key]
                    atoms_vals[e, i, atom_names.index(key), :] = vals# should be size [:n_time - tau]
            except Exception as ex:
                print(f"Error processing epoch {e}, channel {i} '{channel_labels[i]}': {ex}")
                # Fill with NaNs if there's an error
                atoms_vals[e, i, :, :] = np.nan
            end = time.time()
            timings[e, i] = end - start
            if DEBUG:
                print(f"Processed epoch {e}, channel {i} '{channel_labels[i]}' in {timings[e, i]} seconds")
    total_time = np.nansum(timings)
    # Build metadata
    output = {}
    output['metadata'] = {'type': 'Atoms'}

    # Define axis labels
    epoch_labels = [e for e in range(n_epochs)]
    space_names = channel_labels
    atom_names_order = atom_names
    time_axis = np.arange(n_time - tau) / epochs.info['sfreq']  # convert to seconds (optional)

    # Axes dict
    output['metadata']['axes'] = {
        'epochs': epoch_labels,
        'spaces': space_names,
        'atoms': atom_names_order,
        'times': time_axis,
        'timings': timings,
    }

    # Order of axes (matches shape of atoms_vals)
    output['metadata']['order'] = ('epochs', 'spaces', 'atoms', 'times')

    # Store values
    output['values'] = atoms_vals


    return output


def atoms_results(atoms, key='InformationDynamics', aggregation_mode='mean-sum'):
    """
    aggregation_mode:
        - 'sum-mean': sum across atoms at each timepoint, then mean over time (principled)
        - 'mean-sum': mean each atom first, then sum the means (to exactly match original process())
    """
    # assume atoms is a dict with keys 'values' and 'metadata' from single_atoms

    n_epochs = len(atoms['metadata']['axes']['epochs'])
    n_channels = len(atoms['metadata']['axes']['spaces'])
    n_atoms = len(atoms['metadata']['axes']['atoms'])
    n_time = len(atoms['metadata']['axes']['times'])

    assert atoms['metadata']['order'] == ('epochs', 'spaces', 'atoms', 'times')

    # Build output dict in spectrum_multitaper style
    epochs_labels = atoms['metadata']['axes']['epochs']
    space_names = atoms['metadata']['axes']['spaces']
    atom_names = atoms['metadata']['axes']['atoms']

    output = {}

    if key == 'InformationDynamics':
        the_final_vals = np.zeros((n_epochs, n_channels, len(INFORMATION_DYNAMICS_METRICS)), dtype=np.float64)
        metric_names = list(INFORMATION_DYNAMICS_METRICS.keys())
        output['metadata'] = {'type': 'InformationDynamics'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    elif key == 'IntegratedInformationDecomposition':
        the_final_vals = np.zeros((n_epochs,n_channels, n_atoms), dtype=np.float64)
        metric_names = atom_names
        output['metadata'] = {'type': 'IntegratedInformationDecomposition'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names,
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    elif key == 'IntegratedInformationTheory':
        the_final_vals = np.zeros((n_epochs, n_channels, len(IIT_METRICS)), dtype=np.float64)
        metric_names = list(IIT_METRICS.keys())
        output['metadata'] = {'type': 'IntegratedInformationTheory'}
        output['metadata']['axes'] = {
            'epochs': epochs_labels,
            'spaces': space_names,
            'metrics': metric_names
        }
        output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    
    for e in range(n_epochs):
        for c in range(n_channels):
            for j, name in enumerate(metric_names):
                values = atoms['values'][e, c, :, :]

                if key == 'InformationDynamics':
                    # Get the indices of the atoms in the INFORMATION_DYNAMICS_METRICS dict
                    atom_indices = [atom_names.index(atom) for atom in INFORMATION_DYNAMICS_METRICS[name]]
                    # Sum the values of the atoms and average over time

                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices,:], axis=0)))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(np.sum([atoms['values'][e, c, atom_idx, :].mean() for atom_idx in atom_indices]))
                elif key == 'IntegratedInformationDecomposition':
                    # For PhiID, we just take the mean of the atom values
                    atom_index = atom_names.index(name)
                    # NOTE actually this two seem to be the same ??...
                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(values[atom_index, :]))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(atoms['values'][e, c, atom_index, :].mean())
                elif key == 'IntegratedInformationTheory':
                    atom_indices = [atom_names.index(atom) for atom in IIT_METRICS[name]]
                    # Sum the values of the atoms and average over time
                    if aggregation_mode == 'sum-mean':
                        the_final_vals[e, c, j] = float(np.mean(np.sum(values[atom_indices,:], axis=0)))
                    elif aggregation_mode == 'mean-sum':
                        the_final_vals[e, c, j] = float(np.sum([atoms['values'][e, c, atom_idx, :].mean() for atom_idx in atom_indices]))
                    if name == "Integrated information":
                        rtr_index = atom_names.index("rtr")
                        if aggregation_mode == 'sum-mean':
                            the_final_vals[e, c, j] -= float(np.mean(values[rtr_index, :]))
                        elif aggregation_mode == 'mean-sum':
                            the_final_vals[e, c, j] -= float(atoms['values'][e, c, rtr_index, :].mean())

    # Store the computed values
    output['values'] = the_final_vals
    return output


def process(matrix, tau=5, redundancy="MMI", kind="gaussian", channel_labels=None):
        # If no input, do nothing
        # Ensure data is a 2D array: channels x timepoints
        data = np.asarray(matrix, dtype=float)

        n_channels, n_time = data.shape


        # List of atom names in fixed order
        atom_names = [
            "rtr",
            "rtx",
            "rty",
            "rts",
            "xtr",
            "xtx",
            "xty",
            "xts",
            "ytr",
            "ytx",
            "yty",
            "yts",
            "str",
            "stx",
            "sty",
            "sts",
        ]
        n_atoms = len(atom_names)

        # Prepare output array: one row per channel, one col per atom
        PhiID_vals = np.zeros((n_channels, n_atoms), dtype=np.float64)
        inf_dyn_vals = np.zeros((n_channels, len(INFORMATION_DYNAMICS_METRICS)), dtype=np.float64)
        IIT_vals = np.zeros((n_channels, len(IIT_METRICS)), dtype=np.float64)
        # Compute PhiID for each channel vs. the mean of all other channels
        for i in range(n_channels):
            src = data[i]
            if n_channels > 1:
                # target is average of all other channels
                trg = np.mean(data[np.arange(n_channels) != i], axis=0)
            else:
                # only one channel: create trg as the timelagged version of src
                trg = np.roll(src, tau)
                # TODO
                raise ValueError("Only one channel found, cannot compute PhiID with a single channel (TODO).")

            # Run the PhiID calculation
            atoms_res, _ = calc_PhiID(src, trg, tau, kind=kind, redundancy=redundancy)
            # add 'str', 'stx', 'sty', 'sts' together

            # Each atoms_res[name] is a vector length n_time - tau
            # We average over time to get a single scalar per atom
            for j, name in enumerate(atom_names):
                PhiID_vals[i, j] = float(np.mean(atoms_res[name]))
            for j, name in enumerate(INFORMATION_DYNAMICS_METRICS):
                # Get the indices of the atoms in the INFORMATION_DYNAMICS_METRICS dict
                atom_indices = [atom_names.index(atom) for atom in INFORMATION_DYNAMICS_METRICS[name]]
                # Sum the values of the atoms and average over time
                inf_dyn_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
            for j, name in enumerate(IIT_METRICS):
                # Get the indices of the atoms in the IIT_METRICS dict
                atom_indices = [atom_names.index(atom) for atom in IIT_METRICS[name]]
                # Sum the values of the atoms and average over time
                IIT_vals[i, j] = float(np.mean(np.sum(PhiID_vals[i, atom_indices], axis=0)))
                if name == "Integrated information":
                    # Subtract rtr
                    IIT_vals[i, j] -= float(np.mean(atoms_res["rtr"]))

        # Build metadata for output
        # Copy original metadata but replace channel dims
        out_meta = {}
        # Overwrite channels info
        if channel_labels is None:
            channel_labels = [f"ch{i}" for i in range(n_channels)]
        out_meta["channels"] = {"dim0": channel_labels, "dim1": atom_names}
        out_phi = {}
        out_phi["channels"] = {"dim0": channel_labels, "dim1": list(INFORMATION_DYNAMICS_METRICS.keys())}
        out_IIT = {}
        out_IIT["channels"] = {"dim0": channel_labels, "dim1": list(IIT_METRICS.keys())}

        return {"PhiID": (PhiID_vals, out_meta), "inf_dyn": (inf_dyn_vals, out_phi), "IIT": (IIT_vals, out_IIT)}


"""
filepath = "/home/yorguin/scratch/data/MEG_ketamine/derivatives/prepDur30Ov20/sub-S041213N1/ses-ketamine/meg/sub-S041213N1_ses-ketamine_task-resting_desc-None_split-01_epo.fif"
meg = read_epochs(filepath, preload=True)
meg = meg.resample(100, npad="auto")  # Resample to 100 Hz

this_epoch = meg.get_data()[0, :, :]  # Get the first epoch data (shape: channels x timepoints)
phi_epoch = process(this_epoch, tau=5, redundancy="MMI", kind="gaussian", channel_labels=meg.ch_names)

meg._data = meg._data[0:1, :, :]  # Keep only the first epoch

phi_epoch2 = single_atoms(meg, tau=5, redundancy="MMI", kind="gaussian", )

phi_epoch3 = atoms_results(phi_epoch2, key='InformationDynamics')

phi_epoch4 = atoms_results(phi_epoch2, key='IntegratedInformationDecomposition')
phi_epoch5 = atoms_results(phi_epoch2, key='IntegratedInformationTheory')

from pprint import pprint
# compare results
orig_infdynam = phi_epoch['inf_dyn'][0]
my_infdynam = phi_epoch3['values'][0,:,:]

pprint(orig_infdynam == my_infdynam)

orig_iit = phi_epoch['IIT'][0]
my_iit = phi_epoch5['values'][0,:,:]

pprint(orig_iit == my_iit)

orig_phi = phi_epoch['PhiID'][0]
my_phi = phi_epoch4['values'][0,:,:]
pprint(orig_phi == my_phi)
"""


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
    spaces = epochs.info['ch_names']
    
    for epoch_idx in range(n_epochs):
        for space_idx in range(n_spaces):
            
            time_series = data[epoch_idx, space_idx, :]
            start = time.time()
            try:
                envelope = np.abs(scipy.signal.hilbert(sp_stats.zscore(time_series)))
                dfa, _ = nk2.fractal_dfa(envelope)
            except Exception as e:
                print(f"Error processing epoch {epoch_idx}, space {space_idx}: {e}, using NaN")
                dfa = np.nan
            end = time.time()
            print(f"Epoch {epoch_idx}, Space {spaces[space_idx]}, DFA: {dfa:.4f}, Time taken: {end - start:.4f} seconds")
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


import numpy as np
import copy
import scipy.stats as sp_stats
from biotuner.biotuner_object import compute_biotuner


FREQ_BANDS = [
    [1, 3],      # delta
    [3, 7],   # theta
    [7, 12],   # alpha
    [12, 20],  # beta
    [20, 30],  # high beta
    [30, 70],  # gamma
]

def get_metrics(biotuning):
    # Keep only selected metrics and flatten subharm_tension
    d = biotuning.peaks_metrics
    return {
        'cons': d.get('cons', np.nan),
        'tenney': d.get('tenney', np.nan),
        'harmsim': d.get('harmsim', np.nan),
        'subharm_tension': float(d['subharm_tension'][0]) if (isinstance(d.get('subharm_tension'), list) and len(d['subharm_tension']) > 0) else np.nan
    }

results = {}

# --- EMD peaks ---
# biotuning_emd = compute_biotuner(
#     sf=sf,
#     peaks_function="EMD",
#     precision=0.5,
#     n_harm=10
# )
# biotuning_emd.peaks_extraction(
#     data,
#     FREQ_BANDS=FREQ_BANDS,
#     peaks_function="EMD",
#     max_freq=100,
#     n_peaks=5
# )
# biotuning_emd.compute_peaks_metrics(delta_lim=50)
# results['EMD'] = get_metrics(biotuning_emd)

# --- Fixed peaks ---
# biotuning_fixed = compute_biotuner(
#     sf=sf,
#     peaks_function="fixed",
#     precision=0.5,
#     n_harm=10
# )
# biotuning_fixed.peaks_extraction(
#     data,
#     FREQ_BANDS=FREQ_BANDS,
#     peaks_function="fixed",
#     max_freq=100,
#     n_peaks=5
# )
# biotuning_fixed.compute_peaks_metrics(delta_lim=50)
# results['fixed'] = get_metrics(biotuning_fixed)





def biotuning_feature(epochs, biotuner_params={}, peak_extraction_params={}, compute_peaks_metrics_params={}, desired_metrics=None):

    if desired_metrics is None:
        desired_metrics = ['cons', 'tenney', 'harmsim', 'subharm_tension']

    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    n_epochs, n_spaces, n_times = data.shape
    biotuner_values = np.empty((n_epochs, n_spaces, len(desired_metrics)))
    timings = np.nan * np.empty((n_epochs, n_spaces))
    
    for epoch_idx in range(n_epochs):
        for space_idx in range(n_spaces):
            time_series = data[epoch_idx, space_idx, :]
            try:
                start = time.time()
                biotuning = compute_biotuner(
                    sf=sf,
                    **biotuner_params
                )
                biotuning.peaks_extraction(
                    time_series,
                    **peak_extraction_params
                )
                biotuning.compute_peaks_metrics(
                    **compute_peaks_metrics_params
                )
                
                metrics = biotuning.peaks_metrics

                metric_values = []
                for metric in desired_metrics:
                    if metric == 'subharm_tension':
                        val = metrics.get('subharm_tension')
                        if isinstance(val, list) and len(val) > 0:
                            metric_values.append(float(val[0]))
                        else:
                            metric_values.append(np.nan)
                    else:
                        metric_values.append(metrics.get(metric, np.nan))

                biotuner_values[epoch_idx, space_idx, :] = metric_values
                end = time.time()
                timings[epoch_idx, space_idx] = end - start
                print(f"Epoch {epoch_idx}, Space {space_idx}, Metrics: {metric_values}, Time taken: {end - start:.4f} seconds")
                if 'subharm_tension' in desired_metrics:
                    if isinstance(metrics.get('subharm_tension'), list) and len(metrics['subharm_tension']) > 0:
                        biotuner_values[epoch_idx, space_idx, desired_metrics.index('subharm_tension')] = float(metrics['subharm_tension'][0])
                    else:
                        biotuner_values[epoch_idx, space_idx, desired_metrics.index('subharm_tension')] = np.nan
            except Exception as e:
                raise RuntimeError(f"Error processing epoch {epoch_idx}, space {space_idx}: {e}")

    
    epochs_labels = [x for x in range(n_epochs)]
    
    output = {}
    output['metadata'] = {'type': 'Biotuner'}
    output['metadata']['axes'] = {
        'epochs': epochs_labels,
        'spaces': space_names,
        'metrics': desired_metrics
    }
    output['metadata']['order'] = ('epochs', 'spaces', 'metrics')
    output['metadata']['timings'] = timings
    output['metadata']['times'] = times

    output['values'] = biotuner_values
    
    return output


# --- EMD peaks partials ---
biotuner_params_emd = {
    'peaks_function': 'EMD',
    'precision': 0.5,
    'n_harm': 10
}

peak_extraction_params_emd = {
    'FREQ_BANDS': FREQ_BANDS,
    'peaks_function': 'EMD',
    'max_freq': 100,
    'n_peaks': 5
}

compute_peaks_metrics_params_emd = {
    'delta_lim': 50
}

# --- Fixed peaks partials ---
biotuner_params_fixed = {
    'peaks_function': 'fixed',
    'precision': 0.5,
    'n_harm': 10
}

peak_extraction_params_fixed = {
    'FREQ_BANDS': FREQ_BANDS,
    'peaks_function': 'fixed',
    'max_freq': 100,
    'n_peaks': 5
}

compute_peaks_metrics_params_fixed = {
    'delta_lim': 50
}

FREQ_BANDS = [
    [1, 3],    # delta
    [3, 7],    # theta
    [7, 12],   # alpha
    [12, 20],  # beta
    [20, 30],  # high beta
    [30, 70],  # gamma
]
from functools import partial


biotuning_feature_EMD = partial(
    biotuning_feature,
    biotuner_params=biotuner_params_emd,
    peak_extraction_params=peak_extraction_params_emd,
    compute_peaks_metrics_params=compute_peaks_metrics_params_emd
)

biotuning_feature_FIXED = partial(
    biotuning_feature,
    biotuner_params=biotuner_params_fixed,
    peak_extraction_params=peak_extraction_params_fixed,
    compute_peaks_metrics_params=compute_peaks_metrics_params_fixed
)

#biotuning_emd = biotuning_feature_EMD(meg, desired_metrics=['cons', 'tenney', 'harmsim', 'subharm_tension'])



# filepath = r"C:\Users\yjman\Desktop\sub-S1TW_ses-lsd_task-Closed1_desc-None_epo.fif"

# import mne
# meg = mne.read_epochs(filepath,preload=True)

# data = meg.get_data()[0, 100, :]
# sf = meg.info['sfreq']
# # biotuning_fixed = biotuning_feature_FIXED(meg, desired_metrics=['cons', 'tenney', 'harmsim', 'subharm_tension'])
# # biotuning_emd = biotuning_feature_EMD(meg, desired_metrics=['cons', 'tenney', 'harmsim', 'subharm_tension'])

# #

# #spectrum['values'].shape
# #dfa = dfa_feature(meg)


# spectrum = spectrum_multitaper(meg,multitaper={})
# spectrum['metadata']

# [print(label,fun) for label,fun in zip(labels,funs)]



# # DEBUG=True
# # detrended_fluctuation_ = compute_detrendedFluctuation(meg, suffix='DetrendedFluctuation', internal_kwargs={"detrended_fluctuation":{}}, extra_metadata={}, prefoo=lambda x: x)


# # lziv_complexity,sample_entropy,spectral_entropy,app_entropy,hjorth_params,num_zerocross,perm_entropy,svd_entropy,higuchi_fd,katz_fd,petrosian_fd

# # compute_lziv_complexity = compute_lzivComplexity(meg, suffix='LZIVComplexity', internal_kwargs={"lziv_complexity":{}}, extra_metadata={}, prefoo=lambda x: x)


import numpy as np
import pandas as pd
import mne
from scipy import signal, stats
# === Helper Functions for Chaos ===
def _minmaxsig(x):
    maxs = signal.argrelextrema(x, np.greater)[0]
    mins = signal.argrelextrema(x, np.less)[0]
    idx = np.sort(np.concatenate([mins, maxs]))
    return x[idx]
def z1_chaos_test(x, sigma=0.5, rand_seed=0):
    np.random.seed(rand_seed)
    N = len(x)
    j = np.arange(1, N+1)
    t = np.arange(1, int(round(N / 10)) + 1)
    c = np.pi / 5 + np.random.rand(1000) * (3 * np.pi / 5)
    k_corr = np.zeros(1000)
    for i in range(1000):
        p = np.cumsum(x * np.cos(j * c[i]))
        q = np.cumsum(x * np.sin(j * c[i]))
        M = np.array([
            np.mean((p[n:N] - p[:N-n])**2 + (q[n:N] - q[:N-n])**2)
            - np.mean(x)**2 * (1 - np.cos(n * c[i])) / (1 - np.cos(c[i]))
            + sigma * (np.random.rand() - 0.5)
            for n in t
        ])
        k_corr[i] = stats.pearsonr(t, M)[0]
    return np.median(k_corr)
def chaos_pipeline(data, sigma=0.5, downsample=True):
    if downsample:
        data = _minmaxsig(data)
    if len(data) < 20:
        return np.nan
    data = data * (0.5 / np.std(data))
    return z1_chaos_test(data, sigma=sigma)
# === Final Feature Function (same structure as dfa_feature) ===
def chaos_feature(epochs, sigma=0.5, downsample=True):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, _ = data.shape
    k_values = np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            ts_filt = mne.filter.filter_data(ts, sfreq=sf, l_freq=0.5, h_freq=5, verbose=False)
            K = chaos_pipeline(ts_filt, sigma=sigma, downsample=downsample)
            k_values[epoch_idx, ch_idx] = K
    output = {
        'metadata': {
            'type': 'ChaosFeature',
            'axes': {
                'epochs': list(range(n_epochs)),
                'spaces': space_names
            },
            'order': ('epochs', 'spaces'),
            'times': times
        },
        'values': k_values
    }
    return output

def rate_entropy_feature(epochs, kmax=10):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    data = epochs.get_data()
    n_epochs, n_channels, _ = data.shape
    values = np.empty((n_epochs, n_channels))
    timings = np.nan * np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            start = time.time()
            feature_val, _ = nk2.entropy_rate(ts, kmax=kmax, symbolize='mean')
            end = time.time()
            if DEBUG:
                print(f"Epoch {epoch_idx}, Channel {space_names[ch_idx]}, Rate Entropy: {feature_val:.4f}, Time taken: {end - start:.4f} seconds")
            values[epoch_idx, ch_idx] = feature_val
            timings[epoch_idx, ch_idx] = end - start
    total_time = np.nansum(timings)

    return {
        'metadata': {
            'type': 'RateEntropy',
            'axes': {'epochs': list(range(n_epochs)), 'spaces': space_names},
            'order': ('epochs', 'spaces'),
            'times': times,
            'timings': timings,
            'total_time': total_time
        },
        'values': values
    }
def fisher_information_feature(epochs, delay=1, dimension=3):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    data = epochs.get_data()
    n_epochs, n_channels, _ = data.shape
    values = np.empty((n_epochs, n_channels))
    timings = np.nan * np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            start = time.time()
            feature_val, _ = nk2.fisher_information(ts, delay=delay, dimension=dimension)
            end = time.time()
            if DEBUG:
                print(f"Epoch {epoch_idx}, Channel {space_names[ch_idx]}, Fisher Information: {feature_val:.4f}, Time taken: {end - start:.4f} seconds")
            values[epoch_idx, ch_idx] = feature_val
            timings[epoch_idx, ch_idx] = end - start
    total_time = np.nansum(timings)
    return {
        'metadata': {
            'type': 'FisherInformation',
            'axes': {'epochs': list(range(n_epochs)), 'spaces': space_names},
            'order': ('epochs', 'spaces'),
            'times': times
        },
        'values': values
    }
def correlation_dimension_feature(epochs, delay=1, dimension=3, radius=64):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    data = epochs.get_data()
    n_epochs, n_channels, _ = data.shape
    values = np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            feature_val, _ = nk2.fractal_correlation(ts, delay=delay, dimension=dimension, radius=radius, show=False)
            values[epoch_idx, ch_idx] = feature_val
    return {
        'metadata': {
            'type': 'CorrelationDimension',
            'axes': {'epochs': list(range(n_epochs)), 'spaces': space_names},
            'order': ('epochs', 'spaces'),
            'times': times
        },
        'values': values
    }
def lyapunov_exponent_feature(epochs, delay=1, dimension=3, len_trajectory_ratio=0.05):
    epochs = epochs.copy()
    sf = epochs.info['sfreq']
    times = epochs.times
    space_names = epochs.info['ch_names']
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    len_trajectory = int(n_times * len_trajectory_ratio)
    values = np.empty((n_epochs, n_channels))
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            ts = data[epoch_idx, ch_idx, :]
            feature_val, _ = nk2.complexity_lyapunov(
                ts, delay=delay, dimension=dimension,
                method='rosenstein1993', separation='auto',
                len_trajectory=len_trajectory
            )
            values[epoch_idx, ch_idx] = feature_val
    return {
        'metadata': {
            'type': 'LyapunovExponent',
            'axes': {'epochs': list(range(n_epochs)), 'spaces': space_names},
            'order': ('epochs', 'spaces'),
            'times': times
        },
        'values': values
    }


import numpy as np
from scipy.signal import find_peaks
from biotuner.metrics import integral_tenneyHeight, ratios2harmsim, compute_subharmonic_tension


def feature_harmonicity(input_dict, height=None, distance=None, bands=None):
    breakpoint()
    psds = input_dict['values']
    freqs = input_dict['metadata']['axes']['frequencies']
    space_names = input_dict['metadata']['axes']['spaces']
    if bands is None:
        bands = [
            ('delta', (2, 4)),
            ('theta', (4, 8)),
            ('alpha', (8, 12)),
            ('beta', (12, 30)),
            ('gamma', (30, 60)),
        ]
    band_names = [name for name, _ in bands]

    n_epochs, n_channels, n_bins = psds.shape
    nbands = len(bands)
    n_features = 3  # Tenney, HarmSim, Subharmonic Tension
    metrics_list = ['tenney', 'harmsim', 'subharm_tension']

    # Initialize arrays
    max_peaks = np.full((n_epochs, n_channels, nbands), np.nan)
    metrics = np.full((n_epochs, n_channels, n_features), np.nan)

    for ep in range(n_epochs):
        for ch in range(n_channels):

            print(f"Processing epoch {ep+1}/{n_epochs}, channel {ch+1}/{n_channels}")
            peaks_list = []
            for band_idx, (band_name, (low, high)) in enumerate(bands):
                mask = (freqs >= low) & (freqs <= high)
                band_psd = psds[ep, ch, mask]
                band_freqs = freqs[mask]
                if len(band_psd) > 0:
                    peaks, _ = find_peaks(band_psd, height=height, distance=distance)
                    if len(peaks) > 0:
                        max_peak_idx = peaks[np.argmax(band_psd[peaks])]
                        max_peaks[ep, ch, band_idx] = band_freqs[max_peak_idx]
                        peaks_list.append(band_freqs[max_peak_idx])
                    else:
                        peaks_list.append(np.nan)
                else:
                    peaks_list.append(np.nan)
            # Filter NaNs
            valid_peaks = [p for p in peaks_list if not np.isnan(p)]
            valid_peaks = list(np.round(valid_peaks, 1))
            print(f"Valid peaks for epoch {ep+1}, channel {ch+1}: {valid_peaks}")
            if len(valid_peaks) > 0:
                try:
                    _, _, subharm, _ = compute_subharmonic_tension(valid_peaks, n_harmonics=3, delta_lim=50)
                    tenney = integral_tenneyHeight(valid_peaks)
                    harmsim = np.mean(ratios2harmsim(valid_peaks))
                    metrics[ep, ch, 0] = tenney
                    metrics[ep, ch, 1] = harmsim
                    metrics[ep, ch, 2] = subharm[0]
                except Exception:
                    print(f"Error computing metrics for epoch {ep+1}, channel {ch+1}: {valid_peaks}")
                    pass

    # Wrap in template
    out_dict = {
        'metadata': {
            'type': 'MaxBandPeaksAndMetrics',
            'axes': {
                'epochs': list(range(n_epochs)),
                'spaces': space_names if space_names is not None else list(range(n_channels)),
                'bands': band_names,
                'metrics': metrics_list
            },
            'order': ('epochs', 'spaces')
        },
        'values': {
            'max_peaks': max_peaks,
            'metrics': metrics
        }
    }
    return out_dict

def process_harmonicity_output(harmo_dict,args=None):
    #fix order
    #breakpoint()
    harmo_dict['metadata']['order'] = ('epochs', 'spaces','metrics')

    if isinstance(harmo_dict['values'], dict):
        harmo_dict = copy.deepcopy(harmo_dict)
        harmo_dict['metadata']['peaks'] = harmo_dict['values']['max_peaks']
        harmo_dict['values'] = harmo_dict['values']['metrics']
        return harmo_dict
    elif isinstance(harmo_dict['values'], np.ndarray):
        # If values is a numpy array, assume it's already in the right format
        return copy.deepcopy(harmo_dict)
    else:
        raise ValueError("Unexpected format for 'values' in harmonicity output. Expected dict or ndarray.")