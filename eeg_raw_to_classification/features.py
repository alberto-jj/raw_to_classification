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
from antropy import detrended_fluctuation,lziv_complexity,sample_entropy,spectral_entropy,app_entropy,hjorth_params,num_zerocross,perm_entropy,svd_entropy,higuchi_fd,katz_fd,petrosian_fd
from neurokit2 import entropy_multiscale
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
                    if isinstance(fun,str):
                        fun=eval(fun)
                    output = fun(input_data,**inner_featdict['args'])
                    os.makedirs(os.path.dirname(outputfile),exist_ok=True)
                    np.save(outputfile,output)
                else:
                    print(f'Already Exists:{outputfile}')
                    output = np.load(outputfile,allow_pickle=True).item()
            else:
                innerfun=eval(f"inner_featdict['function']")
                innerfun=eval(innerfun)
                output = innerfun(input_data,**inner_featdict['args'])
        input_data = output
    return output

def extract_item(x,fun,newtype):
    if isinstance(fun,str):
        fun=eval(fun.replace('eval%',''))
    x = copy.deepcopy(x)
    newfoo=np.vectorize(fun)
    x['values'] = newfoo(x['values'])
    if newtype:
        x['metadata']['type'] = newtype
    return x
def agg_numpy(x,numpyfun,axisname='epochs',max_numitem=None): # or give a more complex indexing for items
    if isinstance(numpyfun,str):
        numpyfun=eval(numpyfun.replace('eval%',''))
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
    output['metadata']['times']=epochs.times #TODO: times is not standarized across all features
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

def roi_aggregator(data,mapping,numpyfun=None,axisname='spaces',ignore=['none']):
    # maybe a similar strategy could also work when the epochs are inhomoegeneous (different tasks?)
    # numpy fun should be a function that takes data,axis and keepdims
    # we can view this as a new feature or as an aggregate
    # assume dict output format from other features

    # first map spaces to rois
    spaces =  data['metadata']['axes'][axisname]
    scope = {}
    exec(mapping, scope)
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
    for e in range(nepochs):
        result = [%fun%(prefoo(data[e,i,:]),**kwargs['%fun%']) for i in range(len(eeg.ch_names))]
        epochs.append([ {i:v for i,v in enumerate(x)} if isinstance(x,tuple) else x for x in result])


    values = np.array(epochs)

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
    output['values']= values
    output['metadata'].update(extra_metadata)
    return output
"""
antropy_definitions = [fun_template.replace('%label%',label).replace('%fun%',fun) for label,fun in zip(labels,funs)]
for foo in antropy_definitions:
    exec(foo)


if __name__ == '__main__':
    [print(label,fun) for label,fun in zip(labels,funs)]
    print(fun_template.replace('%label%',labels[-1]).replace('%fun%',funs[-1]))