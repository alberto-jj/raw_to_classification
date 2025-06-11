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
        for k,v in val.items():
            if isinstance(v,str) and 'eval%' in v:
                expression = v.replace('eval%','')
                kwargs[key][k] = eval(expression)

    fm = FOOOF(verbose=False,**kwargs['FOOOF'])
    fm.fit(freqs, psds,**kwargs['fit']) # correct, if we used add_data it would ignore for examle freq_range
    return fm

def fooof_from_average(data,internal_kwargs={'FOOOF':{},'fit':{}}):
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

    spaces =  spectra['metadata']['axes']['spaces']
    freqs =   spectra['metadata']['axes']['frequencies']
    # Only the mean
    axes= spectra['metadata']['order']

    output = {}

    values = np.empty(len(spaces),dtype=object)
    output['metadata'] = {'type':'fooofFromAverageSpectrum','kwargs':{'internal_kwargs':internal_kwargs},'freqs':freqs}
    output['metadata']['axes']={'spaces':spaces}
    output['metadata']['order']=('spaces')
    psd = spectra['values']
    for space in spaces:
        space_idx = spaces.index(space)
        thispsd = np.take(psd,indices=space_idx,axis=axes.index('spaces'))
        fm = single_fooof(freqs, thispsd,kwargs)
        values[space_idx]= fm
    output['values'] = values
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
        matrix = epochs.get_data()
        channel_labels = epochs.ch_names
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

    # Compute PhiID for each channel vs. the mean of all other channels
    for e in range(n_epochs):
        data = matrix[e, :, :]  # shape (n_channels, n_time)
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
            #assert [k for k in atoms_res.keys()] == atom_names
            atoms_res['rtr'].shape
            for key in atoms_res.keys():
                vals = atoms_res[key]
                atoms_vals[e, i, atom_names.index(key), :] = vals# should be size [:n_time - tau]




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
        'times': time_axis
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

if __name__ == '__main__':
    [print(label,fun) for label,fun in zip(labels,funs)]
    print(fun_template.replace('%label%',labels[-1]).replace('%fun%',funs[-1]))