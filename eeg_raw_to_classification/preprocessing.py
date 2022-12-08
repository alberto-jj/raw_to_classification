# Import dependencies
import mne
import os
import sys
import autoreject
import numpy as np
import bids
from mne.datasets.eegbci import standardize
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline
import logging

LOGGER = logging.getLogger(__name__)

def rejlog2dict(rejlog):
    d = {}
    d['bad_epochs_bool']=rejlog.bad_epochs.tolist()
    d['num_bad_epochs'] = sum(rejlog.bad_epochs.tolist()) #len(np.where(rejlog.bad_epochs == True)[0].tolist())
    d['total_epochs']=len(rejlog.bad_epochs.tolist())
    d['bad_epoch_ratio']=sum(rejlog.bad_epochs.tolist())/len(rejlog.bad_epochs.tolist())
    d['ch_names']=rejlog.ch_names
    d['labels']=rejlog.labels.tolist()
    return d

def prepare(filename, line_noise, keep_chans=None, epoch_length = 2,
              downsample = 500, normalization = False, ica_method='infomax',skip_prep=False,njobs=1):
    """
    Run PREPARE pipeline for resting-state EEG signal preprocessing.
    Returns the preprocessed mne object in BIDS derivatives path. 
    
    Parameters
    ----------
    filename : str
        Full path of raw file and extension.
    line_noise : float
        The line noise frequency (in Hz) to be removed using PyPREP.
    keep_chans : list
        Channel names to keep. Can be defined in dataset['ch_names'].
    epoch_length : float
        The epoch length in seconds.
    downsample : float
        Sampling frequency (in Hz) for downsamlping.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    normalization : bool NOT IMPLEMENTED
        Returns both non-normalized and normalized .fif MNE objects.
    ica_method : str
        ica method param as per MNE ICA class

    """
    #log_file = os.path.join(bids_path,'code','sovabids','sovabids.log')
    #setup_logging(log_file)

    # Import EEG raw recording + channel standarization
    raw = mne.io.read_raw(filename,preload=True)
    # Remove channels which are not needed
    standardize(raw) #standardize ch_names
    if keep_chans is not None:
        raw.pick_channels(keep_chans)
    eeg_index = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
    ch_names = raw.info["ch_names"]
    ch_names_eeg = list(np.asarray(ch_names)[eeg_index])

    # Add a montage to the data
    montage_kind = "standard_1005"
    montage = mne.channels.make_standard_montage(montage_kind)

    # Extract some info
    sample_rate = raw.info["sfreq"]
    
    # PyPREP
    # parameters
    if not skip_prep:
        print('PREP')
        prep_params = {
            "ref_chs": ch_names_eeg,
            "reref_chs": ch_names_eeg,
            "line_freqs": np.arange(line_noise, sample_rate / 2, line_noise)
            }
        prep = PrepPipeline(raw, prep_params, montage)
        prep.fit()
        raw = prep.raw.copy()
        
        prep_info ={'noisy_channels_original':prep.noisy_channels_original,
                'noisy_channels_before_interpolation':prep.noisy_channels_before_interpolation,
                'noisy_channels_after_interpolation':prep.noisy_channels_after_interpolation,
                'bad_before_interpolation':prep.bad_before_interpolation,
                'interpolated_channels':prep.interpolated_channels,
                'still_noisy_channels':prep.still_noisy_channels,
                }
        del prep
    else:
        raw = raw.copy()
        raw.set_montage(montage)
        prep_info={'status':'skipped'}
    # Filter the data
    raw.filter(l_freq=1, h_freq=None) # bandpassing 1 Hz

    # Extract epochs
    print('EPOCH SEGMENTATION')
    epochs = mne.make_fixed_length_epochs(raw, duration = epoch_length, preload=True)
    epochs.resample(downsample)

    # Automated epoch rejection
    print('PREICA AUTOREJECT')
    ar = autoreject.AutoReject(random_state=11,n_jobs=njobs, verbose=True)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    figures = [reject_log.plot(show=False)]
    info = {'prep':prep_info,'autoreject-preica': rejlog2dict(reject_log)}
    # 1Hz high pass already done before
    filt_epochs = epochs_ar.copy().filter(l_freq=None, h_freq=100.0) # bandpassing 100 Hz (as in the MATLAB implementation of ICLabel)
    
    if ica_method in ['infomax','picard']:
        fit_params = dict(extended=True)
    else:
        fit_params = None
    n_components = np.linalg.matrix_rank(raw.get_data())
    ica = ICA(
        n_components=n_components,
        max_iter="auto",
        method=ica_method,
        random_state=97,
        fit_params=fit_params)
    print('ICA')
    ica.fit(filt_epochs)
    figures=figures+ica.plot_properties(filt_epochs,picks=list(range(n_components)),show=False)
    # Annotate using mne-icalabel
    ic_labels = label_components(filt_epochs, ica, method="iclabel")
    labels = ic_labels["labels"]
    ica_filter = ["brain", "other"]
    ica_info = ic_labels
    ica_info['y_pred_proba']= ica_info['y_pred_proba'].tolist()
    ica_info.update({'included_filter':ica_filter})
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ica_filter] # a conservative approach suggested in mne-icalabel
    ica_info.update({'excluded_idx':exclude_idx})
    print(f"Excluding these ICA components: {exclude_idx}")
    #TODO: Save ica plots???
    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_epochs = epochs.copy() # Use non autoreject epochs
    ica.apply(reconst_epochs, exclude=exclude_idx)
    print('POSTICA AUTOREJECT')
    # Post ICA automated epoch rejection (suggested by Autoreject authors)
    ar = autoreject.AutoReject(random_state=11, n_jobs=njobs, verbose=True)
    ar.fit(reconst_epochs)
    epochs_ar, reject_log = ar.transform(reconst_epochs, return_log=True)
    figures+=[reject_log.plot(show=False)]
    info['icalabel']=ica_info
    info['autoreject-postica']=rejlog2dict(reject_log)
    # Normalization of recording-specific variability (optional)
    
    return epochs_ar,info,figures
