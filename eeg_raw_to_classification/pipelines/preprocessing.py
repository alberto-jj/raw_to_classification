# Import dependencies
import mne
import os
import sys
import autoreject
import numpy as np
import bids
import scipy
from mne.datasets.eegbci import standardize
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pyprep.prep_pipeline import PrepPipeline
import logging
import matplotlib
from eeg_raw_to_classification.utils import load_meeg
import argparse
import os
import traceback
from joblib import delayed, Parallel
import psutil
import bids
from eeg_raw_to_classification.utils import load_yaml
from eeg_raw_to_classification.utils import get_derivative_path,get_path
import time
import pathlib
import importlib

matplotlib.use('Agg') # saves ram https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close
import matplotlib.pyplot as plt

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


def prepare(filename,output_path=None, dataset_cfg=None, njobs=1, downsample = None, filter_args=None, epoch_config={}, notch_filter=None):
    """
    njobs: is ignored, only used to keep the same signature as the original function
    """
    info = {}
    figures = []

    info['filename'] = filename
    info['dataset_cfg'] = dataset_cfg
    info['downsample'] = downsample
    info['filter_args'] = filter_args
    info['njobs'] = njobs
    info['epoch_config'] = epoch_config

    eegpath = filename

    raw = mne.io.read_raw(eegpath,verbose=False,preload=True)

    # Filter the data
    if notch_filter is not None:
        raw = raw.notch_filter(**notch_filter, verbose=False)
        print('FILTERED NOTCH with', notch_filter)

    if filter_args is not None:
        raw = raw.filter(**filter_args, verbose=False)
        print('FILTERED with', filter_args)

    # Extract epochs
    print('EPOCH SEGMENTATION')
    if isinstance(epoch_config, dict):
        epochs = mne.make_fixed_length_epochs(raw,preload=True,**epoch_config)
    elif isinstance(epoch_config, str):
        if epoch_config == 'SingleEpoch':
            epochs = mne.make_fixed_length_epochs(raw, preload=True, duration=raw.times[-1], overlap=0)
        else:
            raise ValueError(f"Unknown epoch_config: {epoch_config}")

    if downsample:
        print(f'DOWNSAMPLING TO {downsample}Hz')
        epochs = epochs.resample(downsample, verbose=False)


    if output_path is not None:
        epochs.save(output_path, split_naming='bids', overwrite=True)


    return epochs

def another_prepare(filename, output_path,dataset_cfg=None, njobs=1, standardize_names=True,epoch_length = 2,
              downsample = 500, normalization = False, ica_method='infomax',skip_prep=False,skip_reject=False,):
    """
    Run PREPARE pipeline for resting-state EEG signal preprocessing.
    Returns the preprocessed mne object in BIDS derivatives path. 
    
    Parameters
    ----------
    filename : str
        Full path of raw file and extension.
    dataset_cfg : dict, which contains the dataset configuration:
        line_noise : float
            The line noise frequency (in Hz) to be removed using PyPREP or notch.
            if skip_prep is True, this will be used for notch filtering if not None
            if skip_prep is False, this will be used for line noise removal in PyPREP unless None
        ch_names : list
            Channel names to keep. Can be defined in dataset['ch_names'].
    standardize_names : bool
        Whether to standardize channel names to MNE standard names.
    epoch_length : float
        The epoch length in seconds.
    downsample : float
        Sampling frequency (in Hz) for downsamlping.
    bandwidth : float
        The bandwidth of the multi taper windowing function in Hz.
    normalization : bool 
        Whether to normalize the data or not (currently z transform).
    ica_method : str
        ica method param as per MNE ICA class. Only used if skip_reject is False.
    """
    #log_file = os.path.join(bids_path,'code','sovabids','sovabids.log')
    #setup_logging(log_file)

    # Import EEG raw recording + channel standarization
    raw = load_meeg(filename)
    # Remove channels which are not needed

    keep_chans = dataset_cfg.get('ch_names', None)
    line_noise = dataset_cfg.get('line_noise', None)

    if keep_chans is not None:
        raw.reorder_channels(keep_chans)
    
    if standardize_names:
        standardize(raw) #standardize ch_names

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
            }
        if line_noise is not None:
            prep_params["line_freqs"] = np.arange(line_noise, sample_rate / 2, line_noise)

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

        # Apply average reference?
        raw.set_montage(montage)
        raw.set_eeg_reference('average',projection=False)
        prep_info={'status':'skipped'}

        # Notch filter
        if line_noise is not None:
            print('NOTCH FILTER')
            raw.notch_filter(line_noise, picks=eeg_index, method='spectrum_fit', verbose=True)

    if normalization:
        # It is debatable where to normalize the data. Here we do it after PyPREP.

        # Sanity check, the argmin of the zscored data should be the same as the argmin of the raw data
        assert np.argmin(raw.get_data()[0,:])==np.argmin(scipy.stats.zscore(raw.get_data(),axis=1)[0,:])
        raw._data = scipy.stats.zscore(raw.get_data(),axis=1)
        print('AMPLITUDE NORMALIZATION DONE')

    # Filter the data
    raw = raw.filter(l_freq=1, h_freq=None) # bandpassing 1 Hz

    # Extract epochs
    print('EPOCH SEGMENTATION')
    epochs = mne.make_fixed_length_epochs(raw, duration = epoch_length, preload=True)
    epochs = epochs.resample(downsample)

    if not skip_reject:
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
    else :
        info = {}
        figures = []
        epochs_ar = epochs.copy()


    try:
        meeg = epochs_ar
        report = mne.Report(title=f'Preprocessing report', verbose='error')

        if isinstance(meeg, mne.io.BaseRaw):
            report.add_raw(meeg, title=f'Raw data')
        elif isinstance(meeg, mne.BaseEpochs):
            report.add_epochs(meeg, title=f'Epochs')

        ## add spectrum to report
        report.add_figure(meeg.plot_psd(show=False), title=f'Spectrum')
        fmax = meeg.info['sfreq'] / 2
        if fmax > 200:
            fmax = 200
        report.add_figure(meeg.plot_psd(show=False, fmax=fmax), title=f'Spectrum (fmax={fmax})')
    except Exception as e:
        print(f"Error creating report: {e}")
        report = None

    if output_path is not None:
        meeg.save(output_path, split_naming='bids', overwrite=True)

    if report is not None and output_path is not None:
        report.save_as_html(output_path.replace('.fif', '_prepareReport.html'), overwrite=True)


    # figs_path = preprocessed_path.replace('_epo.fif', '_prepareFigs.html') # careful with this, its prone to bugs if the name is not correctly replace
    # info_path = preprocessed_path.replace('_epo.fif', '_prepareInfo.txt')
    # save_figs_in_html(figs_path, figures)
    # save_dict_to_json(info_path, info)

    return meeg

def foo(meeg_file, this_prep, DATASET, preprocessed_path, DEBUG, internal_njobs=1, retry_errors=False):
    # imports here to avoid problems with joblib
    import os


    njobs = internal_njobs #internal jobs #len(psutil.Process().cpu_affinity())
    print('Internal NJOBS:', njobs)
    print(meeg_file)

    fifname = os.path.basename(preprocessed_path)
    fifpath = os.path.dirname(preprocessed_path)

    fname = preprocessed_path
    suffix = fifname.split('_')[-1]
    fifname2 = fifname.split('_')[:-1]+['split-01', suffix]
    fifname2 = '_'.join(fifname2)
    preprocessed_path2 = os.path.join(fifpath, fifname2)

    if os.path.isfile(preprocessed_path2):
        print(f'Found split file: {preprocessed_path2}, using it instead of {preprocessed_path}')
        preprocessed_path = preprocessed_path2


    if not (os.path.isfile(preprocessed_path)) or this_prep['overwrite']:
        if os.path.isfile(preprocessed_path.replace('.fif', '_problem.txt')) and not retry_errors:
            print(f'Error file exists: {preprocessed_path.replace(".fif", "_problem.txt")}, skipping')
            return
        import matplotlib.pyplot as plt

        os.makedirs(fifpath, exist_ok=True)
        try:
            from eeg_raw_to_classification.utils import save_dict_to_json

            if 'redefine_prepare' in this_prep:
                print('Redefining prepare function from another module')
                import_dict = this_prep['redefine_prepare']
                module_name = import_dict['from_this']
                func_name = import_dict['import_that']
                module = importlib.import_module(module_name)
                prepare = getattr(module, func_name)
            else:
                pass
                # use the prepare defined in this module


            processed_meeg = prepare(filename=meeg_file,output_path=preprocessed_path, dataset_cfg=DATASET, njobs=njobs, **this_prep['prepare'])
            plt.close('all')

        except Exception:
            print(traceback.format_exc())
            if DEBUG:
                raise
            else:
                save_dict_to_json(preprocessed_path.replace('.fif', '_problem.txt'), {'file': meeg_file, 'problem': traceback.format_exc()})
    else:
        print(f'Already Exists: {preprocessed_path} or overwrite is False')

def pipeline_preprocess(pipeline_yml,max_files=None, external_njobs=1, internal_njobs=1, retry_errors=False, raise_on_error=False, index=None, only_total=False):
    """ Preprocess EEG data according to the pipeline configuration.
    Parameters
    ----------
    pipeline_yml : str
        Path to the pipeline YAML file.
    """

    cfg = load_yaml(pipeline_yml)
    MOUNT = cfg.get('mount', None)
    datasets = load_yaml(get_path(cfg['datasets_file'], MOUNT))
    #datasets = load_yaml(cfg['datasets_file'])

    PROJECT = cfg['project']
    MAX_FILES = max_files
    external_njobs = external_njobs
    DEBUG = raise_on_error
    PARALLELIZE = external_njobs > 1
    internal_njobs = internal_njobs
    only_total = only_total
    single_index = index



    if (only_total or single_index) and external_njobs > 1:
        raise ValueError('Cannot get total number of files or process single file with external_jobs > 1')
    ALL_MEEGS = []
    for preplabel in cfg['3_preprocess']['prep_list']:
        overall_index = 0
        # you may try to do this loop outisde (with inner eeg loop) as in 4_features.py,
        # but notice that foo depends on some loop-state variables,
        # so its a bit more complicated, and perhaps not that worth it
        for dslabel, DATASET in datasets.items(): 
            if DATASET.get('skip', False):
                continue

            this_prep = cfg['3_preprocess']['prep_cfg'][preplabel]

            print(f'PREPROCESSING {dslabel} with {preplabel} pipeline')
            file_filter = DATASET.get('bids_layout', None)

            start_time = time.time()
            bids_root = DATASET.get('bids_root', None)
            bids_root = get_path(bids_root, MOUNT)
            layout = bids.BIDSLayout(bids_root,validate=False)
            # how to make this faster, it takes too long...
            meegs = layout.get(**file_filter)
            end_time = time.time()
            print(f'Time taken to get EEG files from layout: {end_time - start_time} seconds for dataset {dslabel}')

            if MAX_FILES:
                if MAX_FILES > len(meegs):
                    limit = len(meegs)
                else:
                    limit = MAX_FILES
                meegs = meegs[:limit]
            meegs = [pathlib.Path(x).as_posix() for x in meegs]
            meegs = [x for x in meegs if ('split-01' in x or not 'split-' in x)] # 01 will work if at most 99 split files?

            print(len(meegs), meegs)
            

            derivatives_root = DATASET.get('derivatives_root', None)
            
            if derivatives_root is not None:
                derivatives_root = get_path(derivatives_root, MOUNT)
                derivatives_root = os.path.join(derivatives_root, f'{preplabel}/')
            else:
                derivatives_root = os.path.join(layout.root, f'derivatives/{preplabel}/')
            
            get_derivative = lambda x: pathlib.Path(get_derivative_path(layout, x, 'None', 'epo', '.fif', bids_root, derivatives_root)).as_posix()

            if PARALLELIZE:
                Parallel(n_jobs=external_njobs)(delayed(foo)(x, this_prep, DATASET, get_derivative(x), DEBUG,internal_njobs, args.retry_errors ) for x in meegs)
            else:
                for meeg_file in meegs:
                    ALL_MEEGS.append(meeg_file)

                    if only_total:
                        overall_index+=1
                        continue
                    if single_index is not None and overall_index != single_index:
                        overall_index+=1
                        continue
                    foo(meeg_file, this_prep, DATASET, get_derivative(meeg_file), DEBUG, internal_njobs, retry_errors)
                    overall_index+=1
        if only_total:
            print(f'Total number of files: {len(ALL_MEEGS)}')
            for count,eeg in enumerate(ALL_MEEGS):
                print(count,eeg)
            print(f'Total number of files: {len(ALL_MEEGS)}')
            return len(ALL_MEEGS)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess MEEG data.')
    parser.add_argument('pipeline_yml', type=str, help='Path to the pipeline YAML file.')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to process.')
    parser.add_argument('--external_jobs', type=int, default=1, help='Number of external jobs.')
    parser.add_argument('--internal_jobs', type=int, default=1, help='Number of internal jobs.')
    parser.add_argument('--retry_errors', action='store_true', help='Retry files that had errors.')
    parser.add_argument('--raise_on_error', action='store_true', help='Enable raise_on_error mode.')
    parser.add_argument('--index', type=int, default=None, help='Index of the file to process. Total index taking into account the dataset outer loop. Only works with external_jobs=1.')
    parser.add_argument('--only_total', action='store_true', help='Just get the total number of files. Only works with external_jobs=1.')
    args = parser.parse_args()

    pipeline_preprocess(args.pipeline_yml, max_files=args.max_files, external_njobs=args.external_jobs,
                        internal_njobs=args.internal_jobs, retry_errors=args.retry_errors,
                        raise_on_error=args.raise_on_error, index=args.index, only_total=args.only_total)
