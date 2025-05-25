import eeg_raw_to_classification.feature_extraction as fe
# List of available features
fe.FunctionalFeatureRegistry.list()
fe.ChainFeatureRegistry.list()

dir(fe)

import mne
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_raw.fif"
path = sample_data_raw_file.as_posix()
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False, preload=True).crop(tmax=60)

#fe.run_feature(raw, fe.BasicPrep, inspect_only=True)
raw2=fe.run_feature(raw, fe.BasicPrep, inspect_only=False,file_bidspath='sub-sample_eeg.fif')


events = mne.find_events(raw, stim_channel="STI 014")
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7,preload=True, verbose=False)



mne_kwargs={'sfreq':512}
resampled =fe.functional_resample_feature(epochs, mne_kwargs=mne_kwargs)

path = 'test-epochs.fif'
fe.functional_save(resampled, path,'mne')
resampled2=fe.functional_load(path,'mne')


resampled2.values
output_epochs = fe.functional_spectrum_feature(epochs, method='welch', mne_kwargs={'n_fft': 256, 'n_overlap': 128, 'average': 'mean'})
output_raw = fe.functional_spectrum_feature(raw, method='welch', mne_kwargs={'n_fft': 256, 'n_overlap': 128, 'average': 'mean'})

from copy import deepcopy
deepcopy(epochs.ch_names)

path = sample_data_raw_file.as_posix()

# When inspect_only is True it returns a dictionary with all the features in the chain with their computed status (True/False)
# If no path was provided it is False for all of them

output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage,inspect_only=True) # False for all of them
output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage, path,inspect_only=True) # True for the features that are already computed and saved


# When inspect_only is False it returns the output of the last feature in the chain
# If a path is provided it will skip the features that are already computed and saved, and return the output of the last feature in the chain (which it also saves)

#output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage, inspect_only=False)
#output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage, path,inspect_only=False)

# When inspect_only is True and no path is provided, it just returns a dictionary with all the features in the chain and False for all of them
#output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage, path,inspect_only=True)

# When inspect_only is False, and no path is provided it computes the features in the chain and returns the output of the last feature in the chain
#output = fe.run_feature(epochs, fe.SpectrumMultitaperAverage, None,inspect_only=False)


spectrum = fe.run_feature(epochs, fe.SpectrumMultitaper, path,inspect_only=False)

