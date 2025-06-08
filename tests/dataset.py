from mne.datasets import eegbci
for sub in [1, 2, 3]:
    eegbci.load_data(sub, [6, 10, 14], None)