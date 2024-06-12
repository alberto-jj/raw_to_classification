import numpy as np
import pickle
import matplotlib.pyplot as plt
from antropy import detrended_fluctuation,lziv_complexity,sample_entropy,spectral_entropy,app_entropy,hjorth_params,num_zerocross,perm_entropy,svd_entropy,higuchi_fd,katz_fd,petrosian_fd
import scipy
p1=r"Y:\datasets\HenryRailo\bids\derivatives\features30@prep-defaultprep\sub-02\eeg\sub-02_task-eyesClosed_desc-reject_FooofFromAverageROI.npy"
p2=r"Y:\datasets\ds004504-download\derivatives\features-epochs\sub-024\eeg\sub-024_task-eyesclosed_desc-reject_FooofFromAverage.npy"
path = p1

data = np.load(path, allow_pickle=True)

data = data.item()

#plt.show()

#[0] offset, [1] knee, [-1] slope
#"eval%lambda x: x.aperiodic_params_[0]"
dir(data['values'][0].aperiodic_params_)

import mne

lempel_ziv_complexity(data['values'][0],normalize=True)

path=r"Y:\datasets\ds004796-download\sub-22\eeg\sub-22_task-rest_eeg.vhdr"
eeg=mne.io.read_raw(path,preload=True)
norm_data = scipy.stats.zscore(eeg.get_data(),axis=1)

katz_fd(eeg.get_data()[0,:])
katz_fd(norm_data[0,:])

perm_entropy(eeg.get_data()[0,:])
perm_entropy(norm_data[0,:])

perm_entropy(eeg.get_data()[0,:],normalize=True)

.plot()

|