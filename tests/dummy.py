from mne import read_epochs
path = '/home/yorguin/scratch/data/MEG_LSDV2/derivatives/prepDur30Ov20/sub-S15ST/ses-placebo/meg/sub-S15ST_ses-placebo_task-Closed1_desc-None_epo.fif'
read_epochs(path, preload=True, verbose='DEBUG')


from mne.utils import sizeof_fmt
from mne.io import read_info

try:
    info = read_info(path)
    print(info)
except Exception as e:
    print("Low-level read_info error:", e)
