# import pickle

# path=r"Y:\code\raw_to_classification\data\scalingAndFolding\FeaturesChannels@30@prep-defaultprep\noScaling_all\folds-noScaling@all.pkl"
# pickle_file = open(path, 'rb')
# data = pickle.load(pickle_file)
# pickle_file.close()

# data.keys()
# data[-1].keys()


# import numpy as np

# path=r"Y:\datasets\Iowa\Dataset\IowaDataset\bids\derivatives\features30@ROI@prep-defaultprep\sub-Control1021\ses-01\eeg\sub-Control1021_ses-01_task-rest_desc-reject_higuchiROI.npy"
# data=np.load(path,allow_pickle=True)

# data

# import mne

# path=r"Y:\datasets\Iowa\Dataset\IowaDataset\bids\derivatives\defaultprep\sub-Control1201\ses-01\eeg\sub-Control1201_ses-01_task-rest_desc-reject_epo.fif"
# eeg=mne.read_epochs(path,preload=True)

# path=r"C:\datasets\Data and Code\Dataset\IowaDataset\Raw data\Control1081.vhdr"
# path=r"Y:\datasets\ds004796-download\sub-03\eeg\sub-03_task-rest_eeg.vhdr"
# eeg=mne.io.read_raw(path,preload=True)

# eeg.annotations


# use alberto environment (mat73)
import glob
import os
import scipy.io as sio
import numpy as np
import mat73
import mne
import pandas as pd

path=r'Y:\datasets\epilepsy\rawdata'
path=path.replace('\\','/')

pattern = os.path.join(path,'**','*_iEEG.*').replace('\\','/')

eeg_files = glob.glob(pattern,recursive=True)
eeg_files = [x.replace('\\','/') for x in eeg_files]
foo=lambda x: os.path.join(os.path.dirname(x),'*labelresect*').replace('\\','/')
labelresect_files = []
for file in eeg_files:
    match_files = glob.glob(foo(file),recursive=True)
    assert len(match_files)==1
    labelresect_files.append(match_files[0])
def parse_bids(bidsname):
    entities=bidsname.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    d['suffix']=suffix
    return d

def parse_bids(bidsname):
    name = os.path.basename(bidsname)
    entities=name.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    if not '-' in suffix:
        d['suffix']=suffix
    else:
        a,b=suffix.split('-')
        d[a]=b
    return d


def get_suffix_from_path(x,suffixext):
    subrundir = os.path.dirname(x)
    files = glob.glob(os.path.join(subrundir,f'*_{suffixext}'))
    files = [x for x in files if os.path.isfile(x)]
    query = parse_bids(x)
    for f in files:
        candidate = parse_bids(f)
        winner = True
        if 'suffix' in candidate:
            del candidate['suffix']
        
        for key,val in candidate.items():
            if key in query:
                if query[key]!=candidate[key]:
                    winner = False
                if not winner:
                    break
        if winner:
            return f
    return None

labelresect2=[]
for file in eeg_files:
    labelresect_file = get_suffix_from_path(file,'labelresect*')
    labelresect2.append(labelresect_file)
    print(labelresect_file)

def loadmat(x,kwargs={}):
    try:
        return sio.loadmat(x,**kwargs)
    except:
        #print(f'Failed to load {x} as a mat file')
        return mat73.loadmat(x,**kwargs)

for file in labelresect2:
    if '.mat' in file:
        data = loadmat(file)
    elif '.npy' in file:
        data = np.load(file,allow_pickle=True)
    break

def make_suffix_from_path(x,suffixext):
    oldsuffixext= x.split('_')[-1]
    return x.replace(oldsuffixext,suffixext)

def get_subdict_from_path(x):
    bidspath = x[:x.find('sub-')]
    acq = parse_bids(x)['acq']
    subinfo = loadmat(os.path.join(bidspath,'subinfo.mat',),dict(simplify_cells=True))['SubInfo'][acq]
    subs = [d['Name'] for d in subinfo]
    sub_idx = subs.index(parse_bids(x)['sub'])
    subdict = subinfo[sub_idx]
    return subdict

def get_rundict_from_path(x):
    subdict = get_subdict_from_path(x)
    run = parse_bids(x)['run']
    if isinstance(subdict['SZ'],list):
        subruns =[sz['Name'].replace('_','') for sz in subdict['SZ']]
    else:
        if isinstance(subdict['SZ'],dict):
            subruns = [subdict['SZ']['Name'].replace('_','')]
            subdict['SZ'] = [subdict['SZ']]
        else:
            raise Exception
    idx_run = subruns.index(run)
    rundict = subdict['SZ'][idx_run]
    assert rundict['Name'].replace('_','') == run
    return rundict

def loader_iEEG(x,sfreq=None):
    data = loadmat(x)['F']
    subdict=get_subdict_from_path(x)
    rundict = get_rundict_from_path(x)
    if sfreq is None:
        sfreq1 = subdict['sfreq_orig']
        sfreq = loadmat(x)['Time']
        Ts = sfreq[1]-sfreq[0]
        sfreq = 1/Ts
    ch_types = 'eeg'
    ch_names=rundict['Channel']['SEEG']['Name'].tolist()
    assert data.shape[0] == len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(data, info,verbose=False)
    return raw

import copy
eeg_files2=copy.deepcopy(eeg_files)
# remove fif files that have both mat and fif
for x in eeg_files2:
    if '.fif' in x:
        if x.replace('.fif','.mat') in eeg_files:
            eeg_files.remove(x)

ok=[]
total=len(eeg_files)
for i,file in enumerate(eeg_files):
    try:
        if '.mat' in file:
            raw = loader_iEEG(file)
            if not os.path.isfile(make_suffix_from_path(file,'iEEG.fif')):
                raw.save(make_suffix_from_path(file,'iEEG.fif'),overwrite=True)
        elif '.fif' in file:
            raw = mne.io.read_raw_fif(file,preload=True)
        
        labelresect=get_suffix_from_path(file,'labelresect*')
        if '.mat' in labelresect:
            labelresect = np.squeeze(loadmat(labelresect)['LabelResect']).astype(bool).tolist()
        elif '.npy' in labelresect:
            labelresect = np.load(labelresect,allow_pickle=True).astype(bool).tolist()
        flag=get_suffix_from_path(file,'flag*')
        if '.mat' in flag:
            flag = np.squeeze(loadmat(flag)['BadChannel']).astype(bool).tolist()
        elif '.npy' in flag:
            flag = np.load(flag,allow_pickle=True).astype(bool).tolist()
        chs=raw.info['ch_names']
        dict1={'labelresect':labelresect,'flag':flag,'chs':chs}
        parse_bids(file)
        num=len(chs)
        dict1.update({x:[y]*num for x,y in parse_bids(file).items()})
        ok.append(dict1)

        print('ok',i,total)
    except:
        print(f'Failed to load {file}')
        #ok.append(False)

df=[pd.DataFrame.from_dict(k) for k in ok]
df=pd.concat(df,axis=0,ignore_index=True)
df.to_csv('Y:/datasets/epilepsy/epilepsy.csv',index=False)
ok[-1]

