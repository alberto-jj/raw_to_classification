
"""
pattern = '/home/yorguin/scratch/data/MEG_*/derivatives/prepDur30Ov20/**/*_featureError.txt'
files = glob.glob(pattern, recursive=True)

errors = []
for file in files:
    data = load_json(file)
    data['file'] = file
    errors.append(data)

df = pd.DataFrame(errors)

df.to_csv('/home/yorguin/scratch/data/featureErrors.csv', index=False)

def make_kind(x):
    if 'Disk quota exceeded' in x:
        return 'disk_quota'
    elif 'file id tag' in x:
        return 'id_tag'
    elif 'KeyboardInterrupt' in x:
        return 'keyboard_interrupt'
    else:
        return 'other'

df['kind'] = df['error'].apply(make_kind)

# save csv for each kind
for kind in df['kind'].unique():
    kind_df = df[df['kind'] == kind]
    kind_df.to_csv(f'/home/yorguin/scratch/data/featureErrors_{kind}.csv', index=False)

df['kind'].value_counts().to_csv('/home/yorguin/scratch/data/featureErrors_counts.csv')

dfIdTag = df[df['kind'] == 'id_tag']

def get_file_id_tag(x):
    if 'PosixPath' in x:
        return x.split("PosixPath('")[1].split("')")[0]
    else:
        return None

dfIdTag['file_id_tag'] = dfIdTag['error'].apply(get_file_id_tag)

dfIdTag['file_id_tag'].value_counts().to_csv('/home/yorguin/scratch/data/featureErrors_id_tag_counts.csv')


for file_id_tag in dfIdTag['file_id_tag'].unique():
    print(file_id_tag)
    os.remove(file_id_tag)

# try to preprocess again to see if the error is fixed

# check if there are any split files in the bids source

pattern = '/home/yorguin/scratch/data/MEG_*/meg_data_BIDS/sub-*/ses-*/meg/*split*.fif'

splits = glob.glob(pattern, recursive=True)
"""
# no split files found...


import json
import glob
import pandas as pd
import shutil
import os
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


filemissing = '/home/yorguin/scratch/code/raw_to_classification/data/cocosprint/feature_inspection/features@prepDur30Ov20_missing_features.json'

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
data = load_json(filemissing)


from pprint import pprint

files = list(data["file_to_missing_indices"].keys())


for f in files:
    if os.path.isfile(f):
        print(f"File exists: {f}, removing it.")
        os.remove(f)