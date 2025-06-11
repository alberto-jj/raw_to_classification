import glob


pattern = "Y:/computecanada/cocosprint/home/yorguin/scratch/data/*/derivatives/features@prepDur30Ov20/**/*.npy"

pattern = "/home/yorguin/scratch/data/*/derivatives/features@prepDur30Ov20/**/*.npy"

files = glob.glob(pattern, recursive=True)
files = [x for x in files if (not 'split' in x or 'split-01' in x)] # IMPORTANT TO AVOID CONFUSING SPLITS FOR UNIQUE RECORDS

if False:
    files1 = [f for f in files if 'split-01' in f and not 'Mean' in f and not 'Var' in f]
    files2 = [f for f in files if 'split-02' in f and not 'Mean' in f and not 'Var' in f]

    import numpy as np

    epoch_len_dict = {}
    for this_f in files1 + files2:
        this_feat = np.load(this_f, allow_pickle=True).item()
        if 'epochs' in this_feat['metadata']['axes']:
            this_len = len(this_feat['metadata']['axes']['epochs'])
            this_file = '_'.join(this_f.split('_')[:-1])
            if not this_file in epoch_len_dict:
                epoch_len_dict[this_file] = this_len
                print(f"{this_file}: {this_len}")

    print(epoch_len_dict)


import pandas as pd

df = pd.DataFrame(files, columns=["file_path"])

# for s in df['file_path']:
#     print(s)
df['suffix'] = df['file_path'].apply(lambda x: x.split('_')[-1])


df['suffix'].value_counts()

df.columns

import os
os.path.basename(df.iloc[0]['file_path'])