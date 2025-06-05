import glob

pattern = "/home/yorguin/scratch/data/*/derivatives/features@prepDur30Ov20/**/*.npy"

pattern = "Y:/computecanada/cocosprint/home/yorguin/scratch/data/*/derivatives/features@prepDur30Ov20/**/*.npy"

files = glob.glob(pattern, recursive=True)

import pandas as pd

df = pd.DataFrame(files, columns=["file_path"])

for s in df['file_path']:
    print(s)
df['suffix'] = df['file_path'].apply(lambda x: x.split('_')[-1])


df['suffix'].value_counts()

df.columns

import os
os.path.basename(df.iloc[0]['file_path'])