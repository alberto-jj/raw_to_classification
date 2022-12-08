import bids
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict
import itertools
import pandas as pd
datasets = load_yaml('datasets.yml')
DATASET = datasets['FINLAND']
pipeline_name = 'aggregate'

metric = 'relativePower'
pattern = os.path.join(DATASET.get('bids_root', None),'derivatives','prepare',f'**/*_{metric}.npy').replace('\\','/')
eegs = glob.glob(pattern,recursive=True)
output =os.path.join(DATASET.get('bids_root', None),'derivatives',pipeline_name,f'{metric}.csv')
os.makedirs(os.path.dirname(output),exist_ok=True)
dict_list=[]
for eeg_file in eegs:
    dict_list += get_output_dict(eeg_file,'WIDE',DATASET['dataset_label'])

df = pd.DataFrame(dict_list)
df.to_csv(output,index=False)