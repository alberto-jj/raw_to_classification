import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from neuroCombat import neuroCombat
import numpy as np
import copy


datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['harmonization']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)
df_path = os.path.join(PIPELINE['aggregate']['path'],PIPELINE['aggregate']['filename'])
df = pd.read_csv(df_path)

SP = PIPELINE['harmonization']['split']
NC = PIPELINE['harmonization']['neuroCombat']

dforig = df.copy()

# Drop
df = df.drop(NC['drop'],axis='columns')

# Deal with categoricals
categorical_mappings ={}
for cat in NC['categorical']:
    levels = list(set(list(df[cat])))
    levels.sort()
    categorical_mappings[cat]={l:levels.index(l) for l in levels}

for key,val in categorical_mappings.items():
    foo = lambda x:val[x]
    df[key]=df[key].apply(foo)

save_dict_to_json(os.path.join(OUTPUT_DIR,'categoricalMappings.txt'),categorical_mappings)
df.to_csv(os.path.join(OUTPUT_DIR,'df_categoricalNumbers.csv'),index=False)

dfFullNumber = df.copy()

def combatModel(df,NC):
    # Full NeuroCombat, No Split

    # Specifying the batch (scanner variable) as well as a biological covariate to preserve:
    covars = {key:list(df[key]) for key in NC['covars']}
    covars = pd.DataFrame(covars)  

    # To specify names of the variables that are categorical:
    categorical_cols = NC['categorical']

    # To specify the name of the variable that encodes for the scanner/batch covariate:
    batch_col = NC['batch']

    cat_no_batch = copy.deepcopy(categorical_cols)
    cat_no_batch.remove(batch_col)

    data = df.drop(NC['covars'],axis='columns')
    data_cols = list(data.columns)
    data = data.to_numpy().transpose()

    #Harmonization step:
    combat = neuroCombat(dat=data,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=cat_no_batch)
    combat['data_cols']=data_cols
    combat['data_axes']=('features','samples')
    return combat

combat = combatModel(df,NC)
np.save(os.path.join(OUTPUT_DIR,'combatFullData.npy'),combat)

df = dfFullNumber.copy()
X_train, X_test, y_train, y_test = train_test_split(df, df[SP['target']], test_size=SP['test'], random_state=42, stratify=df[SP['target']])
idx_train= y_train.index
idx_test = y_test.index
combat_train = combatModel(X_train,NC)
combat_test = combatModel(X_test,NC)
combat_train['idx']=idx_train
combat_test['idx']=idx_test
np.save(os.path.join(OUTPUT_DIR,'combatTrainData.npy'),combat_train)
np.save(os.path.join(OUTPUT_DIR,'combatTestData.npy'),combat_test)

# Scaling

# fit the scaler on the training data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(combat_train['data'].transpose())
features = combat_train['data_cols']
# apply the same transformation to the test data
X_test = scaler.transform(combat_test['data'].transpose())
data_axes = ('samples','features')
target_axes = ('label')
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

data = {
    'X_train':X_train,
    'X_test':X_test,
    'y_train':y_train,
    'y_test':y_test,
    'X_axes':data_axes,
    'y_axes':target_axes,
    'train_idx':idx_train,
    'test_idx':idx_test
}

np.save(os.path.join(OUTPUT_DIR,'combatScaledData.npy'),data)
