import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from neuroCombat import neuroCombat,neuroCombatFromTraining
import numpy as np
import copy
from sklearn.model_selection import StratifiedKFold


datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['harmonization']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)
df_path = os.path.join(PIPELINE['aggregate']['path'],PIPELINE['aggregate']['filename'])
df = pd.read_csv(df_path)

MAX_FEATURES=PIPELINE['harmonization']['MAX_FEATURES']
if MAX_FEATURES is not None:
    df = df.iloc[:,:MAX_FEATURES]

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
    combat['orig_data']=data
    combat['batch_col']=batch_col
    combat['batch_vals']=covars[batch_col]
    combat['estimates']['batches']= combat['estimates']['batches'].astype(int)
    return combat

df = dfFullNumber.copy()
combat = combatModel(df,NC)
np.save(os.path.join(OUTPUT_DIR,'combatFullData.npy'),combat)


X_train, X_test, y_train, y_test = train_test_split(df, df[SP['target']], test_size=SP['test'], random_state=42, stratify=df[SP['target']])

def combatAndScale(X_train,X_test,y_train,y_test):
    output = {}
    #idx_train= y_train.index
    #idx_test = y_test.index
    combat_train = combatModel(X_train,NC)
    combat_test = combatModel(X_test,NC)
    combat_test = neuroCombatFromTraining(combat_test['orig_data'],combat_test['batch_vals'],combat_train['estimates'])
    #combat_train['idx']=idx_train
    #combat_test['idx']=idx_test
    output['combat_train']=combat_train
    output['combat_test']=combat_test

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

    output['data'] = {
        'X_train':X_train,
        'X_test':X_test,
        'y_train':y_train,
        'y_test':y_test,
        'X_axes':data_axes,
        'y_axes':target_axes,
        #'train_idx':idx_train,
        #'test_idx':idx_test,
        'features':features
    }
    return output

singleSplit = combatAndScale(X_train, X_test, y_train, y_test)
singleSplit['train_idx']=y_train.index
singleSplit['test_idx']=y_test.index
np.save(os.path.join(OUTPUT_DIR,'combatScaleSingleSplit.npy'),singleSplit)



df = dfFullNumber.copy()
cv    = StratifiedKFold(n_splits=SP['folds'], shuffle = True, random_state = 42)
folds = [(train,test) for train, test in cv.split(df, df[SP['target']])]

cv_dict = {}
fold = 'fold-'
for k,vals in enumerate(folds):
    train,test = vals
    cv_dict[fold+str(k)]={}
    cv_dict[fold+str(k)]['train_idx']=train
    cv_dict[fold+str(k)]['test_idx']=test
    Y = df.copy()[SP['target']]
    X_train = df.copy().iloc[train,:]
    X_test = df.copy().iloc[test,:]
    y_train = Y.iloc[train]
    y_test = Y.iloc[test]
    singleSplit = combatAndScale(X_train, X_test, y_train, y_test)
    singleSplit['train_idx']=y_train.index
    singleSplit['test_idx']=y_test.index

    cv_dict[fold+str(k)]=singleSplit 

np.save(os.path.join(OUTPUT_DIR,'combatScaleCV.npy'),cv_dict)
