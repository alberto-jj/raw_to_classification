#use automl environment
import pandas as pd
import pickle
from itertools import product
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
#import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedKFold
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#import umap
#import umap.plot
from itertools import product
import os
import matplotlib
from sklearn.svm import SVC
import yaml
import json
from sklearn.ensemble import GradientBoostingClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
from featurewiz import FeatureWiz
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.model_selection import StratifiedKFold
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
from sklearn.preprocessing import OrdinalEncoder
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import itertools

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['ml']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)

df_path = os.path.join(PIPELINE['aggregate']['path'],PIPELINE['aggregate']['filename'])
df = pd.read_csv(df_path)

# SETUP
NUM_FOLDS = 5
SCALING = 'combat' # combat or standard
targets=['age']
dropsToOnlyFeatures=['id', 'sex','group', 'dataset', 'subject', 'task'] + targets
stratifiedvars=['dataset','group','sex'] # these are categorical in general... , they should be the same as the covars in combat?
# you may include or not some variables if too many combinations dont exist, for example only dataset (site)

# The thing here is that some combinations may not actually exist in the dataset
# Guess we should just ignore those and collect up to the lowest represented

joined_stratifiedvars = [list(x) for i,x in df[stratifiedvars].iterrows()]
joined_stratifiedvars = ['_'.join(x) for x in joined_stratifiedvars]
df['combination']=joined_stratifiedvars
dropsToOnlyFeatures +=['combination']
existing_combinations = pd.unique(joined_stratifiedvars)
lowest_representation = pd.value_counts(joined_stratifiedvars).min()

# Another interesting check... (?)

for stratifiedvar in stratifiedvars:
    print(df[stratifiedvar].value_counts())

possibilities = []
for stratifiedvar in stratifiedvars:
    uni=df[stratifiedvar].unique()
    print(stratifiedvar,list(uni))
    possibilities.append(list(uni))

# See non existing combinations and report that

possible_combinations=list(itertools.product(*possibilities))
possible_combinations=['_'.join(x) for x in possible_combinations]

nonexisting_combinations = set(possible_combinations).difference(joined_stratifiedvars)

if len(nonexisting_combinations) > 0:
    print('WARNING, the following combinations dont exist:')
    print(nonexisting_combinations)

# For now balanced withing the combinations that exist?

random_seed = 0
df_balanced = []
for comb_ in existing_combinations:
    subdf=df[df['combination']==comb_]
    subdf=subdf.sample(n=lowest_representation,replace=False,random_state=random_seed)
    df_balanced.append(subdf)

df_orig = df.copy()
df = pd.concat(df_balanced)
dfX = df.drop(dropsToOnlyFeatures, axis=1)


assert lowest_representation*len(existing_combinations) == df.shape[0]

# Does this means that max val of NUM_FOLDS == lowest_representation, that is, if they are equal then 1 fold will only take 1 from each combination...
assert NUM_FOLDS <= lowest_representation

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_seed)

# Verify correct features
print(dfX.columns) # maybe this should go to some log...

# stratification based on the "y" variable in these arguments, hence, combinations
# But this would work better if all combinations were present...
# See disparities when printing the value_counts of the stratifiedvars

df['fold']=-1
folds = {}
for foldi,indexes in enumerate(skf.split(dfX,df['combination'])):
   print(foldi)
   train_index = indexes[0]
   test_index = indexes[1]
   folds[foldi] = {'train':train_index,'test':test_index}
   df['fold'].iloc[test_index]=foldi

# To maintain good account of column transformations

dropsToOnlyFeatures += ['fold']

# Now with the fold column we should be able to leverage the OneGroupOut of Autogluon
# But first apply scaling and transform from the training data of each fold

for foldi,fold in folds.items():
    train_index = folds['train']
    test_index = folds['test']

    dfX = df.copy().drop(dropsToOnlyFeatures,axis=-1)
    dfX_train = dfX.copy().iloc[train_index]
    dfX_test = dfX.copy().iloc[test_index]
    # we will have to save this scaled data for each fold as multiple scaling exist for the same row depending on which fold it is
    # remember that we scale on the training indexes (not unique across folds) rather than the test indexes (unique)

    if SCALING == 'combat':
        pass
    elif SCALING == 'standard':
        scaler = StandardScaler()
        scaler = scaler.fit(dfX_train)
        dfX_train = scaler.transform(dfX_train)
        dfX_test = scaler.transform(dfX_test)
        folds[foldi]['X_train'] = dfX_train.copy()
        folds[foldi]['X_test'] = dfX_test.copy() # this would be unique, but for simplicity just maintain with the same structure we have
        # dfX_scaled = pd.DataFrame(dfX_scaled, columns=df.drop(all_drops, axis=1).columns)
    else:
        # NO SCALING
        dfX = df.drop(dropsToOnlyFeatures,axis=-1)
# df this should be done later after getting the balanced dataset

## BUILD A REDUNDANT DATAFRAME WITH TEST AND TRAIN OF EACH FOLD IN SEPARATE ROWS, WITH EXTRA COLUMNS OF FOLDS AND TYPE (TRAIN OR TEST)
## FOR AUTOMLJAR PASS THE APPROPIATE INDEXES FOR EACH FOLD'S TRAIN AND TEST
## FOR AUTOGULON IS HARDER, WE HAVE TO MANIPULATE THE LEAVEONEGROUPOUT
## WHEN TRAINING FOLD 1 the training will include folds 2,3,4,5 training data which include for example data from fold 2 with multiple scalings
## SO THIS IS NOT TRIVIAL...
## MAYBE THE BEST IS TO USE AUTOGLUON FROM THE OUTSIDE, THAT IS PASS ONE FOR PER AUTOGLUON CALL (NO BAGGING/FOLDS)
## AND COLLECT THE RESULTING DATAFRAMES FROM STATITSTICS

if False:
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)
    # for train_index, test_index in skf.split(X_train, y_train):
    #     X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
    #     y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

        # Your code for training and evaluating the model on each fold goes here




    fwiz = FeatureWiz(corr_limit=0.75, feature_engg='', category_encoders='', dask_xgboost_flag=False, nrows=None, verbose=0)
    X_train_selected = fwiz.fit(df_scaled, df[targets])
    ### get list of selected features ###
    fwiz.features  


    #https://supervised.mljar.com/api/
    mode='Explain'#'Perform'#'Compete'#'Explain'
    imb = 'default'
    CV_TYPES={'kfold':{
        "validation_type": "kfold",
        "k_folds": 5,
        "shuffle": True,
        "stratify": True,
        "random_seed": 123
        },
    'custom':{'validation_type': 'custom'},
    'split':{
        "validation_type": "split",
        "train_ratio": 0.75,
        "shuffle": True,
        "stratify": True
        }
    }

    #algos = ['Baseline', 'Linear', 'Decision Tree','LightGBM', 'Xgboost', 'CatBoost']
    #'auto'#['Baseline', 'Linear', 'Decision Tree', 'Extra Trees', 'LightGBM', 'Xgboost', 'CatBoost', 'Nearest Neighbors']
    algos = ['Baseline', 'Linear', 'Decision Tree', 'Random Forest', 'Extra Trees', 'LightGBM', 'Xgboost', 'CatBoost', 'Neural Network', 'Nearest Neighbors']
    auto_init = dict(algorithms=algos, explain_level=0, ml_task='auto', mode=mode, eval_metric='rmse', validation_strategy=CV_TYPES['custom'], model_time_limit=60*60)
    internal_njobs=10
    automl = AutoML(**auto_init,n_jobs=internal_njobs)
    automl.fit(df_scaled, df[targets],cv=list(skf.split(df_scaled, df[targets])))



#https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.html

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import LeaveOneGroupOut
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold



np.unique()


for train_index, test_index in mskf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

df['multilabel']=[[i,j] for i,j in zip(df['split'],df['sex'])]

# Determine the the leaveoutvar with least samples


logo = LeaveOneGroupOut()
splits=logo.split(df,df[targets],df[leaveoutvar])
splitnum=0
for train_index, test_index in splits:

    for leaveoutInstance in df[leaveoutvar].unique():




    X_train, X_test = df_scaled.iloc[train_index], df_scaled.iloc[test_index]
    y_train, y_test = df[targets].iloc[train_index], df[targets].iloc[test_index]
    groups=df['split'].iloc[train_index]
    print(np.unique(groups,return_counts=True))
    df_train=df.iloc[train_index]
    #print(df_train.describe())
    print(splitnum)
    splitnum+=1



mskf = MultilabelStratifiedKFold(n_splits=df['split'].unique().shape[0], shuffle=True, random_state=0)
# this is cool, but this is not what i need, We want leaveonegroupout but also stratified
for train_index, test_index in mskf.split(df_scaled, enc.transform(df[strata])):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = df_scaled.iloc[train_index], df_scaled.iloc[test_index]
   y_train, y_test = df[targets].iloc[train_index], df[targets].iloc[test_index]
   print(np.unique(df['split'].iloc[train_index],return_counts=True))


unique_combinations = np.array([f"{gender}_{group}" for gender, group in zip(df['sex'], df['split'])])

# Create folds while leaving one group out
skf = StratifiedKFold(n_splits=len(df['split'].unique()), shuffle=True, random_state=42)

for train_index, test_index in skf.split(df_scaled, unique_combinations):
    train_data, test_data = df_scaled.iloc[train_index], df_scaled.iloc[test_index]
    print(np.unique(df['split'].iloc[train_index],return_counts=True))
    print(np.unique(df['sex'].iloc[train_index],return_counts=True))



from sklearn.model_selection import StratifiedKFold
import pandas as pd

# Assuming 'df' is your DataFrame where each row contains features and labels, and 'sex' and 'split' are columns representing the respective variables

# Split data by sex
male_data = df[df['sex'] == 'M']
female_data = df[df['sex'] == 'F']

# Combine 'sex' and 'split' to create a unique identifier for each combination
unique_combinations_male = male_data['sex'] + '_' + male_data['split']
unique_combinations_female = female_data['sex'] + '_' + female_data['split']

# Create folds leaving one group out
skf = StratifiedKFold(n_splits=len(df['split'].unique()), shuffle=True, random_state=42)

for male_index, female_index in skf.split(male_data, unique_combinations_male):
    train_male, test_male = male_data.iloc[male_index], male_data.iloc[female_index]
    train_female, test_female = female_data.iloc[male_index], female_data.iloc[female_index]
    
    train_data = pd.concat([train_male, train_female])
    test_data = pd.concat([test_male, test_female])
    
    print(np.unique(df['split'].iloc[train_index],return_counts=True))
    print(np.unique(df['sex'].iloc[train_index],return_counts=True))
    # Now 'train_data' and 'test_data' contain balanced folds with respect to sex, leaving out one group in each fold


logo = LeaveOneGroupOut()
splits=logo.split(df_scaled, df[targets], df['split'])
splitnum=0
for train_index, test_index in splits:
    X_train, X_test = df_scaled.iloc[train_index], df_scaled.iloc[test_index]
    y_train, y_test = df[targets].iloc[train_index], df[targets].iloc[test_index]
    groups=df['split'].iloc[train_index]
    print(np.unique(groups,return_counts=True))
    df_train=df.iloc[train_index]
    #print(df_train.describe())
    print(splitnum)
    splitnum+=1


numfolder=0
while True:
    numpath=f'./data/autogluon/autogluon_{numfolder}'
    numpathlog=numpath.replace('autogluon','autogluonlogs')
    if not os.path.exists(numpath):
        #os.makedirs(numpath) autogluon will create it
        break
    numfolder+=1

df_scaled_target=df_scaled.copy()
df_scaled_target[targets[0]]=df[targets[0]]
df_scaled_target['split']=df['split']
# test_data = TabularDataset(f'{data_url}test.csv')
# predictor.evaluate(test_data, silent=True)
# y_pred = predictor.predict(test_data.drop(columns=[label]))
# y_pred.head()
# predictor.leaderboard(test_data)

params=dict(label=targets[0],
problem_type=None,
eval_metric='root_mean_squared_error',
path=numpath,
verbosity= 2, 
log_to_file=True,
log_file_path=numpathlog,
sample_weight= None,
weight_evaluation= False,
groups='split',
)
#groups=targets[0])

predictor = TabularPredictor(**params)

predictor.fit(df_scaled_target, 
tuning_data=None, 
time_limit = None,
presets = None,
hyperparameters = None,
feature_metadata='infer',
infer_limit = None,
infer_limit_batch_size = None,
fit_weighted_ensemble = True, 
fit_full_last_level_weighted_ensemble = True,
full_weighted_ensemble_additionally  = False,
dynamic_stacking = False,
calibrate_decision_threshold = False,
num_cpus=10,
num_gpus='auto',
num_bag_folds=None,
)

leadf=predictor.leaderboard()
leadf.shape

