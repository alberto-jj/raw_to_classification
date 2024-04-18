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

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['ml']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)

df_path = os.path.join(PIPELINE['aggregate']['path'],PIPELINE['aggregate']['filename'])
df = pd.read_csv(df_path)


targets=['age']
drops=['id','sex','group','dataset','subject','task']#,'id_ultimo_jefe']
all_drops=drops+targets
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.drop(all_drops, axis=1))
df_scaled = pd.DataFrame(df_scaled, columns=df.drop(all_drops, axis=1).columns)
num_folds=5

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
df_scaled_target=df_scaled.copy()
df_scaled_target[targets[0]]=df[targets[0]]
# test_data = TabularDataset(f'{data_url}test.csv')
# predictor.evaluate(test_data, silent=True)
# y_pred = predictor.predict(test_data.drop(columns=[label]))
# y_pred.head()
# predictor.leaderboard(test_data)

params=dict(label=targets[0],
problem_type=None,
eval_metric='root_mean_squared_error',
path='None',
verbosity= 2, 
log_to_file=True,
log_file_path='auto',
sample_weight= None,
weight_evaluation= False,)
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
num_bag_folds=5,
)

predictor.leaderboard()