import os
import pandas as pd
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.model_selection import RepeatedKFold
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import shap
from eeg_raw_to_classification.utils import parse_bids,load_yaml,get_output_dict,save_dict_to_json,save_figs_in_html
import psutil
njobs = len(psutil.Process().cpu_affinity())

datasets = load_yaml('datasets.yml')
PIPELINE = load_yaml('pipeline.yml')
OUTPUT_DIR = PIPELINE['classification']['path']
os.makedirs(OUTPUT_DIR,exist_ok=True)


#### Single Split

input_datapath = os.path.join(PIPELINE['harmonization']['path'],'combatScaleSingleSplit.npy')
data = np.load(input_datapath,allow_pickle=True).item()['data']

X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
features = data['features']

lr = XGBClassifier(random_state=42, n_jobs=njobs, tree_method='gpu_hist' , gpu_id=0, predictor='auto')

sfs = SFS(lr, 
          k_features="parsimonious", 
          forward=True, 
          floating=False, 
          scoring='f1',
          verbose=2,
          cv=5,
          n_jobs = njobs)

sfs = sfs.fit(X_train, y_train)

plt.rcParams["figure.figsize"] = (40,10)
fig,ax = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
print(sfs.k_feature_names_)
print(sfs.k_score_)
save_figs_in_html(os.path.join(OUTPUT_DIR,'singleSFS.html'),[fig])
plt.close('all')
selected_feats = [int(x) for x in list(sfs.k_feature_names_)]

selected_feats = [features[i]for i in selected_feats]
save_dict_to_json(os.path.join(OUTPUT_DIR,'singleSFS.txt'),{'selected_features':selected_feats,'features':features})


###### CV

input_datapath = os.path.join(PIPELINE['harmonization']['path'],'combatScaleCV.npy')
data = np.load(input_datapath,allow_pickle=True).item()

metrics = ['auc', 'fpr', 'tpr', 'thresholds']
results = {
    'train': {m:[] for m in metrics},
    'val'  : {m:[] for m in metrics},
    'test' : {m:[] for m in metrics}
}

params = {
    'objective'   : 'binary:logistic',
    'eval_metric' : 'logloss'
}

plt.rcParams["figure.figsize"] = (10,10)

for k,fold in data.items():
    print(k)
    dataFold = fold['data']
    #print(dataFold.keys())
    X_train = dataFold['X_train']
    X_test  = dataFold['X_test']
    y_train = dataFold['y_train']
    y_test  = dataFold['y_test']
    X_val = np.concatenate([X_train,X_test],axis=0)
    y_val = np.concatenate([y_train,y_test],axis=0)
    dtest = xgb.DMatrix(X_test, label=y_test)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val, label=y_val)
    model  = xgb.train(
        dtrain                = dtrain,
        params                = params, 
        evals                 = [(dtrain, 'train'), (dtest, 'val')],
        num_boost_round       = 1000,
        verbose_eval          = False,
        early_stopping_rounds = 10,
    )
    sets = [dtrain, dtest,dval]
    for i,ds in enumerate(results.keys()):
        y_preds              = model.predict(sets[i])
        labels               = sets[i].get_label()
        fpr, tpr, thresholds = roc_curve(labels, y_preds)
        results[ds]['fpr'].append(fpr)
        results[ds]['tpr'].append(tpr)
        results[ds]['thresholds'].append(thresholds)
        results[ds]['auc'].append(roc_auc_score(labels, y_preds))
        #print(ds)

kind = 'val'
c_fill      = 'rgba(52, 152, 219, 0.2)'
c_line      = 'rgba(52, 152, 219, 0.5)'
c_line_main = 'rgba(41, 128, 185, 1.0)'
c_grid      = 'rgba(189, 195, 199, 0.5)'
c_annot     = 'rgba(149, 165, 166, 0.5)'
c_highlight = 'rgba(192, 57, 43, 1.0)'
fpr_mean    = np.linspace(0, 1, 100)
interp_tprs = []
for i in range(4):
    fpr           = results[kind]['fpr'][i]
    tpr           = results[kind]['tpr'][i]
    interp_tpr    = np.interp(fpr_mean, fpr, tpr)
    interp_tpr[0] = 0.0
    interp_tprs.append(interp_tpr)
tpr_mean     = np.mean(interp_tprs, axis=0)
tpr_mean[-1] = 1.0
tpr_std      = np.std(interp_tprs, axis=0)
tpr_upper    = np.clip(tpr_mean+tpr_std, 0, 1)
tpr_lower    = tpr_mean-tpr_std
auc          = np.mean(results[kind]['auc'])
fig = go.Figure([
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_upper,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'upper'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_lower,
        fill       = 'tonexty',
        fillcolor  = c_fill,
        line       = dict(color=c_line, width=1),
        hoverinfo  = "skip",
        showlegend = False,
        name       = 'lower'),
    go.Scatter(
        x          = fpr_mean,
        y          = tpr_mean,
        line       = dict(color=c_line_main, width=2),
        hoverinfo  = "skip",
        showlegend = True,
        name       = f'AUC: {auc:.3f}')
])
fig.add_shape(
    type ='line', 
    line =dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)
fig.update_layout(
    template    = 'plotly_white', 
    title_x     = 0.5,
    xaxis_title = "1 - Specificity",
    yaxis_title = "Sensitivity",
    width       = 600,
    height      = 600,
    legend      = dict(
        yanchor="bottom", 
        xanchor="right", 
        x=0.95,
        y=0.01,
    )
)
fig.update_yaxes(
    range       = [0, 1],
    gridcolor   = c_grid,
    scaleanchor = "x", 
    scaleratio  = 1,
    linecolor   = 'black')
fig.update_xaxes(
    range       = [0, 1],
    gridcolor   = c_grid,
    constrain   = 'domain',
    linecolor   = 'black')

#fig.show()
fig.write_image(os.path.join(OUTPUT_DIR,'CV_graph.png'))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
fig, ax = plt.subplots(figsize=(16, 8))
shap.summary_plot(shap_values, X_val,feature_names=features,plot_type="bar",show=False)
fig.savefig(os.path.join(OUTPUT_DIR,'CV_SHAP.png'))