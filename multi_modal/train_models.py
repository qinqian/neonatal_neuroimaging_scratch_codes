## add imbalance resampling: https:a/imbalanced-learn.org/en/stable/auto_examples/applications/plot_over_sampling_benchmark_lfw.html#sphx-glr-auto-examples-applications-plot-over-sampling-benchmark-lfw-py

## https://www.kaggle.com/qianchao/smote-with-imbalance-data
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, f_regression, SelectKBest, SelectPercentile, SelectFwe, \
  f_classif, SelectFdr
from sklearn.metrics.scorer import make_scorer
import xgboost as xgb

import threading
threading.current_thread().name = 'MainThread'
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
import ants
from imblearn.pipeline import make_pipeline

import matplotlib
from collections import OrderedDict
import multiprocessing as mp
matplotlib.use('Agg')
import pandas as pd
#from sklearn.externals import joblib
import joblib
import scipy
from itertools import combinations
import numpy as np
#from skimage import measure
from matplotlib import gridspec
import os, sys
import glob
import json
import os
import copy
#from statsmodels.regression.linear_model import OLS
from sklearn.linear_model import Lasso, logistic, LinearRegression, LogisticRegression, RidgeClassifier, RidgeClassifierCV, Ridge
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_selection import SelectFromModel, f_regression, SelectKBest, SelectPercentile
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
from scipy import stats
from shutil import copyfile
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

class SelectP(SelectFdr):
    def __init__(self, score_func, alpha):
        super().__init__(score_func, alpha)
        
    def _get_support_mask(self):
        return self.pvalues_ <= self.alpha  # similar strategy as STM autism

def specificity_loss_func(ground_truth, predictions):
    tp, tn, fn, fp = 0.0,0.0,0.0,0.0
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    return tn/(tn+fp)

def fpr_loss_func(ground_truth, predictions):
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    return 1.0*fp/(tn+fp)

spec_score = make_scorer(specificity_loss_func, greater_is_better=True)
#fpr_score = make_scorer(fpr_loss_func, greater_is_better=False)
fpr_score = make_scorer(fpr_loss_func, greater_is_better=True)

def get_model(reg=False, grid=False):
    if reg:
        lasso_steps = [
            ('scaler', StandardScaler()),
            ('clf', Lasso(max_iter=2000))
        ]
        lasso_params = {'clf__alpha': scipy.stats.distributions.uniform(0, 10000)}
        lasso_pipe = Pipeline(lasso_steps)

        svm_k_steps = [
            ('scaler', StandardScaler()),
            #('fs', SelectKBest(f_regression)),
            #('fs', SelectPercentile(f_regression)),
            ('clf', SVR(kernel='linear', epsilon=1e-4))
        ]
        svm_k_params = {
            #'fs__k': np.arange(2, 100, 3),
            #'SVR__C': np.linspace(1e-5, 1e2, 100),
            'clf__C': scipy.stats.distributions.uniform(0, 10000),
            'clf__kernel': ['linear'] # , 'rbf', 'sigmoid', 'poly']
        }
        svm_k_pipe = Pipeline(svm_k_steps)

        ridge_steps = [
            ('scaler', StandardScaler()),
            ('clf', Ridge())
        ]
        ridge_params = {'clf__alpha': scipy.stats.distributions.uniform(0, 10000)}
        ridge_pipe = Pipeline(ridge_steps)

        rf_k_steps = [
            ('scaler', StandardScaler()),
            ('fs', SelectKBest(f_regression)),
            ('RF', RandomForestRegressor())
        ]
        rf_k_params = {
            'fs__k': np.arange(10, 50, 10),
            'RF__n_estimators': np.arange(2, 10, 2),
            'RF__max_features': np.arange(2, 10, 3)
        }
        rf_k_pipe = Pipeline(rf_k_steps)
    else:
        clf = xgb.XGBClassifier(learning_rate=0.1, #reg_lambda=1e-8, reg_alpha=1e-8,
                          n_estimators=10,
                          max_depth=10,
                          min_child_weight = 1,
                          gamma=0.,
                          subsample=0.9,
                          colsample_btree=0.9,
                          objective='binary:logistic',
                          random_state=seed)
        clf = Pipeline([
            ("std", StandardScaler()),
          #  ("fs",  SelectP(f_classif, alpha=0.05)),
            #('smote', SMOTE(random_state=seed)),    
        #        ("fs",  SelectKBest(f_classif, k=122)),
            ("clf", clf)
        ])
        params = {
            'clf__n_estimators': [2, 8, 10, 20],
            'clf__max_depth': [2, 8, 10, 20],
        #    'clf__reg_lambda': [100, 200, 300, 400, 500]
            'clf__reg_lambda': scipy.stats.distributions.uniform(1000, 100000)    
        }

        lasso_steps = [
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty="l1", solver='liblinear'))
        ]
        lasso_params = {'clf__C': scipy.stats.distributions.uniform(1000, 1000000)}
        lasso_pipe = Pipeline(lasso_steps)

        ridge_steps = [
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty="l2", solver='liblinear'))
        ]
        ridge_params = {'clf__C': scipy.stats.distributions.uniform(10, 10000)}
        ridge_pipe = Pipeline(ridge_steps)

        svm_k_steps = [
            ('scaler', StandardScaler()),
            #('fs', SelectKBest(f_regression)),
            #('fs', SelectPercentile(f_regression)),
            ('clf', SVC(probability=True, kernel='linear'))
        ]
        svm_k_params = {
            #'fs__k': np.arange(2, 100, 3),
            #'SVR__C': np.linspace(1e-5, 1e2, 100),
            'clf__C': scipy.stats.distributions.uniform(1000, 1000000),
            'clf__kernel': ['linear'] #, 'rbf', 'sigmoid', 'poly']
        }
        svm_k_pipe = Pipeline(svm_k_steps)
        rf_k_steps = [
            ('scaler', StandardScaler()),
            ('fs', SelectKBest(f_regression)),
            ('RF', RandomForestClassifier())
        ]
        rf_k_params = {
            'fs__k': np.arange(10, 50, 10),
            'RF__n_estimators': np.arange(2, 10, 2),
            'RF__max_features': np.arange(2, 10, 3)
        }
        rf_k_pipe = Pipeline(rf_k_steps)
    #return zip(['SVM', 'LASSO', 'Ridge'], [svm_k_pipe, lasso_pipe, ridge_pipe], [svm_k_params, lasso_params, ridge_params])
    #return zip(['XGBOOST', 'SVM', 'LASSO'], [xgb_model, svm_k_pipe, lasso_pipe], [xgb_params, svm_k_params, lasso_params])
    #return zip(['SVM', 'LASSO'], [svm_k_pipe, lasso_pipe], [svm_k_params, lasso_params])
    return zip(['LASSO'], [lasso_pipe], [lasso_params])


def get_high_quality():
    bold_bad_subs = []
    for i in glob.glob('results/processed/*badtimes0.2*'):
        with open(i) as inf:
            if 155-len(inf.readline().strip().split())<=30:
                bold_bad_subs.append(os.path.basename(i).split('_')[0])
    bold_bad_subs.append('sub-19')
    bold_bad_subs.append('sub-54')
    df = pd.read_table('kai_2017_dti_t1_bold_to_be_processed.xls')
    df.drop('ID.1', axis=1, inplace=True)
    df.loc[pd.isnull(df.ga), 'ga'] = df.ga.mean()
    df.loc[pd.isnull(df.age), 'age'] = df.age.mean()
    cga = df.ga + df.age
    df.loc[:, 'cga'] = cga
    df.loc[:, 'sex'] = LabelBinarizer().fit(df.sex).transform(df.sex)
    df.loc[pd.isnull(df.bw), 'bw'] = np.median(df.bw[~pd.isnull(df.bw)])

    bold_quality = np.ones(df.shape[0])
    bold_quality[df.participant_id.isin(bold_bad_subs)] = 0
    df.loc[:, 'bold_quality'] = bold_quality
    df.loc[:, 'preterm'] = (df.ga < 37).astype(int)
    native_fs = pd.read_table('all_files_in_native_space.txt', header=None, delimiter=' ')
    native_fs.index = native_fs.loc[:, 0]
    native_fs_tmp = df.merge(native_fs, left_on='participant_id', right_index=True)
    native_fs_tmp.index = native_fs_tmp.participant_id
    native_fs_out = native_fs_tmp.loc[list(native_fs.index), :]
    return native_fs_out


def get_images_atlas(all_modalities_metric):
    template_b0 = ants.image_read('results/processed/sub-143_T1w_noise_corrected_reoriented_enh_b0.nii.gz')
    template_mask = template_b0.get_mask()
    if not os.path.exists('tissue_label_mat.npy'):
        labels4d  = ants.image_read(f'{native_modalities[0]}_local_template.nii.gz')
        labelsmat = ants.timeseries_to_matrix(labels4d, template_mask)
        np.save('tissue_label_mat.npy', labelsmat)
    else:
        labelsmat = np.load('tissue_label_mat.npy') 

    template_images_dict = OrderedDict()
    for i, n in enumerate(all_modalities_metric):
        print(n, 'image')
        template_images_dict[n] = f'{all_modalities_metric[i]}_local_template.nii.gz'

    template_mat_dict = OrderedDict()
    for i, n in enumerate(all_modalities_metric):
        print(n, 'load')
        #template_mat_dict[n] = ants.timeseries_to_matrix(ants.image_read(template_images_dict[n]), template_mask)
        #np.save(f'{n}_template_mat.npy', template_mat_dict[n])
        template_mat_dict[n] = np.load(f'{n}_template_mat.npy')
    jhu_atlas = ants.image_read('results/template/JHU_Atlas_JHU_neonate_SS_122parcellation.nii.gz')
    jhu_mat = ants.images_to_matrix([jhu_atlas], template_mask)
    return labelsmat, template_mat_dict, jhu_mat, template_mask, template_b0


def non_zero_mean(x):
    if np.sum(x>0) == 0:
        return 0
    return np.median(x[x>0]) # x>0

def non_zero_md(x):
    if np.sum(x>0) == 0:
        return 0
    return np.median(x[x>0]) * 1000

def get_mean_jac(x):
    if np.sum(x>0) == 0:
        return 0
    return np.mean(x[x>0])

def get_roi_val(outcomes_continue):
    native_modalities = ['tissue_label', 'b0', 't1', 
                         'jac', 'md', 'alff', 'falff', 'reho', 'fa',  
                         'od', 'icvf', 'isovf']
    all_modalities_metric = native_modalities[3:]
    features_label = ['JAC', 'MD', 'ALFF', 'fALFF', 'ReHo', 'FA', 'OD', 'ICVF', 'ISOVF']
    native_fs_out = get_high_quality()
    labelsmat, template_mat_dict, jhu_mat, template_mask, template_b0 = get_images_atlas(all_modalities_metric)
    whole_brain = (labelsmat == 2) | (labelsmat == 3) | (labelsmat == 6) | (labelsmat == 7) | (labelsmat == 8) | (labelsmat == 9)

    jhu_annotations = pd.read_csv('../../data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv', header=None)
    jhu_annotations_dict = {}
    jhu_annotations_dict2 = {}
    for i in range(jhu_annotations.shape[0]):
        jhu_annotations_dict[jhu_annotations.iloc[i, 0]] = '_'.join(jhu_annotations.iloc[i, 1:-1].tolist())
        jhu_annotations_dict2['_'.join(jhu_annotations.iloc[i, 1:-1].tolist())] = jhu_annotations.iloc[i, 0]

    native_fs_quality, = np.where((native_fs_out.bold_quality ==1))
    template_mat_array = []
    in_modals = list(template_mat_dict.keys())

    for i in template_mat_dict:
        print(i, 'filter')
        template_mat_array.append(template_mat_dict[i][native_fs_quality] * whole_brain[native_fs_quality])

    template_mat_array = np.stack(template_mat_array, axis=0)
    del template_mat_dict
    col_label = []
    jhu_rois = []
   
    for roi_num in np.unique(jhu_mat)[1:]:
        roi_selection = jhu_mat == roi_num
        filter_nonbrain_img = template_mat_array[:, :, roi_selection[0]]
        print(np.sum(roi_selection))
        print(filter_nonbrain_img.shape)
        filter_nonbrain_img_meanj = np.apply_along_axis(get_mean_jac, 1, filter_nonbrain_img[0, :, :]).reshape(-1, filter_nonbrain_img.shape[1]) # jac
        filter_nonbrain_img_meanm = np.apply_along_axis(non_zero_md, 1, filter_nonbrain_img[1, :, :]).reshape(-1, filter_nonbrain_img.shape[1]) # md
        filter_nonbrain_img_meano = np.apply_along_axis(non_zero_mean, 2, filter_nonbrain_img[2:, :, :]) 
        print(filter_nonbrain_img_meanj.shape, filter_nonbrain_img_meanm.shape, filter_nonbrain_img_meano.shape)
        filter_nonbrain_img_mean = np.concatenate([filter_nonbrain_img_meanj, filter_nonbrain_img_meanm, filter_nonbrain_img_meano], axis=0)
        print(filter_nonbrain_img_mean.shape)
        #print(roi_num, n, f'{label}_{jhu_annotations_dict[roi_num]}')
        jhu_rois.append(filter_nonbrain_img_mean)
        col_label += [f'{n}_{jhu_annotations_dict[roi_num]}' for n, m in zip(features_label, all_modalities_metric)]

    jhu_rois_df = pd.DataFrame(np.vstack(jhu_rois).T, columns=col_label)
    jhu_rois_df.loc[:, 'CGA'] = native_fs_out.cga[native_fs_quality].values
    jhu_rois_df.loc[:, 'GA'] = native_fs_out.ga[native_fs_quality].values
    jhu_rois_df.loc[:, 'age'] = native_fs_out.age[native_fs_quality].values
    jhu_rois_df.loc[:, 'BEAP'] = native_fs_out.BEAP[native_fs_quality].values
    jhu_rois_df.index = native_fs_out.iloc[native_fs_quality, ].index
    jhu_roi_outcome_model = pd.concat([jhu_rois_df, native_fs_out.iloc[native_fs_quality, ].loc[:, outcomes_continue+['preterm', 'bw', 'sex']]], axis=1)
    jhu_roi_outcome_model.to_csv('jhu_roi_model_raw_fillna.csv')
    return jhu_roi_outcome_model


def train_model(jhu_roi_outcome_model, outcomes_continue, seed, reg=True, with_BEAP=True):
    if with_BEAP:
       jhu_roi_outcome_model = jhu_roi_outcome_model.loc[~pd.isnull(jhu_roi_outcome_model.BEAP), :]
    jhu_roi_outcome_model.columns = jhu_roi_outcome_model.columns.map(lambda x:x.replace(' ', '_').replace('-', '_'))

    cog = ((jhu_roi_outcome_model.loc[:, ['Cognitive_12m']] < 85).sum(axis=1) >= 1).astype(int)
    lan = ((jhu_roi_outcome_model.loc[:, ['Receptive_Communication_12m', 'Receptive_Communication_18m', 
                             'Expressive_Communication_12m', 'Expressive_Communication_18m']] < 85).sum(axis=1) >= 1).astype(int)
    mot = ((jhu_roi_outcome_model.loc[:, ['Fine_Motor_12m', 'Fine_Motor_18m', 
                             'Gross_Motor_12m', 'Gross_Motor_18m']] < 85).sum(axis=1) >= 1).astype(int)

    features_list = [
        list(filter(lambda x:x.startswith('fALFF'), jhu_roi_outcome_model.columns[:9*122])),
        list(filter(lambda x:x.startswith('ALFF'), jhu_roi_outcome_model.columns[:9*122])),
        list(filter(lambda x:x.startswith('ReHo'), jhu_roi_outcome_model.columns[:9*122])),
        list(filter(lambda x:x.startswith('FA'), jhu_roi_outcome_model.columns[:9*122])),
        list(filter(lambda x:x.startswith('MD'), jhu_roi_outcome_model.columns[:9*122])),
        list(filter(lambda x:x.startswith('JAC'), jhu_roi_outcome_model.columns[:9*122])),   
        list(filter(lambda x:x.startswith('OD'), jhu_roi_outcome_model.columns[:9*122])),    
        list(filter(lambda x:x.startswith('ISOVF'), jhu_roi_outcome_model.columns[:9*122])),    
        list(filter(lambda x:x.startswith('ICVF'), jhu_roi_outcome_model.columns[:9*122])),            
        jhu_roi_outcome_model.columns[:(9*122)]
    ]
    features_label = ['fALFF', 'ALFF', 'ReHo', 'FA', 'MD', 'JAC', 'OD', 'ISOVF', 'ICVF', 'All']
    performances_jhu122 = []
    jhu_roi_outcome_model.loc[pd.isnull(jhu_roi_outcome_model.bw), 'bw'] = np.median(jhu_roi_outcome_model.bw[~pd.isnull(jhu_roi_outcome_model.bw)])
    outcome_features = jhu_roi_outcome_model.loc[:, list(jhu_roi_outcome_model.columns[:(122*9)])+['Peak_TSB', 'GA', 'CGA', 'bw', 'sex']]

    for outcome, name in zip([cog, lan, mot],
                             ['Cognition', 'Language', 'Motor']):
        all_perfs = []
        cv = RepeatedStratifiedKFold(10, 10, random_state=seed)
        predictions = []
        print('running cross validation')
        clf = xgb.XGBClassifier(learning_rate=0.1, #reg_lambda=1e-8, reg_alpha=1e-8,
                          n_estimators=10,
                          max_depth=10,
                          min_child_weight = 1,
                          gamma=0.,
                          subsample=0.9,
                          colsample_btree=0.9,
                          objective='binary:logistic',
                          random_state=seed)
        clf = Pipeline([
            ("std", StandardScaler()),
            ("fs",  SelectP(f_classif, alpha=0.05)),
            #('smote', SMOTE(random_state=seed)),    
        #   ("fs",  SelectKBest(f_classif, k=122)),
            ("clf", clf)
        ])
        params = {
            'clf__n_estimators': [2, 5, 10, 20],
            'clf__max_depth': [2, 5, 10],
        #    'clf__reg_lambda': [100, 200, 300, 400, 500]
            'clf__reg_lambda': scipy.stats.distributions.uniform(1000, 1e5)
        }
        
        clf1 = Pipeline([
            ("std", StandardScaler()),
            ("fs",  SelectP(f_classif, alpha=0.05)),
            #('smote', SMOTE(random_state=seed)),
        #        ("fs",  SelectKBest(f_classif, k=122)),
            ("clf", SVC(C=1e8, kernel='linear', probability=True, max_iter=50000, random_state=seed))
        ])    
        params1 = {
        #    'clf__C': [100, 200, 300, 400, 500]
            'clf__C': scipy.stats.distributions.uniform(1000, 1e5)        
        }
        
        clf2 = Pipeline([
            ("std", StandardScaler()),
            ("fs",  SelectP(f_classif, alpha=0.05)),
        #        ("fs",  SelectKBest(f_classif, k=122)), 
            #('smote', SMOTE(random_state=seed)),    
            ("clf", LogisticRegression(penalty='l1', solver='liblinear', C=100000000,
                                       max_iter=5000, random_state=seed))
        ])
        params2 = {
        #    'clf__C': [100, 200, 300, 400, 500]
            'clf__C': scipy.stats.distributions.uniform(1000, 1e5)
        } 
        scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'PRAUC': make_scorer(average_precision_score, greater_is_better=True),
                   "specificity": spec_score, "FPR": fpr_score,
                   'f1': 'f1', 'recall': 'recall', 'precision': 'precision'}
        
        for m, clf, param in zip(['xgboost', 'svm', 'logit'], [clf, clf1, clf2], [params, params1, params2]):    
        #     model = GridSearchCV(clf,  param_grid=param, scoring=scoring, 
        #                          refit='f1', cv=cv, n_jobs=30, return_train_score=True)    
            model = RandomizedSearchCV(clf, param_distributions=param, n_iter=60, cv=cv, refit='f1', n_jobs=30,
                                       scoring=scoring,
                                       return_train_score=False)
            #joblib.dump(grid, f'{name}_{n}_{out_label}_{seed}_{with_BEAP}.pklz')

            model.fit(outcome_features, outcome, groups=outcome)
            print(model.best_params_)
            model_perf = pd.DataFrame(model.cv_results_)    
            model_perf = model_perf.loc[:, model_perf.columns.str.contains('split') & model_perf.columns.str.contains('test')]
            all_perfs.append([m, seed, name,
                              model.cv_results_['mean_test_AUC'][model.best_index_], 
                              model.cv_results_['std_test_AUC'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('AUC')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('AUC')].max(),
        
                              model.cv_results_['mean_test_PRAUC'][model.best_index_], 
                              model.cv_results_['std_test_PRAUC'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_PRAUC')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_PRAUC')].max(),
        
                              model.cv_results_['mean_test_f1'][model.best_index_], 
                              model.cv_results_['std_test_f1'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_f1')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_f1')].max(),

                              model.cv_results_['mean_test_recall'][model.best_index_], 
                              model.cv_results_['std_test_recall'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_recall')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_recall')].max(),
        
                              model.cv_results_['mean_test_precision'][model.best_index_], 
                              model.cv_results_['std_test_precision'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_precision')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_precision')].max(),

                              model.cv_results_['mean_test_Accuracy'][model.best_index_], 
                              model.cv_results_['std_test_Accuracy'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_Accuracy')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_Accuracy')].max(),
        
        
                              model.cv_results_['mean_test_specificity'][model.best_index_], 
                              model.cv_results_['std_test_specificity'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_specificity')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_specificity')].max(),

                              model.cv_results_['mean_test_FPR'][model.best_index_], 
                              model.cv_results_['std_test_FPR'][model.best_index_],
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_FPR')].min(),
                              model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_FPR')].max()])
        
        cols = ['AUC', 'PRAUC', 'f1', 'recall', 'precision', 'Accuracy', 'specificity', 'FPR']
        cols_all = []
        for i in cols:
            cols_all.append(i+'_mean')
            cols_all.append(i+'_sd')
            cols_all.append(i+'_min')
            cols_all.append(i+'_max')     
            
        perfs = pd.DataFrame(all_perfs)
        perfs.columns = ['method', 'seed', 'outcome'] + cols_all
        perfs = perfs.T
        perfs.to_csv('%s_model%s.csv' % (name, seed))

def train_beap_model(jhu_roi_outcome_model, outcomes_continue, seed, reg=True, with_BEAP=True):
    #from imblearn.pipeline import Pipeline
    jhu_roi_outcome_model.columns = jhu_roi_outcome_model.columns.map(lambda x:x.replace(' ', '_').replace('-', '_'))
    beap_features = jhu_roi_outcome_model.loc[:, list(jhu_roi_outcome_model.columns[:(122*9)])+['Peak_TSB', 'GA', 'CGA', 'bw', 'sex']].loc[~pd.isnull(jhu_roi_outcome_model.BEAP), :]
    beap = jhu_roi_outcome_model.BEAP[~pd.isnull(jhu_roi_outcome_model.BEAP)]
    print(beap_features.shape)
    print(beap.shape, beap.sum())
    all_perfs = []
    cv = RepeatedStratifiedKFold(5, 10, random_state=seed)
    predictions = []
    print('running cross validation')
    clf = xgb.XGBClassifier(learning_rate=0.1, #reg_lambda=1e-8, reg_alpha=1e-8,
                      n_estimators=10,
                      max_depth=10,
                      min_child_weight = 1,
                      gamma=0.,
                      subsample=0.9,
                      colsample_btree=0.9,
                      objective='binary:logistic',
                      random_state=seed)
    clf = Pipeline([
        ("std", StandardScaler()),
        ("fs",  SelectP(f_classif, alpha=0.05)),
        #('smote', SMOTE(random_state=seed)),    
    #   ("fs",  SelectKBest(f_classif, k=122)),
        ("clf", clf)
    ])
    params = {
        'clf__n_estimators': [2, 5, 10],
        'clf__max_depth': [2, 5, 10],
    #    'clf__reg_lambda': [100, 200, 300, 400, 500]
        'clf__reg_lambda': scipy.stats.distributions.uniform(1000, 1e5)
    }
    
    clf1 = Pipeline([
        ("std", StandardScaler()),
        ("fs",  SelectP(f_classif, alpha=0.05)),
        #('smote', SMOTE(random_state=seed)),
    #        ("fs",  SelectKBest(f_classif, k=122)),
        ("clf", SVC(C=1e8, kernel='linear', probability=True, max_iter=50000, random_state=seed))
    ])    
    params1 = {
    #    'clf__C': [100, 200, 300, 400, 500]
        'clf__C': scipy.stats.distributions.uniform(1000, 1e5)        
    }
    
    clf2 = Pipeline([
        ("std", StandardScaler()),
        ("fs",  SelectP(f_classif, alpha=0.05)),
    #        ("fs",  SelectKBest(f_classif, k=122)), 
        #('smote', SMOTE(random_state=seed)),    
        ("clf", LogisticRegression(penalty='l1', solver='liblinear', C=100000000,
                                   max_iter=5000, random_state=seed))
    ])    
    params2 = {
    #    'clf__C': [100, 200, 300, 400, 500]
        'clf__C': scipy.stats.distributions.uniform(1000, 1e5)
    } 
    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'PRAUC': make_scorer(average_precision_score, greater_is_better=True),
               "specificity": spec_score, "FPR": fpr_score,
               'f1': 'f1', 'recall': 'recall', 'precision': 'precision'}
    
    for m, clf, param in zip(['xgboost', 'svm', 'logit'], [clf, clf1, clf2], [params, params1, params2]):    
    #     model = GridSearchCV(clf,  param_grid=param, scoring=scoring, 
    #                          refit='f1', cv=cv, n_jobs=30, return_train_score=True)    
        model = RandomizedSearchCV(clf, param_distributions=param, n_iter=30, cv=cv, refit='f1', n_jobs=30,
                                   scoring=scoring,
                                   return_train_score=False)
        model.fit(beap_features, beap, groups=beap)
        print(model.best_params_)
        model_perf = pd.DataFrame(model.cv_results_)    
        model_perf = model_perf.loc[:, model_perf.columns.str.contains('split') & model_perf.columns.str.contains('test')]
        all_perfs.append([m, seed, 
                          model.cv_results_['mean_test_AUC'][model.best_index_], 
                          model.cv_results_['std_test_AUC'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('AUC')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('AUC')].max(),
    
                          model.cv_results_['mean_test_PRAUC'][model.best_index_], 
                          model.cv_results_['std_test_PRAUC'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_PRAUC')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_PRAUC')].max(),
    
                          model.cv_results_['mean_test_f1'][model.best_index_], 
                          model.cv_results_['std_test_f1'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_f1')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_f1')].max(),

                          model.cv_results_['mean_test_recall'][model.best_index_], 
                          model.cv_results_['std_test_recall'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_recall')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_recall')].max(),
    
                          model.cv_results_['mean_test_precision'][model.best_index_], 
                          model.cv_results_['std_test_precision'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_precision')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_precision')].max(),

                          model.cv_results_['mean_test_Accuracy'][model.best_index_], 
                          model.cv_results_['std_test_Accuracy'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_Accuracy')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_Accuracy')].max(),
    
                          model.cv_results_['mean_test_specificity'][model.best_index_], 
                          model.cv_results_['std_test_specificity'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_specificity')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_specificity')].max(),

                          model.cv_results_['mean_test_FPR'][model.best_index_], 
                          model.cv_results_['std_test_FPR'][model.best_index_],
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_FPR')].min(),
                          model_perf.iloc[model.best_index_, :].loc[model_perf.columns.str.contains('_FPR')].max()])
    
    cols = ['AUC', 'PRAUC', 'f1', 'recall', 'precision', 'Accuracy', 'specificity', 'FPR']
    cols_all = []
    for i in cols:
        cols_all.append(i+'_mean')
        cols_all.append(i+'_sd')
        cols_all.append(i+'_min')
        cols_all.append(i+'_max')     
        
    perfs = pd.DataFrame(all_perfs)
    perfs.columns = ['method', 'seed'] + cols_all
    perfs = perfs.T
    perfs.to_csv('beap_model%s.csv' % seed)

def main():
    force = True
    outcomes_continue = ['Cognitive_12m', 'Fine Motor_12m',
           'Gross Motor_12m', 'Receptive Communication_12m',
           'Expressive Communication_12m', 'Cognitive_18m', 'Fine Motor_18m',
           'Gross Motor_18m', 'Receptive Communication_18m',
           'Expressive Communication_18m']
    outcomes_binary = ['BEAP']

    if not os.path.exists('jhu_roi_model_raw_fillna.csv'):
        jhu_roi_outcome_model = get_roi_val(outcomes_continue)
    else:
        jhu_roi_outcome_model = pd.read_csv('jhu_roi_model_raw_fillna.csv', index_col=0)
    peaktsb = pd.read_csv('peak_tsb.txt', sep='\t')
    jhu_roi_outcome_model = jhu_roi_outcome_model.merge(peaktsb, left_on='participant_id', right_on='participant_id')
    final = jhu_roi_outcome_model.drop('participant_id', axis=1)
    seed = 9999
    #for seed in range(9000, 10000):
    #train_model(final, outcomes_continue, seed, reg=False, with_BEAP=False)
    train_beap_model(final, outcomes_continue, seed, reg=False, with_BEAP=False)

if __name__ == '__main__':
    main()
