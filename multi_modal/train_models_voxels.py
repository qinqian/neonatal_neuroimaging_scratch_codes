## add imbalance resampling: https:a/imbalanced-learn.org/en/stable/auto_examples/applications/plot_over_sampling_benchmark_lfw.html#sphx-glr-auto-examples-applications-plot-over-sampling-benchmark-lfw-py
from sklearn.feature_selection import SelectFromModel, f_regression, SelectKBest, SelectPercentile, SelectFwe, \
  f_classif, SelectFdr
from sklearn.metrics.scorer import make_scorer


## https://www.kaggle.com/qianchao/smote-with-imbalance-data

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
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer, RobustScaler
from sklearn.feature_selection import SelectFromModel, f_regression, SelectKBest, SelectPercentile, SelectFdr
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
from scipy import stats
from shutil import copyfile
import numpy as np
import pandas as pd

class SelectP(SelectFdr):
    def __init__(self, score_func, alpha):
        super().__init__(score_func, alpha)
        
    def _get_support_mask(self):
        return self.pvalues_ <= self.alpha  # similar strategy as STM autism

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
        lasso_steps = [
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(penalty="l1", solver='liblinear'))
        ]
        lasso_params = {'clf__C': scipy.stats.distributions.uniform(0, 10000)}
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
            'clf__C': scipy.stats.distributions.uniform(0, 10000),
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
    native_fs_quality, = np.where((native_fs_out.bold_quality ==1))
    if not os.path.exists('jhu_roi_model_raw_fillna.csv'):
        labelsmat, template_mat_dict, jhu_mat, template_mask, template_b0 = get_images_atlas(all_modalities_metric)
        whole_brain = (labelsmat == 2) | (labelsmat == 3) | (labelsmat == 6) | (labelsmat == 7) | (labelsmat == 8) | (labelsmat == 9)
    
        jhu_annotations = pd.read_csv('../../data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv', header=None)
        jhu_annotations_dict = {}
        jhu_annotations_dict2 = {}
        for i in range(jhu_annotations.shape[0]):
            jhu_annotations_dict[jhu_annotations.iloc[i, 0]] = '_'.join(jhu_annotations.iloc[i, 1:-1].tolist())
            jhu_annotations_dict2['_'.join(jhu_annotations.iloc[i, 1:-1].tolist())] = jhu_annotations.iloc[i, 0]
    
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
        return jhu_roi_outcome_model, native_fs_quality
    else:
        jhu_roi_outcome_model = pd.read_csv('jhu_roi_model_raw_fillna.csv')
        return  jhu_roi_outcome_model, native_fs_quality


def train_model(jhu_roi_outcome_model, outcomes_continue, seed, reg=True, with_BEAP=True):
    if with_BEAP:
       jhu_roi_outcome_model = jhu_roi_outcome_model.loc[~pd.isnull(jhu_roi_outcome_model.BEAP), :]

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
    for name, pipeline, parameters in get_model(reg):
        for feature_name, features in zip(features_label[-1:], features_list[-1:]):
            for i, n in enumerate(outcomes_continue):
                if isinstance(feature_name, str):
                    features_combined = features
                    features_combined_labels = [feature_name]
                else:
                    features_combined = []
                    for f in features:
                        features_combined += f
                    features_combined_labels = []
                    for f in feature_name:
                        features_combined_labels += f                
                if reg:
                    X_train, X_test, y_train, y_test =  train_test_split(jhu_roi_outcome_model.loc[:, list(features_combined)+['preterm', 'bw', 'sex', 'CGA', 'Peak-TSB']],
                                     jhu_roi_outcome_model.loc[:, outcomes_continue[i]], 
                                     random_state=seed, test_size=0.3)
                    y_train = (y_train-y_train.mean(axis=0))/y_train.std(axis=0)
                    y_test = (y_test-y_test.mean(axis=0))/y_test.std(axis=0)
                else:
                    if 'Cog' in outcomes_continue[i]:
                         cutoff = 75
                    elif 'Motor' in outcomes_continue[i]:
                         cutoff = 76
                    else:
                         cutoff = 74 # language
                    groups = np.zeros(jhu_roi_outcome_model.shape[0])
                    groups[(jhu_roi_outcome_model.GA < 35) & (jhu_roi_outcome_model.loc[:, 'Peak-TSB']<20)] = 1
                    groups[(jhu_roi_outcome_model.GA >= 35) & (jhu_roi_outcome_model.loc[:, 'Peak-TSB']<20)] = 2
                    groups[(jhu_roi_outcome_model.GA < 35) & (jhu_roi_outcome_model.loc[:, 'Peak-TSB']>=20)] = 3
                    print((jhu_roi_outcome_model.loc[:, outcomes_continue[i]] < cutoff).sum())
                    if with_BEAP:
                        baseline_feat = ['preterm', 'GA', 'bw', 'sex', 'CGA', 'Peak-TSB', 'BEAP']
                    else:
                        baseline_feat = ['preterm', 'GA', 'bw', 'sex', 'CGA', 'Peak-TSB']
                    X_train, X_test, y_train, y_test =  train_test_split(jhu_roi_outcome_model.loc[:, list(features_combined)+baseline_feat],
                                         jhu_roi_outcome_model.loc[:, outcomes_continue[i]] < cutoff,
                                         stratify=(jhu_roi_outcome_model.loc[:, outcomes_continue[i]] < cutoff),
                                         #stratify=groups*((jhu_roi_outcome_model.loc[:, outcomes_continue[i]] < cutoff)+1),
                                         random_state=seed, test_size=0.3)
                #grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, scoring='r2', n_jobs=30)
                #cv = LeaveOneOut()
                #cv = LeavePOut(10)
                smote = SMOTE(random_state=seed)
                X_train_res, Y_train_res = smote.fit_sample(X_train, y_train.ravel())

                cv = StratifiedKFold(3, True, seed)
                ##cv = RepeatedStratifiedKFold(3, 3, seed)
                if reg:
                    grid = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=500, cv=3, iid=True, n_jobs=30, scoring='r2')
                else:
                    print('weird slow...')
                    #grid = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=1, cv=cv, iid=True, n_jobs=1, scoring='roc_auc')
                    grid = RandomizedSearchCV(pipeline, param_distributions=parameters, n_iter=540, cv=3, iid=True, n_jobs=90, scoring='roc_auc')

                grid.fit(X_train_res, Y_train_res, groups=Y_train_res) # stratified by outcome binary
                X_test_prediction = grid.predict(X_test)
                if not reg:
                    X_test_prob = grid.predict_proba(X_test)[:, 1]
                if reg:
                    spr = stats.spearmanr(X_test_prediction, y_test)
                    model_corr = [spr[0], spr[1], r2_score(y_test, X_test_prediction)]
                else:
                    model_corr = [roc_auc_score(y_test, X_test_prob), average_precision_score(y_test, X_test_prob), f1_score(y_test, X_test_prediction)]

                cvres = grid.cv_results_
                #print(X_test.columns[grid.best_estimator_.named_steps['clf'].importance])
                inc_features = X_test.columns[grid.best_estimator_.named_steps['clf'].coef_[0]>1e-8]
                print(inc_features)
                print(X_test.columns.shape)
                print(np.unique(X_test.columns).shape)
                print(X_test.columns)
                print(X_train.columns)

                best_val = grid.cv_results_['mean_test_score'][grid.cv_results_['params']==grid.best_params_]
                for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
                    if params == grid.best_params_: 
                        #best_score, best_param = np.sqrt(-mean_score), params
                        best_score, best_param = mean_score, params

                out_label = '__'.join(features_combined_labels)
                print([name, n, out_label, best_score, model_corr[0], model_corr[1]])
                if reg:
                    performances_jhu122.append([name, n, out_label, ';'.join(list(inc_features)), best_score, 
                                                model_corr[0], model_corr[1], model_corr[2]])
                else:
                    performances_jhu122.append([name, n, out_label, ';'.join(list(inc_features)), best_score, 
                                                model_corr[0], model_corr[1], model_corr[2], (X_test_prediction==y_test).mean()])
                joblib.dump(grid, f'{name}_{n}_{out_label}_{seed}_{with_BEAP}.pklz')

    if reg:
        jhu_model_results = pd.DataFrame(performances_jhu122, columns=['Method', 'Follow-up', 'Parameter', 'Features', 'Mean Valid ROCAUC', 'Test ROCAUC', 'Test PRAUC', 'F1'])
    else:
        jhu_model_results = pd.DataFrame(performances_jhu122, columns=['Method', 'Follow-up', 'Parameter', 'Features', 'Mean Valid ROCAUC', 'Test ROCAUC', 'Test PRAUC', 'F1', 'Accuracy'])
    #jhu_model_results.to_csv('performances_jhu122_new.csv')
    jhu_model_results.to_csv('performances_jhu122_new_preterm_medium_%s_%s.csv'%(seed, with_BEAP))


def train_voxel_model(jhu_roi_outcome_model, outcomes_continue, seed, native_fs_quality, reg=True, with_BEAP=True):
    scoring = {'precision': 'precision',
               'recall': 'recall',
               'f1': make_scorer(f1_score),
               'auc': make_scorer(roc_auc_score)}
    mask_filename = 'template_mask.nii.gz'
    jhu_roi_outcome_model.loc[pd.isnull(jhu_roi_outcome_model.bw), 'bw'] = np.median(jhu_roi_outcome_model.bw[~pd.isnull(jhu_roi_outcome_model.bw)])
    from nilearn.input_data import NiftiMasker
    from nilearn.plotting import plot_stat_map
    seed = 9999
    cv = RepeatedStratifiedKFold(3, 5, random_state=seed)
    all_perfs = []
    jhu_roi_outcome_model.columns = jhu_roi_outcome_model.columns.map(lambda x:x.replace(' ', '_').replace('-', '_'))

    cog = ((jhu_roi_outcome_model.loc[:, ['Cognitive_12m']] < 85).sum(axis=1) >= 1).astype(int)
    lan = ((jhu_roi_outcome_model.loc[:, ['Receptive_Communication_12m', 'Receptive_Communication_18m', 
                             'Expressive_Communication_12m', 'Expressive_Communication_18m']] < 85).sum(axis=1) >= 1).astype(int)
    mot = ((jhu_roi_outcome_model.loc[:, ['Fine_Motor_12m', 'Fine_Motor_18m', 
                             'Gross_Motor_12m', 'Gross_Motor_18m']] < 85).sum(axis=1) >= 1).astype(int)

    for outcome, name in zip([cog, lan, mot],
                             ['Cognition', 'Language', 'Motor']):
        for modal in ['od', 'md', 'fa', 'reho', 'falff', 'alff']:
            img = f'{modal}_local_template.nii.gz'
            masker = NiftiMasker(mask_img=mask_filename, smoothing_fwhm=6,
                                 standardize=True, memory="nilearn_cache", memory_level=1)
            X = masker.fit_transform(img)
            X = X[native_fs_quality]
            print(X.shape)
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
                ("std", RobustScaler()),
                ("fs",  SelectP(f_classif, alpha=0.01)),
         #       ("smote", SMOTE(random_state=seed)),        
        #        ("fs",  SelectKBest(f_classif, k=122)),
                ("clf", clf)
            ])
            params = {
                'clf__n_estimators': [2, 8, 10],
                'clf__max_depth': [2, 8, 10],
                'clf__reg_lambda': [1e-3, 0.1, 1, 10, 100, 200, 500, 1000, 10000, 1e8]
            }
            
            clf1 = Pipeline([
                ("std", RobustScaler()),
                ("fs",  SelectP(f_classif, alpha=0.01)),
                #("smote", SMOTE(random_state=seed)),
        #        ("fs",  SelectKBest(f_classif, k=122)),
                ("clf", SVC(C=1e8, kernel='linear', probability=True, max_iter=500000, random_state=seed))
            ])    
            params1 = {
                'clf__C': [1e-3, 0.1, 1, 10, 100, 200, 500, 1000, 10000, 1e8]
            }
            
            clf2 = Pipeline([
                ("std", RobustScaler()),
                ("fs",  SelectP(f_classif, alpha=0.01)),
                #("smote", SMOTE(random_state=seed)),        
        #        ("fs",  SelectKBest(f_classif, k=122)), 
                ("clf", LogisticRegression(penalty='l1', solver='liblinear', C=100000000,
                                           max_iter=5000, random_state=seed))
            ])    
            params2 = {
                'clf__C': [1e-3, 0.1, 1, 10, 100, 200, 500, 1000, 10000, 1e8]
            } 
            for m, clf, param in zip(['xgboost', 'svm', 'logit'], [clf, clf1, clf2], [params, params1, params2]):
                model = GridSearchCV(clf,  param_grid=param, scoring=scoring, 
                                     refit='f1', cv=cv, n_jobs=30, return_train_score=True)    
                model.fit(X, outcome, groups=outcome)
                all_perfs.append([m, name, seed, 
                                  model.cv_results_['mean_test_AUC'][model.best_index_],
                                  model.cv_results_['mean_test_recall'][model.best_index_],
                                  model.cv_results_['mean_test_Accuracy'][model.best_index_],
                                  model.cv_results_['mean_test_precision'][model.best_index_],                      
                                  model.cv_results_['mean_test_f1'][model.best_index_]])
    perfs = pd.DataFrame(all_perfs)
    perfs.sort_values(4, ascending=False).to_csv('voxel_predictions.csv')


def main():
    force = True
    outcomes_continue = ['Cognitive_12m', 'Fine Motor_12m',
           'Gross Motor_12m', 'Receptive Communication_12m',
           'Expressive Communication_12m', 'Cognitive_18m', 'Fine Motor_18m',
           'Gross Motor_18m', 'Receptive Communication_18m',
           'Expressive Communication_18m']
    outcomes_binary = ['BEAP']

    jhu_roi_outcome_model, native_fs_quality = get_roi_val(outcomes_continue)
    peaktsb = pd.read_csv('peak_tsb.txt', sep='\t')

    jhu_roi_outcome_model = jhu_roi_outcome_model.merge(peaktsb, left_on='participant_id', right_on='participant_id')
    final = jhu_roi_outcome_model.drop('participant_id', axis=1)

    seed = 9999
    train_voxel_model(final, outcomes_continue, seed, native_fs_quality, reg=False, with_BEAP=False)
    #for seed in range(9000, 10000):
        #train_model(final, outcomes_continue, seed, reg=False, with_BEAP=False)

    # add BAEP
    #for seed in range(9000, 10000):
    #    train_model(final, outcomes_continue, seed, reg=False, with_BEAP=True)

if __name__ == '__main__':
    main()
