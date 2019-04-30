#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
from glob import glob

fs = glob('/data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/control/transform/*/*1Warp_jac.nii.gz')
fs = dict(zip(map(lambda i: os.path.basename(os.path.dirname(i)), fs), fs))
fs2 = glob('/data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/control/applytransformimage/*/*2/*run-02_dwi*.nii.gz')
fs2 = dict(zip(map(lambda i: os.path.basename(os.path.dirname(os.path.dirname(i))), fs2), fs2))
print(len(fs.keys()))
print(len(fs2.keys()))

def get_age(x):
    if 'D' in x:
        return float(str(x).replace('D', ''))/7
    if 'W' in x:
        return float(str(x).replace('W', ''))
    if 'M' in x:
        return float(str(x).replace('M', '')) * 4

df = pd.read_excel('pvl_final_control_patient.xlsx')
df.loc[:, 'participant_id'] = df.loc[:, 'participant_id'].map(str)

cont = pd.read_table('/data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/control_evidence_with_no_significant_anamoly.txt')
source = pd.read_table('/data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/source/participants.tsv')
source.loc[:, 'participant_id'] = source.loc[:, 'participant_id'].map(lambda x:x.replace('sub-', ''))
source = source.merge(cont, on='participant_id')
print(source.head())

file_id = source.apply(lambda x: "%s_%s_%s" % (x[4], x[3], x[0]), axis=1)

print(len(file_id))
print(len(fs2.keys()))
print(len(np.intersect1d(np.unique(file_id), np.unique(list(fs2.keys())))))

#for i in file_id:
#    print(i)
source.loc[:, 'adc']  = file_id.map(lambda x: fs2[x] if x in fs else np.nan)
source.loc[:, 'jac']  = file_id.map(lambda x: fs[x] if x in fs else np.nan)
source = source.loc[~pd.isnull(source.loc[:, 'adc']), :]
print(source.shape)

source = source.merge(df, on='participant_id')
source.loc[:, 'age'] = source.loc[:, 'age'].map(get_age)
source.loc[:, '胎龄计算'] = source.loc[:, '胎龄计算'].map(lambda x: np.nan if isinstance(x, str) else float(x))

source = source.loc[~pd.isnull(source.loc[:, '胎龄计算']), :]
source.loc[:, 'cga'] = source.loc[:, 'age'] + source.loc[:, '胎龄计算']
print(source.shape)
source = source.loc[source.loc[:, 'include'] == 1, :]
source.to_csv('control_merge_info_file.txt', sep='\t')
print(source.shape)

