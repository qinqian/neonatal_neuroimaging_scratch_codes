import pandas as pd
import numpy as np
import os
import glob 
def get_age(x):
    if 'D' in x:
        return float(str(x).replace('D', ''))/7
    if 'W' in x:
        return float(str(x).replace('W', ''))
    if 'M' in x:
        return float(str(x).replace('M', '')) * 4

fs = glob.glob('/data1/qinqian/MRI/Genetics/phaseB_get_yanglin_positive_ones/yanglin_genetics/transform/*/*Warp_jac.nii.gz')
fs = dict(zip(map(lambda i: int(os.path.basename(i).replace('1Warp_jac.nii.gz', '')), fs), fs))

fs2 = glob.glob('/data1/qinqian/MRI/Genetics/phaseB_get_yanglin_positive_ones/yanglin_genetics/applytransformimage/*/*2/*run-02_dwi*.nii.gz')
fs2 = dict(zip(map(lambda i: int(os.path.basename(i).split('_')[0].replace('sub-', '')), fs2), fs2))

exclude = pd.read_table('exclude_ids.txt', delimiter=' ', header=None)

yanglin_age = pd.read_table('source/participants.tsv')
yanglin_age.iloc[:, 0] = yanglin_age.iloc[:, 0].map(lambda x:x.replace('sub-', ''))

fs_dict = {}
fs_dict2 = {}
print(fs)
for i in range(yanglin_age.shape[0]):
    if i+1 in fs:
        fs_dict[yanglin_age.iloc[i, 0]] = fs[i+1]
    else:
        fs_dict[yanglin_age.iloc[i, 0]] = np.nan
    if i+1 in fs2:
        fs_dict2[yanglin_age.iloc[i, 0]] = fs2[i+1]
    else:
        fs_dict2[yanglin_age.iloc[i, 0]] = np.nan

yanglin_list = pd.read_table('yanglin.list')
yanglin = yanglin_age.merge(yanglin_list, left_on='participant_id', right_on='participant_id')

df = pd.read_table('yanglin_final.txt')
df.iloc[:, -1] = df.iloc[:, -1].map(lambda x: os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(x)))))
df = df.merge(yanglin, left_on = 'directory', right_on = 'participant_id')

exclude_yanglin = exclude.loc[exclude.iloc[:, 0].str.contains('yang'), 1]-1
df = df.loc[(~(df.loc[:, 'directory'].isin(yanglin_age.iloc[exclude_yanglin, 0]))), :]

df.loc[:, 'jac'] = df.loc[:, 'directory'].map(lambda x: fs_dict[x])
df.loc[:, 'adc'] = df.loc[:, 'directory'].map(lambda x: fs_dict2[x])
df = df.loc[~pd.isnull(df.loc[:, 'jac']), :]
df.loc[:, 'cga'] = df.ga + df.age.map(get_age)
print(df.age)
for i in df.age:
    print(i)
df.drop(['ID_y', 'Gender', 'participant_id', 'group', 'ID_x', 'DISK', 'original_ga', 'directory'], axis=1, inplace=True)
print(df.shape)
df.to_csv('yanglin_final_after_exclude.txt', sep='\t')
