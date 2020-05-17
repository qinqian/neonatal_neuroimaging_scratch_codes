#!/usr/bin/env python
import pandas as pd
import numpy as np
import glob 

def plot_stat(model, seed, loss, patch):
    alls = []
    for i in ['train', 'valid', 'test']:
        es, ds, rs = [], [], []
        #for j in glob.glob('%s_%s%s%s_*' % (loss, i, model, seed)):
        print('process/npy/%s*_%s%s%s_*_%s*npy' % (loss, i, model, seed, patch))
        for j in glob.glob('process/npy/%s*%s%s%s*_%s*npy' % (loss, i, model, seed, patch)):
            print(j)
            e = int(j.split('_')[8].replace('epoch',''))
            ##e = int(j.split('_')[6].replace('epoch',''))
            es.append(e)
            metric = np.load(j)
            ds.append(np.mean(metric))
            rs.append((metric>0).mean())
        df = pd.DataFrame({'dataset': i, 'epoch': es, 'dice_score': ds, 'detection_rate': rs})
        df.sort_values(by=['epoch'], inplace=True)
        alls.append(df)
        print(df)
    df = pd.concat(alls, axis=0)
    return df

print('999')
#plot_stat('_edd_', '999', 'focal')
#plot_stat('_edd_', '999', 'bce_', 32)
df = plot_stat('_edd_', '999', 'bce_', 64)
df.to_csv('64_999_patch.csv')
#plot_stat('_edd_', '999', 'bce_', 96)
#plot_stat('_edd_', '999', '')

