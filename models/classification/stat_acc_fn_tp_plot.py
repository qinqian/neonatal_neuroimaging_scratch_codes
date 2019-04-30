#!/usr/bin/env python
import numpy as np
import glob
import seaborn as sns
import pandas as pd

def get_stat(model):
    result = []
    #for t in ['train', 'valid', 'test']:
    for t in ['test']:
    #for t in ['']:
        for i in ['0_1_2_3_4', '0_1_2_3', '0_1_2', '0_1', '0']:
            #for j in glob.glob('model_store.bak/%s*modal%s_lr0.0002*lossbce.npy' % (model, i)):
            #vgg_gpu2_batch16_label1_modal0_1_2_3_lr0.0001_lossbce_seed99.npy
            #for j in glob.glob('model_store/%s*modal%s_lr0.0001_lossbce_seed99.npy' % (model, i)):
            #for j in glob.glob('model_store/%s_gpu*_batch32*modal%s_lr0.0002_lossbce_seed99_%s.npy' % (model, i, t)):
            for j in glob.glob('model_store/%s_gpu*_batch32*modal%s_lr0.0001_lossbce_seed99_%s.npy' % (model, i, t)):
            #for j in glob.glob('model_store.bak/%s_gpu*_batch32*modal%s_lr0.0001_lossbce_seed99%s.npy' % (model, i, t)):
            #for j in glob.glob('model_store/%s_gpu*_batch32*modal%s_lr0.0001_lossbce_seed99%s.npy' % (model, i, t)):
                print(j.replace('.npy', ''))
            #for j in glob.glob('model_store/%s_gpu*_batch16*modal%s_lr0.0001_lossbce_seed99.npy' % (model, i)):
                stat = np.load(j)
                recall = stat[1,1]/np.sum(stat[1,:])
                precision = stat[1,1]/np.sum(stat[:, 1])
                result.append([model, t, i, (stat[0,0]+stat[1,1])/stat.sum(), (stat[1,0]/np.sum(stat[1,:])), (stat[0,1]/np.sum(stat[:,1])), recall, precision, recall*precision*2/(recall+precision), stat[0,0]/np.sum(stat[0,:])])
        cols = ('model', 'dataset', 'modal', 'accuracy', 'false negative', 'false positive', 'recall', 'precision', 'f1', 'specificity')
    return pd.DataFrame(result, columns=cols)


vgg = get_stat('vgg')
inception = get_stat('inception')
final = pd.concat([vgg, inception], axis=0)
print(final)
final.to_csv('results/table1_3d_metrics.csv')
print(final.f1.mean())
#sns.factorplot(data=final)
