#!/usr/bin/env python

import numpy as np
import cv2
from time import time
import mxnet as mx
from mxnet import image, nd, sym, init
from mxnet.gluon import data as gdata, loss as gloss, model_zoo as mz, utils as gutils
from mxnet.gluon import Trainer
from mxnet import autograd
from model import Vgg3d
from transform import get_crop, elastic_transform
import os
mx.random.seed(999)
import pandas as pd
import SimpleITK
from multiprocessing import cpu_count
import collections
import tqdm
np.random.seed(999)

CPU_COUNT = cpu_count()

normalize = gdata.vision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.RandomResizedCrop(224),
                                              gdata.vision.transforms.RandomFlipLeftRight(),
                                              gdata.vision.transforms.ToTensor(), normalize])

test_augs = gdata.vision.transforms.Compose([gdata.vision.transforms.Resize(256),
                                            gdata.vision.transforms.CenterCrop(224),
                                            gdata.vision.transforms.ToTensor(),
                                            normalize])

def read_image(x):
    import nibabel
    loader = nibabel.load
    image = loader(x)
    return image.get_data()

class MRWithMask(gdata.Dataset):
    def __init__(self, df, aug):
        self.image_list = df
        print(self.image_list.shape)

    def __getitem__(self, idx):
        flair = read_image(self.image_list.loc[:, 'flair'].values[idx])
        dwi = read_image(self.image_list.loc[:, 'dwi'].values[idx])
        t1 = read_image(self.image_list.loc[:, 't1'].values[idx])
        mask = read_image(self.image_list.loc[:, 'label'].values[idx])
        mask2 = read_image(self.image_list.loc[:, 'label2'].values[idx])

        mask = mask + mask2
        mask[mask>0] = 1
        mask_slice, = np.where(mask.sum(axis=0).sum(axis=0) > 0) # depth only, which slice has mask

        ## random select equal number of image
        nonmask_slice, = np.where(mask.sum(axis=0).sum(axis=0) == 0) # depth only, which slice has mask
        nonmask = np.random.choice(nonmask_slice, size=len(mask_slice))

        image3chan = np.stack([flair[:, :, mask_slice], dwi[:, :, mask_slice], t1[:, :, mask_slice]], axis=-1)
        labelmask = mask[:, :, mask_slice]

        nonmaskimage = np.stack([flair[:, :, nonmask], dwi[:, :, nonmask], t1[:, :, nonmask]], axis=-1)

        nonmask = mask[:, :, nonmask]
        return idx, image3chan, labelmask, nonmaskimage, nonmask

    def __len__(self):
        return self.image_list.shape[0]


def pretrained_model():
    pretrain = mz.vision.densenet201(pretrained=True)
    return pretrain


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate(data_iter, net, loss):
    accuracy = mx.metric.Accuracy()
    acc = 0
    l_sum = 0
    l_n = 0
    for x, y in data_iter:
        output = net(x.as_in_context(mx.gpu(0)))
        l = loss(output, y.as_in_context(mx.gpu(0)))
        l_sum += l.sum().asscalar()
        l_n += l.size
        accuracy.update(preds=output, labels=y)
    return l_sum / l_n, accuracy.get()[1]


def evaluate_multi(data_iter, net, loss, ctx):
    accuracy = mx.metric.Accuracy()
    acc = 0
    l_sum = 0
    l_n = 0
    for x, y in data_iter:
        gpuy = gutils.split_and_load(y, ctx)
        gpux = gutils.split_and_load(x, ctx)
        output = [net(i) for i in gpux]
        ls = [loss(o, sub_y) for o, sub_y in zip(output, gpuy)]
        l_sum += sum([l.sum().asscalar() for l in ls])
        l_n += sum([l.size for l in ls])
        for o, sub_y in zip(output, gpuy):
            # pred = nd.zeros(sub_y.shape)
            # pred[o[:,2] > 0.4] = 1
            accuracy.update(preds=o, labels=sub_y)
    nd.waitall()
    return l_sum / l_n, accuracy.get()[1]

def train(batch_size, num_gpus, lr, n_epochs):
    ctx = [mx.gpu(i) for i in range(num_gpus)]

def get_meta():
    df = pd.read_table('all_original_flair.txt', header=None)
    df.loc[:, 'ids'] = df.iloc[:, 0].map(lambda x: int(os.path.basename(x).replace('_run-01_T2w_corrected.nii.gz','').replace('sub-', '')))
    df.loc[:, 'flair'] = df.iloc[:, 0].map(lambda x: '../%s' % x)

    df.loc[:, 't1'] = df.apply(lambda x: os.path.join(os.path.dirname(x[2]).replace('flair', 'align'), '_intra_align0/sub-%s_run-01_T1w_corrected_flirt.nii.gz' % str(x[1])), axis=1)
    df.loc[:, 'dwi'] = df.apply(lambda x: os.path.join(os.path.dirname(x[2]).replace('flair', 'align'), '_intra_align3/sub-%s_run-01_dwi_roi_corrected_flirt.nii.gz' % str(x[1])), axis=1)

    mask = pd.read_table('mask/all/1-3.txt', header=None)
    df = df.loc[df.ids.isin(mask.iloc[:, 0]),:]
    df.loc[:, 'label'] = df.ids.map(lambda x: 'mask/all/qq_labels/1-3/%s.nii.gz' % str(x))
    df.loc[:, 'label2'] = df.ids.map(lambda x: 'mask/all/chen_labels/1-3/%s.nii.gz' % str(x))
    df.to_csv('data_3channel_with_mask.xls')

    data = MRWithMask(df, False)
    print(len(data))
    nslices = 0
    n = 0
    for i in data:
        nslices += i[1].shape[2]

        #norm_image = i[1]/np.max(i[1])*255  # normalize by whole brain
        for j in range(i[1].shape[2]):
            image = i[1][:, :, j,:]
            #image = norm_image[:, :, j,:]
            image = image/np.max(image) * 255 ## this is work...surprise, normalize by image slice only

            #nonmaskimage = i[3][:, :, j,:]
            label = i[2][:, :, j]
            #nonmasklabel = i[4][:, :, j]
            #print(np.unique(label))
            #label = np.expand_dims(label, axis=2)
            cv2.imwrite("2d_pvls/%s_%s_image0.jpg" % (n, j), image[:,:,0])
            cv2.imwrite("2d_pvls/%s_%s_image1.jpg" % (n, j), image[:,:,1])
            cv2.imwrite("2d_pvls/%s_%s_image2.jpg" % (n, j), image[:,:,2])
            cv2.imwrite("2d_pvls/%s_%s_image.jpg" % (n, j), image)
            cv2.imwrite("2d_pvls/%s_%s_label.jpg" % (n, j), label*255)
            np.save("2d_pvls/%s_%s_image.npy" % (n, j), image)
            np.save("2d_pvls/%s_%s_label.npy" % (n, j), label)

            # nonmask * 11 + 1 to differ the names
            # np.save("2d_pvls/%s_%s_image.npy" % (n, j*11+1), nonmaskimage)
            # np.save("2d_pvls/%s_%s_label.npy" % (n, j*11+1), nonmasklabel)
        n += 1
    print(n, nslices)
    return df

def main():
    meta = get_meta()

if __name__ == '__main__':
    main()

