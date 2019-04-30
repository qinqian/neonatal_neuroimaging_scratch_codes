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

    def __getitem__(self, idx):
        image = read_image(self.image_list.iloc[idx, 0])
        mask = read_image(self.image_list.iloc[idx, -1])

        image = image/np.max(image)*255
        mask = mask/np.max(mask)*255
        mask_slice, = np.where(mask.sum(axis=0).sum(axis=0) > 0) # depth only, which slice has mask

        ## random select equal number of image
        nonmask_slice, = np.where(mask.sum(axis=0).sum(axis=0) == 0) # depth only, which slice has mask
        nonmask = np.random.choice(nonmask_slice, size=len(mask_slice))

        return image[:, :, mask_slice], mask[:, :, mask_slice], image[:, :, nonmask], mask[:, :, nonmask]

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
    mask = pd.read_table('mask/all/1-3.txt', header=None)
    print(mask)
    df = df.loc[df.ids.isin(mask.iloc[:, 0]),:]
    df.loc[:, 'mask'] = df.ids.map(lambda x: 'mask/all/qq_labels/1-3/%s.nii.gz' % str(x))

    print(df.head())
    print(df.shape)
    data = MRWithMask(df, False)
    nslices = 0
    n = 0
    for i in data:
        print(i[0].shape, i[1].shape)
        nslices += i[0].shape[2]
        for j in range(i[0].shape[2]):
            cv2.imwrite("2d_pvls/%s_%s.jpg" % (n, j), np.expand_dims(i[0][:, :, j], axis=2))
            cv2.imwrite("2d_pvls/%s_%s_mask.jpg" % (n, j), np.expand_dims(i[1][:, :, j], axis=2))

            cv2.imwrite("2d_pvls/%s_%s_nolesion.jpg" % (n, j), np.expand_dims(i[2][:, :, j], axis=2))
            cv2.imwrite("2d_pvls/%s_%s_nonmask.jpg" % (n, j), np.expand_dims(i[3][:, :, j], axis=2))
            print(j)
        n += 1

    print(nslices)
    return df

def main():
    meta = get_meta()
    print(meta.head())

if __name__ == '__main__':
    main()
