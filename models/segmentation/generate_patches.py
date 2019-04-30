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


