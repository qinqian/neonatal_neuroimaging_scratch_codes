#!/usr/bin/env python

from sklearn.metrics import confusion_matrix
import h5py
import numpy as np
from time import time
import mxnet as mx
from mxboard import SummaryWriter
from mxnet import image, nd, sym, init
from mxnet.gluon import data as gdata, loss as gloss, model_zoo as mz, utils as gutils
import os
from mxnet.gluon import Trainer
from mxnet import autograd
from tqdm import tqdm
import logging

from mxmodel import Vgg3d
from transform import get_crop, elastic_transform
import os
mx.random.seed(999)
import pandas as pd
#import SimpleITK
from multiprocessing import cpu_count
import collections
CPU_COUNT = cpu_count()
#import ants
#loader = ants.image_read

def read_image(x):
    import nibabel
    loader = nibabel.load
    image = loader(x)
    return image.get_data()

class MRI2DDataH5(gdata.Dataset):
    """ custom a nii dataset class
    following: http://mxnet.incubator.apache.org/tutorials/python/data_augmentation_with_masks.html
    """
    def __init__(self, image_list, transform=False):
        self.image_list = pd.read_table(image_list)
        self._transform = transform

    def __getitem__(self, idx):
        # crop
        if os.path.exists("%s.h5" % idx) or os.path.exists("valid%s.h5" % idx):
            if self._transform:
                with h5py.File("%s.h5" % idx) as h5:
                    ic = h5['image'][:]
                    label = h5['label'].value
            else:
                with h5py.File("valid%s.h5" % idx) as h5:
                    ic = h5['image'][:]
                    label = h5['label'].value
        # augment
        if self._transform:
            do_flip = np.random.random(1)
            if do_flip > 0.5:
                ic = np.flip(ic, axis=2)
            add_gaussian = np.random.random(1)
            if add_gaussian:
                ic = ic + np.random.normal(0, 0.1, ic.shape)
            #elastic = np.random.random(1)
            #if elastic:
            #    i = elastic_transform(i, alpha=[1, 1e5, 1e5], sigma=[1, 10, 10])

        std = np.std(ic)
        ic = (ic - np.mean(ic)) / std
        return nd.array(ic).astype(np.float32), label, idx

    def __len__(self):
        return self.image_list.shape[0]

class MRI2DDataRaw(gdata.Dataset):
    """ custom a nii dataset class
    following: http://mxnet.incubator.apache.org/tutorials/python/data_augmentation_with_masks.html
    """
    def __init__(self, image_list, transform=False, size=0):
        self.image_list = pd.read_table(image_list)
        if size > 0:
            select = np.random.choice(np.arange(self.image_list.shape[0]), size)
            self.image_list = self.image_list.iloc[select, :]
        self._transform = transform

    def __getitem__(self, idx):
        i = read_image(self.image_list.iloc[idx, 1]) # T2 flair
        ic = np.zeros((3, )+i.shape, dtype=np.float32)
        ic[0, :, :, :] = i

        i0 = read_image(self.image_list.iloc[idx, 0]) # T1
        ic[1, :, :, :] = i0

        #i2 = read_image(self.image_list.iloc[idx, 2]) # ADC
        #ic[2, :, :, :] = i2

        i3 = read_image(self.image_list.iloc[idx, 3]) # DWI
        ic[2, :, :, :] = i3

        #xmin, xmax, ymin, ymax, zmin, zmax = get_crop(i)
        #i = i[xmin:xmax:1, ymin:ymax:1, zmin:zmax:1]
        #i = i[20:100, 20:100, 20:100]
        label = self.image_list.iloc[idx, -1]

        # augment
        if self._transform:
            do_flip = np.random.random(1)
            if do_flip > 0.5:
                ic = np.flip(ic, axis=2)
            add_gaussian = np.random.random(1)
            if add_gaussian:
                ic = ic + np.random.normal(0, 0.1, ic.shape)
            #elastic = np.random.random(1)
            #if elastic:
            #    i = elastic_transform(i, alpha=[1, 1e5, 1e5], sigma=[1, 10, 10])

        std = np.std(ic)
        ic = (ic - np.mean(ic)) / std
        return ic.astype(np.float32), label, idx

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
    return l_sum/l_n, accuracy.get()[1]

def evaluate_multi(data_iter, net, loss, ctx):
    #accuracy = mx.metric.Accuracy()
    #f1 = eval_metrics_2 = mx.metric.F1()
    #eval_metrics = mx.metric.CompositeEvalMetric()
    #for child in [ accuracy, f1 ]:
    #    eval_metrics.add(child)
    eval_metrics = mx.metric.create(['acc', 'f1'])

    l_sum = 0
    l_n = 0
    tn, fp, fn, tp = 0, 0, 0, 0
    for x, y, idx in data_iter:
        gpuy = gutils.split_and_load(y, ctx)
        gpux = gutils.split_and_load(x, ctx)
        output =  [ net(i) for i in gpux ]
        ls = [ loss(o, sub_y) for o, sub_y in zip(output, gpuy) ]
        l_sum += sum([l.sum().asscalar() for l in ls])
        l_n += sum([l.size for l in ls])
        for o, sub_y in zip(output, gpuy):
            eval_metrics.update(preds=o, labels=sub_y)
            confusion = confusion_matrix(sub_y.asnumpy(), o.argmax(axis=1).asnumpy(), labels=[0, 1])
            print(confusion)
            confusion = confusion.ravel()
            tn += confusion[0]
            fp += confusion[1]
            fn += confusion[2]
            tp += confusion[3]
   
    nd.waitall()
    output = eval_metrics.get()
    eval_metrics.reset()
    return tn, fp, fn, tp, l_sum/l_n, output[0], output[1]

def metric_str(acc, names):
    return '.'.join(["%s=%s" % (n, a) for a, n in zip(acc, names)])

def generate_hdf5():
    import h5py
#    image_iter = MRI2DData('xaa', False)
    image_iter = MRI2DDataRaw('xab', False)

    for i, label, idx in image_iter:
        with h5py.File('valid%s.h5' % idx) as fout:
            fout['label'] = label
            fout['image'] = i
    return

def train(batch_size, num_gpus, lr, n_epochs, size=20):
    ctx = [mx.gpu(i) for i in range(num_gpus)]
    # conv_arch = ((1, 64), (1, 128), (2, 256), (2,512), (2, 512))
    conv_arch = ((1, 8), (1, 16))
    net = Vgg3d(conv_arch)
    #schedule = mx.lr_scheduler.FactorScheduler(step=20, factor=0.5)
    #schedule.base_lr = lr

    #schedule = mx.lr_scheduler.PolyScheduler(max_update=400, pwr=2, base_lr=lr)
    #sgd = mx.optimizer.SGD(learning_rate=lr, lr_scheduler=schedule)

    adam = mx.optimizer.Adam(learning_rate=lr)
    net.hybridize()

    #net.initialize()
    #net.forward(mx.nd.ones((1, 3, 117, 159, 126)))

    #net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)

    print(net.collect_params())
    trainer = Trainer(
        #net.collect_params(), 'adam', {'learning_rate': lr})
        net.collect_params(), optimizer=adam)
    loss = gloss.SoftmaxCrossEntropyLoss()
    image_iter = MRI2DDataRaw('xaa', True, size=20)
    image_iter2 = MRI2DDataRaw('xab', False)

    train_loader = gdata.DataLoader(image_iter, shuffle=True, num_workers=30, batch_size=batch_size, last_batch='rollover')
    test_loader = gdata.DataLoader(image_iter2, shuffle=False, num_workers=30, batch_size=batch_size, last_batch='rollover')

    start = time()
    bestacc = 0.5
    bestloss = 10
    original_patience = 3
    patience = 3

    accuracy = mx.metric.Accuracy()
    trainls = []
    validls = []
    with SummaryWriter(logdir='./mxlogs') as sw:
        for n in range(n_epochs):
            if n==1:
                sw.add_graph(net)

            train_loss = 0
            train_size = 0
            for x, y, idx in train_loader:
                gpuy = gutils.split_and_load(y, ctx)
                gpux = gutils.split_and_load(x, ctx)
                # print(len(gpux))
                with autograd.record():
                    os = [ net(sub_x) for sub_x in gpux ]
                    ls = [ loss(o, sub_y) for o, sub_y in zip(os, gpuy) ]
                for l in ls:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in ls])
                train_size += sum([l.size for l in ls])
                [ accuracy.update(preds=o, labels=sub_y) for o, sub_y in zip(os, gpuy) ]

                nd.waitall()
            tn, fp, fn, tp, valloss, names, valacc = evaluate_multi(test_loader, net, loss, ctx)
            trainls.append(train_loss/train_size)
            validls.append(valloss)
            
            print(valacc, names)
            sw.add_scalar(tag='true_negative', value=tn, global_step=n)
            sw.add_scalar(tag='false_positive', value=fp, global_step=n)
            sw.add_scalar(tag='false_negative', value=fn, global_step=n)
            sw.add_scalar(tag='true_positive', value=tp, global_step=n)

            sw.add_scalar(tag='validation_loss', value=valloss, global_step=n)
            sw.add_scalar(tag='validation_accuracy', value=valacc[0], global_step=n)
            sw.add_scalar(tag='validation_f1', value=valacc[1], global_step=n)
            sw.add_scalar(tag='train_loss', value=train_loss/train_size, global_step=n)
            sw.add_scalar(tag='train_accuracy', value=accuracy.get()[1], global_step=n)

            print("%s Epoch " % n, 'time %.1f sec' %(time()-start), train_loss/train_size, trainer.learning_rate, accuracy.get()[1], metric_str(valacc, names), valloss)
            accuracy.reset()

            if valloss >= bestloss:
                patience = patience - 1
                if patience <= 0:
                    if lr >= 1e-8:
                        lr = lr / 2
                        print("setting learning rate %s" % lr)
                    trainer.set_learning_rate(lr)
                    patience = original_patience
            else:
                bestloss = valloss
                if valacc[0] - bestacc >= 1e-4:
                    bestacc = valacc[0]
                    sw.add_scalar(tag='check_point', value=n, global_step=n)
                    net.save_parameters('vgg_%s.params' % n)
                patience = original_patience

            sw.add_scalar(tag='learning_rate', value=lr, global_step=n)

    df = pd.DataFrame(train_error=trainls, valid_error = validls)
    df.to_csv('%s_error.txt' % size)

def preprocess():
    return

def main():
    #os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    #os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
    #generate_hdf5()
    train(9, 3, 0.00002, 80, 102)

if __name__ == '__main__':
    main()
