import gc
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import random
from numpy.random import choice
import nibabel as nib
from keras_deeplabv3plus import Deeplabv3
import cv2

import numpy as np
import argparse
from sklearn.model_selection import KFold, train_test_split

from unet import densenet_unet, edd_net
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.python.keras.utils import multi_gpu_model, Sequence
import tensorflow as tf
import imgaug
import pandas as pd
from glob import glob
from tqdm import tqdm
from tensorflow.python.client import device_lib
from transforms import aug_mega_hardcore
from skimage import measure
from prediction_valid import predict_whole_brain, dice_ratio
from losses import double_head_loss, sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1, dice_coef_loss, focal_loss, binary_crossentropy, pixwise_crossentropy

def load_2d_image(image_path, label_path, crop_shape, modal, train=True):
    if train:
        x, y = center_on_lesion_crop(np.load(image_path), np.load(label_path), crop_shape, modal)
    else:
        x, y = center_on_lesion_crop(np.load(image_path), np.load(label_path), crop_shape, modal)

    #if train:
    #trans = aug_mega_hardcore() ## rotation, elastic, affine, flip
    #data = trans(image=x, mask=y)
    #x, y = data['image'], data['mask']

    #x = (x - np.mean(x, axis=(0,1)))
    x = x / 127.5 - 1 ## !!! should be the same as the prediction
    #print(image_path)
    #print(x.shape)
    #print(label_path)
    #print(y.shape)
    return x, y

def random_crop(x, y, random_crop_size, **kwargs):
    x0 = 16
    y0 = 16
    if x.shape[0] < random_crop_size:
        x0 += random_crop_size-x.shape[0]
    if x.shape[1] < random_crop_size:
        y0 += random_crop_size-x.shape[1]

    x = np.pad(x, ((x0, x0), (y0, y0), (0, 0)), mode='symmetric')
    y = np.pad(y, ((x0, x0), (y0, y0)), mode='symmetric')

    randw = np.random.randint(0, x.shape[0]-random_crop_size)
    randh = np.random.randint(0, x.shape[1]-random_crop_size)

    x = x[randw:(randw+random_crop_size), randh:(randh+random_crop_size)]
    y = y[randw:(randw+random_crop_size), randh:(randh+random_crop_size)]

    label = np.zeros((y.shape[0], y.shape[1], 2)) ## sigmoid ones
    label[:, :, 0][y==0] = 1
    label[:, :, 1][y>0] = 1
    #label = label.astype('float')
    #x = x.astype('uint8')
    return x, label

def center_on_lesion_crop(x, y, center_crop_size, modal, **kwargs):
    """
    """
    if np.sum(y) > 0:
        labelimg = measure.label(y, return_num=True, connectivity=2)
        props = measure.regionprops(labelimg[0])

        centers = np.array([p.centroid for p in props]).astype('int')
        areas = np.array([p.area for p in props])
        #select = choice(np.arange(len(centers)), 1, p=areas/np.sum(areas))[0] # not reasonable

        #new_centers = []
        #for i, j in zip(centers, areas):
        #    if j > 5: ## exclude two small areas
        #        new_centers.append(i)
        #select = choice(np.arange(len(new_centers)), 1)[0]
        select = choice(np.arange(len(centers)), 1)[0]
        center = centers[select]

        shiftw = center_crop_size // 2
        shifth = center_crop_size // 2
    
        if np.random.uniform(0)<0.5: ## random shift across lesions
            centerw = center[0] + np.random.randint(-shiftw, shiftw)
            centerh = center[1] + np.random.randint(-shifth, shifth)
        else:
            centerw = center[0] 
            centerh = center[1]

        halfw, halfh = center_crop_size // 2, center_crop_size // 2

        if centerw + halfw >= x.shape[0]:
            pad = centerw + halfw - x.shape[0] + 1
            x = np.pad(x, ((0, pad), (0, 0), (0, 0)), mode='symmetric')
            y = np.pad(y, ((0, pad), (0, 0)), mode='symmetric')

        if centerh + halfh >= x.shape[1]:
            pad = centerh + halfh - x.shape[1] + 1
            x = np.pad(x, ((0, 0), (0, pad), (0, 0)), mode='symmetric')
            y = np.pad(y, ((0, 0), (0, pad)), mode='symmetric')

        if centerh - halfh <= 0:
            pad = halfh - centerh + 1
            centerh += pad
            x = np.pad(x, ((0, 0), (pad, 0), (0, 0)), mode='symmetric')
            y = np.pad(y, ((0, 0), (pad, 0)), mode='symmetric')

        if centerw - halfw <= 0:
            pad = halfw - centerw + 1
            centerw += pad
            x = np.pad(x, ((pad, 0), (0, 0), (0, 0)), mode='symmetric')
            y = np.pad(y, ((pad, 0), (0, 0)), mode='symmetric')

        label = np.zeros((y.shape[0], y.shape[1], 2)) ## sigmoid ones
        label[:, :, 0][y==0] = 1
        label[:, :, 1][y==1] = 1

        if modal == 3: # all three modal
            x = x[(centerw - halfw):(centerw + halfw), (centerh - halfh):(centerh + halfh), :]
        elif modal == 2:
            x = x[(centerw - halfw):(centerw + halfw), (centerh - halfh):(centerh + halfh), [0, 2]]
        else:
            x = x[(centerw - halfw):(centerw + halfw), (centerh - halfh):(centerh + halfh), 0]
            ## only flair
            x = np.expand_dims(x, axis=-1)

        label = label[(centerw - halfw):(centerw + halfw), (centerh - halfh):(centerh + halfh), :]
        #import time
        #start = time.time()
        ###cv2.imwrite('test%s.jpg' % start, x)
        ###cv2.imwrite('label%s.jpg' % start, np.expand_dims(y*255, axis=-1))
        #fig = plt.figure(figsize=(5, 5))
        #plt.imshow(np.rot90(x[:, :, 0]), interpolation='nearest', cmap=plt.cm.gray)
        #contours0 = measure.find_contours(np.rot90(label[:, :, 1]>0), 0.8)
        #for _, contour in enumerate(contours0):
        #    plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
        #plt.savefig('test%s.jpg' % start)
        return x, label

    return random_crop(x, y, center_crop_size)


def center_crop(x, y, center_crop_size, **kwargs):
    """
    https://github.com/keras-team/keras/issues/3338
    :param x:
    :param center_crop_size:
    :param kwargs:
    :return:
    """
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    if centerw < halfw:
        x = np.pad(x, ((0, halfw - centerw), (0, 0), (0, 0)), mode='symmetric')
        y = np.pad(y, ((0, halfw - centerw), (0, 0), (0, 0)), mode='symmetric')
    if centerh < halfh:
        x = np.pad(x, ((0, 0), (0, halfh-centerh), (0, 0)), mode='symmetric')
        y = np.pad(y, ((0, 0), (0, halfh-centerh), (0, 0)), mode='symmetric')

    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2

    label = np.zeros((y.shape[0], y.shape[1], 2)) ## sigmoid ones

    label[:, :, 0][y[:, :] == 0] = 1
    label[:, :, 1][y[:, :] > 0] = 1
    #label = label.astype('float')
    #x = x.astype('uint8')
    return x[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh, :], \
           label[centerw - halfw:centerw + halfw, centerh - halfh:centerh + halfh, :]

def unfreeze_model(model):
    for l in model.layers:
        l.trainable = True

def freeze_model(model, freeze_before_layer):
    if freeze_before_layer == "ALL":
        for l in model.layers:
            l.trainable = False
    else:
        freeze_before_layer_index = -1
        for i, l in enumerate(model.layers):
            if l.name == freeze_before_layer:
                freeze_before_layer_index = i
        for l in model.layers[:freeze_before_layer_index + 1]:
            l.trainable = False

class ImageSequence(Sequence):
    def __init__(self, df, batch, crop_shape, modality=3, train=True):
        self.df = df
        self.train = train
        self.batch = batch
        self.crop_shape = crop_shape
        self.modal = modality
        self.images = []
        self.labels = []
        for i in range(self.df.shape[0]):
            images = np.array(glob('2d_pvls/%s_*_image.npy' % i))
            images = list(map(lambda x: os.path.basename(x).split('_')[1], images))
            for j in images:
                self.images.append('2d_pvls/%s_%s_image.npy' % (i, j))
                self.labels.append('2d_pvls/%s_%s_label.npy' % (i, j))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return int(len(self.images)/float(self.batch))

    def get_batch(self, idx):
        images, labels = [], []
        for i in range(idx * self.batch, (idx+1) * self.batch):
            #subject_slices = np.arange(len(self.data_list[idx]['image']))
            #j = random.choice(subject_slices)
            #x, y = load_2d_image(self.data_list[idx]['image'][j], self.data_list[idx]['label'][j], self.crop_shape, self.train)
            x, y = load_2d_image(self.images[i], self.labels[i], self.crop_shape[0], self.modal, self.train)
            images.append(x)
            labels.append(y)
        #print(images[0].shape)
        #print(labels[0].shape)
        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0)
        #images = np.array(images)
        #labels = np.array(labels)
        return images, labels

    def __getitem__(self, idx):
        return self.get_batch(idx)

    def on_epoch_end(self):
        n = np.arange(len(self.images))
        n = np.random.permutation(n)
        self.images = self.images[n]
        self.labels = self.labels[n]


class ModelCheckMGPU(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-gpu')
    p.add_argument('-batch', default=8, type=int)
    p.add_argument('--model_dir', default='model_dir')
    p.add_argument('--model', default='dense')
    p.add_argument('--shape', default=64, type=int)
    p.add_argument('--seed', default=999, type=int)
    p.add_argument('--lr', default=2e-4, type=float)
    p.add_argument('--decay', default=1e-2, type=float)
    p.add_argument('--modality', default=3, type=float)
    p.add_argument('--opt', default='adam')
    p.add_argument('--schedule', default=None)
    p.add_argument('--loss')
    p.add_argument('-resume', default=False, type=bool, help='resume traing from the best checkpoint') ## important TODO list:::
    args = p.parse_args()

    np.random.seed(int(args.seed))
    random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))
    #tf.set_random_seed(int(args.seed))

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    batch = int(args.batch)
    initial_lr = float(args.lr)

    model_dir = os.path.join('models_store', args.model_dir)
    os.system('mkdir -p %s' % model_dir)
    tb_dir = os.path.join('tb_store', args.model_dir)
    os.system('mkdir -p %s' % tb_dir)

    if args.loss == 'bce':
        loss = binary_crossentropy
        #loss = pixwise_crossentropy
    if args.loss == 'focal':
        loss = focal_loss(2, 0.3)
    if args.loss == 'double':
        loss = double_head_loss
    if args.loss == 'dice_loss':
        loss = dice_coef_loss

    shape = int(args.shape)
    shape = (shape, shape, int(args.modality))
    if args.model == 'dense':
        model = densenet_unet(2, shape)
    elif args.model == 'deeplab':
        model = Deeplabv3(input_shape=shape, classes=21, backbone='xception', OS=8)
    else:
        model = edd_net(2, shape)

    #freeze_model(model, 'bn')
    #freeze_model(model, 'input_1')
    print(model.summary())

    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

    #df = pd.read_table('data_3channel_with_mask.xls')
    df = pd.read_csv('data_3channel_with_mask.xls')
    train, test = train_test_split(np.arange(df.shape[0]), test_size=0.2, random_state=args.seed) # for validation
    train, valid = train_test_split(train, test_size=0.2, random_state=args.seed) # for validation
    df.iloc[train, :].to_csv('train_%s.csv' % args.seed)
    df.iloc[valid, :].to_csv('valid_%s.csv' % args.seed)
    df.iloc[test, :].to_csv('test_%s.csv' % args.seed)

    train_data = ImageSequence(df.iloc[train, :], batch, shape[:2], int(args.modality), True)
    val_data = ImageSequence(df.iloc[valid, :], batch, shape[:2], int(args.modality), False)

    def schedule_steps(epoch, steps):
        for step in steps:
            if step[1] > epoch:
                return step[0]
        return 1e-7

    def step_decay(epoch):
        drop_every = 1
        print(initial_lr)
        decay_rate = 1.*initial_lr*(1-args.decay)**(epoch//drop_every)
        print(decay_rate)
        return decay_rate

    # Kfold
    best_model_path = '%s/best_%s_{epoch:003d}_{val_loss:.5f}_{val_hard_dice_coef_ch1:.5f}.h5' % (model_dir, args.model)
    #last_model_path = 'models/last_{epoch:02d}_{val_loss:.2f}.h5'
    #escallback = tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', mode='min')
    if args.schedule == 'fluc':
        lrcallbacks = tf.keras.callbacks.LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(2e-4, 2), (3e-4, 4), (1e-4, 6), (5e-6, 8), (2e-4, 10), (1e-4, 40), (2e-5, 65), (1e-5, 70), (1e-6, 80)]))
    if args.schedule == 'plat':
        lrcallbacks = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    if args.schedule == 'decay':
        print(args.schedule)
        lrcallbacks = tf.keras.callbacks.LearningRateScheduler(step_decay)

    #best_model_check = ModelCheckMGPU(model, best_model_path, monitor='val_loss', verbose=1,
    #                                  mode='min', period=1, save_best_only=True, save_weights_only=True),
    #model = multi_gpu_model(model, len(gpus))

    opt = args.opt
    if opt == 'adam':
        adam = tf.keras.optimizers.Adam(lr=initial_lr)
    if opt == 'ams':
        adam = tf.keras.optimizers.Adam(lr=initial_lr, amsgrad=True)
    if opt == 'rms':
        #adam = tf.keras.optimizers.RMSprop(lr=initial_lr, decay=float(args.decay))
        adam = tf.keras.optimizers.RMSprop(lr=initial_lr)
    if opt == 'sgd':
        #adam = tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True, decay=float(args.decay))
        adam = tf.keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True)

    print('test')
    best_model_check = tf.keras.callbacks.ModelCheckpoint(best_model_path, mode='max', period=1, monitor='val_hard_dice_coef_ch1',
                                                          save_best_only=True, save_weights_only=True)
    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)

    ## https://github.com/bonlime/keras-deeplab-v3-plus/issues/10
    model.compile(loss = loss, optimizer=adam,
                  metrics=[sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1, tf.keras.metrics.binary_crossentropy])

    def eval_whole_brain(index, modal):
        whole_brain = df.iloc[index, :]
        dices = []
        for i in range(whole_brain.shape[0]):
            print(i)
            niis = [df.loc[:, 'flair'][i], df.loc[:, 'dwi'][i], df.loc[:, 't1'][i]]
            if modal == 3:
                niis = niis
            elif modal == 2:
                niis = [niis[0], niis[2]]
            elif modal == 1:
                niis = [niis[0]]

            real_label1 = nib.load(df.loc[:, 'label'][i]).get_data()
            real_label2 = nib.load(df.loc[:, 'label2'][i]).get_data()
            real_label = real_label1 + real_label2
            real_label[real_label>0] = 1
            output = predict_whole_brain(model, niis, shape[0], real_label, npy=True, prob=False, batch=int(args.batch))
            score = dice_ratio(output, real_label)
            print(score)
            dices.append(score)
        return dices

    if args.schedule:
        callbacks = [lrcallbacks,
                     best_model_check,
                    # escallback,
                     tb]
    else:
        callbacks = [tb, best_model_check]


    def brain_eval(epoch, logs):
        os.system('mkdir -p npy')
        if (epoch+1) % 10 == 0:
            print('test')
            dices = eval_whole_brain(train, int(args.modality))
            np.save('npy/%s_%s_train_%s_%s_%s_%s_dice_epoch%s_%s_%s_modal%s.npy' % (args.loss, args.lr, args.model, args.seed, args.batch, args.shape, epoch, args.schedule, args.decay, args.modality), dices)
            print(dices)
            print(np.mean(dices))
            dices = eval_whole_brain(valid, int(args.modality))
            np.save('npy/%s_%s_valid_%s_%s_%s_%s_dice_epoch%s_%s_%s_modal%s.npy' % (args.loss, args.lr, args.model, args.seed, args.batch, args.shape, epoch, args.schedule, args.decay, args.modality), dices)
            print(dices)
            print(np.mean(dices))
            dices = eval_whole_brain(test, int(args.modality))
            print(dices)
            print(np.mean(dices))
            np.save('npy/%s_%s_test_%s_%s_%s_%s_dice_epoch%s_%s_%s_modal%s.npy' % (args.loss, args.lr, args.model, args.seed, args.batch, args.shape, epoch, args.schedule, args.decay, args.modality), dices)
   
    whole_brain_callbacks = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch,logs: brain_eval(epoch, logs)
    )
    callbacks.append(whole_brain_callbacks)

    n_extend = 5
    #for epoch in range(300):
    #model.train_on_batch()
    history = model.fit_generator(train_data, steps_per_epoch=150, workers=10, use_multiprocessing=True, epochs=300,
                                  callbacks=callbacks, max_queue_size=20,
                                  validation_data=val_data, validation_steps=len(val_data))
                                  #initial_epoch=n_extend*epoch)
    history = history.history

    #lrcallbacks = tf.keras.callbacks.LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 10), (1e-4, 40), (2e-5, 65), (1e-5, 70)]))
    ##unfreeze_model(model)
    #last_model_check = tf.keras.callbacks.ModelCheckpoint('%s/last_dense_unet_{epoch:02d}_{val_loss:.2f}.h5' % model_dir, mode='min',
    #                                                      save_best_only=False, save_weights_only=True)
    #history = model.fit_generator(train_data, steps_per_epoch=20, workers=10, use_multiprocessing=True, epochs=71,
    #                              callbacks=[lrcallbacks],
    #                              max_queue_size=9,
    #                              validation_data=val_data, validation_steps=10)

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return


if __name__ == '__main__':
    main()
