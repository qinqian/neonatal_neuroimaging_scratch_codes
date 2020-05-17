#!/usr/bin/env python
'''
augmentation: 
https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_2D.py
https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/example_ipynb.ipynb
https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

'''
import sys
from random import choice
import random
import matplotlib
from batchviewer import view_batch
from skimage import measure
from sklearn.model_selection import train_test_split
matplotlib.use('Agg')
import os
import nibabel as nib
import ants
import matplotlib.pyplot as plt
import glob
import pandas as pd
from skimage.util.shape import view_as_windows
from torch.utils import data
import numpy as np
import torch
from torch.nn import BCELoss
from torch import optim
from torchvision import transforms
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
## template = ants.image_read('results/template/8weeks_t2f.nii.gz')


def get_crop(mat):
    """ find first slice with non-zero voxels in each dimension borrow from NeBSS"""
    for i in range(mat.shape[0]):
        xmin = i
        if np.sum(mat[i, :, :]) > 0:
            break
    for i in range(mat.shape[0] - 1, 0, -1):
        xmax = i
        if np.sum(mat[i, :, :]) > 0:
            break
    for i in range(mat.shape[1]):
        ymin = i
        if np.sum(mat[:, i, :]) > 0:
            break
    for i in range(mat.shape[1] - 1, 0, -1):
        ymax = i
        if np.sum(mat[:, i, :]) > 0:
            break
    for i in range(mat.shape[2]):
        zmin = i
        if np.sum(mat[:, :, i]) > 0:
            break
    for i in range(mat.shape[2] - 1, 0, -1):
        zmax = i
        if np.sum(mat[:, :, i]) > 0:
            break
    return xmin, xmax, ymin, ymax, zmin, zmax


def estimate_3d_patch(path):
    """
    :param path: path to numpy array storing images and labels
    :return: None
    """
    patch_idx = []
    for image_path in glob.glob(f'{path}/*t2f*npy'):
        label_path = f'{path}/%s_lesion.nii.gz.npy' % os.path.basename(image_path).split('_')[0]

        patient = os.path.basename(image_path).split('_')[0]
        img = np.load(image_path)
        label = np.load(label_path)

        window_imgs = view_as_windows(img, (64, 64, 64), step=32)
        window_labels = view_as_windows(label, (64, 64, 64), step=32)
        patch_num = 0

        for p in range(window_imgs.shape[0]):
            for ip in range(window_imgs.shape[1]):
                for jp in range(window_imgs.shape[2]):
                    patch_idx.append([patient, image_path, label_path, patch_num, p, ip, jp])  # patient, x, y, z
                    patch_num += 1
        window_imgs = window_imgs.reshape((np.prod(window_imgs.shape[:3]), 64, 64, 64))
        window_labels = window_labels.reshape((np.prod(window_labels.shape[:3]), 64, 64, 64))
        for pn in range(window_imgs.shape[0]):
            np.save(f'{image_path}_{pn}_patchimg.npy', window_imgs[pn, :, :, :])
            np.save(f'{image_path}_{pn}_patchlabel.npy', window_labels[pn, :, :, :])
    meta = np.vstack(patch_idx)
    pd.DataFrame(meta).to_csv('meta_patch.csv')


def estimate_2d_patch(image_path, image_label, train=True, step_size=2, view=False):
    """
    :param step_size:
    :param path: path to numpy array storing images and labels
    :return: None
    """
    patch_idx = []
    img = np.load(image_path)
    label = np.load(image_label)
    xmin, xmax, ymin, ymax, zmin, zmax = get_crop(img)
    img = img[xmin:xmax, ymin:ymax, zmin:zmax]
    label = label[xmin:xmax, ymin:ymax, zmin:zmax]
   
    if train:
        pos_slices = []
        for i in range(label.shape[2]):
            if label[:, :, i].sum() > 0:
                pos_slices.append(i)
        img = img[:, :, pos_slices]
        label = label[:, :, pos_slices]

    window_patches = split_patches(img, 64, step_size)
    image_patches = []
    label_patches = []
    for p, ip, jp in window_patches:
        if train:
            if np.sum(label[p:(p + 64), ip:(ip + 64), jp]) > 0:
                img_p = img[p:(p + 64), ip:(ip + 64), jp]
                label_p = label[p:(p + 64), ip:(ip + 64), jp]
                image_patches.append(img_p)
                image_patches.append(img_p[:, ::-1])
                image_patches.append(img_p[::-1, :])
                label_patches.append(label_p)
                label_patches.append(label_p[:, ::-1])
                label_patches.append(label_p[::-1, :])
                for i in range(3):
                    img_p = np.rot90(img_p)
                    label_p = np.rot90(label_p)
                    image_patches.append(img_p)
                    image_patches.append(img_p[:, ::-1])
                    image_patches.append(img_p[::-1, :])
                    label_patches.append(label_p)
                    label_patches.append(label_p[:, ::-1])
                    label_patches.append(label_p[::-1, :])
        else:
            image_patches.append(img[p:(p + 64), ip:(ip + 64), jp])
            label_patches.append(label[p:(p + 64), ip:(ip + 64), jp])

    window_imgs = np.stack(image_patches, axis=0)
    window_labels = np.stack(label_patches, axis=0)
    if train:
        if view:
            vis = np.stack([window_imgs, window_labels], axis=0)
            view_batch(vis, width=300, height=300)
        return window_imgs, window_labels.astype('float32')
    else:
        if view:
            vis = np.stack([img.transpose(2, 0, 1), label.transpose(2, 0, 1)], axis=0)
            view_batch(vis, width=300, height=300)
        return label, img, window_labels.astype('float32'), window_imgs


def prepare_train_valid_test(path):
    image_paths = []
    label_paths = []
    patients = []
    for image_path in glob.glob(f'{path}/*t2f*npy'):
        patient = os.path.basename(image_path).split('_')[0]
        label_path = f'{path}/%s_lesion.nii.gz.npy' % os.path.basename(image_path).split('_')[0]
        image_paths.append(image_path)
        label_paths.append(label_path)
        patients.append(patient)

    df = pd.DataFrame({'image': image_paths, 'label': label_paths, 'patient': patients})
    return df


class NiftyPatches(data.Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index, :, :]
        #label = np.stack([1-label, label], axis=0) # 1-hot encoder
        img = self.imgs[index, :, :]
        img = np.expand_dims(img, 0)
        return [ img, label ]

    def __len__(self):
        return self.imgs.shape[0]


def prepare_patches_loader(meta, train=True):
    imgs = []
    labels = []
    for img, label in zip(meta.iloc[:, 0], meta.iloc[:, 1]):
        window_imgs, window_labels = estimate_2d_patch(img, label, train=train, view=False)
        imgs.append(window_imgs)
        labels.append(window_labels)
    imgs = np.vstack(imgs)
    imgs = (imgs - 0.5872402) / 0.2024262  ## T2 FLAIR
    labels = np.vstack(labels)
    return imgs, labels


def split_patches(image, patch_size, overlap_size):
    """
    :param patch_size: sliding window size
    :param overlap_size: overlap between windows
    :return:
    """
    if len(image.shape) == 3:
        D, H, W = image.shape
    else:
        _, D, H, W = image.shape
    overlap_size = int(overlap_size)
    drange = list(range(0, D - patch_size + 1, overlap_size))
    hrange = list(range(0, H - patch_size + 1, overlap_size))
    zrange = list(range(0, W))
    patches = []
    if (D - patch_size) % overlap_size != 0:
        drange.append(D - patch_size)
    if (H - patch_size) % overlap_size != 0:
        hrange.append(H - patch_size)
    for d in drange:
        for h in hrange:
            for z in zrange:
                patches.append((d, h, z))
    return patches


def global_brain_eval(net, df, idx, view=False):
    dices = []
    mean_dice = 0
    for v in idx:
        raw_label, raw_img, window_labels, window_imgs = estimate_2d_patch(df.iloc[v, 0], df.iloc[v, 1], train=False, step_size=32, view=False)
        window_imgs =  (window_imgs - 0.5872402) / 0.2024262  ## T2 FLAIR
        img = torch.from_numpy(np.expand_dims(window_imgs, 1)).cuda()
        prediction = net(img)
        label = torch.from_numpy(raw_label).cuda()
        mask = torch.zeros_like(label, dtype=torch.float).cuda()
        count_overlap = torch.zeros_like(label, dtype=torch.float).cuda()
        patches = split_patches(label, 64, 32)
        window_labels = torch.from_numpy(window_labels).cuda()
        i = 0
        for d, h, z in patches:
            mask[d:(d+64), h:(h+64), z] += prediction[i, 0, :, :]
            count_overlap[d:(d+64), h:(h+64), z] += 1
            i += 1
        mask /= count_overlap
        predicted = (mask > 0.5).float()
        dice = 2 * ((predicted * label.float()).sum()) / (predicted.sum() + label.float().sum())
        dices.append(dice.cpu().item())
        mean_dice += dice
        if view:
            raw_img = raw_img.transpose(2, 0, 1)
            raw_label = raw_label.transpose(2, 0, 1)
            predicted = predicted.cpu().numpy().transpose(2, 0, 1)
            vis = np.stack([raw_img, raw_label, predicted], axis=0)
            view_batch(vis, width=300, height=300)
    return dices


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from unet import UNet
    from edd_net import EDDNet

    #net = UNet(n_channels=1, n_classes=1).to(device)
    net = EDDNet(1, 1, 2).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)

    criterion = BCELoss()  # already divides batch size
    #scheduler = optim.lr_scheduler.StepLR(optimizer,
    #                                      step_size=1, gamma=0.333)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3333, patience=5)

    epochs = 30
    ## patch_num = estimate_3d_patch('results/segment_iso/')     ## because of interpolation error is obvious, use 2d models
    path = 'results/segment_low_res'

    ## split patients into groups
    df = prepare_train_valid_test(path)
    seed = 999
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train, test = train_test_split(np.arange(df.shape[0]), test_size=0.1, random_state=seed)
    train, valid = train_test_split(train, test_size=0.11, random_state=seed)

    df.iloc[train, :].to_csv('results/train_low_%s.csv' % seed)
    df.iloc[valid, :].to_csv('results/valid_low_%s.csv' % seed)
    df.iloc[test, :].to_csv('results/test_low_%s.csv' % seed)
    estimate_2d_patch(df.iloc[3, 0], df.iloc[3, 1], train=True,
                      view=True)
    print(df.head())
    print(df.iloc[3, 0])
    #estimate_2d_patch(df.iloc[3, 0], df.iloc[3, 1], train=False,
    #                   view=True)

    imgs, labels = prepare_patches_loader(df.iloc[train, :])
    print(imgs.shape)
    train_nii = NiftyPatches(imgs, labels)
    train_loader = data.DataLoader(train_nii, shuffle=True, batch_size=256, num_workers=32)
    test_lr = True
    if test_lr:
        from lr_finder import LRFinder
        lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader, end_lr=0.1, num_iter=200, step_mode='exp')
        lr_finder.plot(log_lr=False)
        plt.savefig('test.pdf')
        sys.exit(0)

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        total = 0
        for i, j in train_loader:
            i = i.to(device)
            j = j.to(device)
            prediction = net(i)
            prediction = prediction.squeeze()
            print(((prediction > 0.5) * (j==1)).sum(dim=2).sum(dim=1))
            prediction = prediction.view(-1)
            j = j.view(-1)
            loss = criterion(prediction, j)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch %s, Training Loss:%s' % (epoch, 1.0*loss.item()))
            total += 1
            if total % 1000 == 0:
                valid_dice = global_brain_eval(net, df, valid)
                scheduler.step(np.mean(valid_dice))
    
        print('Epoch %s, Training Loss:%s' % (epoch, 1.0*epoch_loss/total))
        net.eval()
        train_dice = global_brain_eval(net, df, train)
        print('Epoch %s train dice %s' % (epoch, train_dice))
        valid_dice = global_brain_eval(net, df, valid)
        print('Epoch %s valid dice %s' % (epoch, valid_dice))
        test_dice = global_brain_eval(net, df, test)
        print('Epoch %s test dice %s' % (epoch, test_dice))
        torch.save(net.state_dict(),
                   #'unet_epoch{}.pth'.format(epoch + 1))
                   'unet_step_size2_epoch{}.pth'.format(epoch + 1))
                  # 'unet_step_size1_epoch{}.pth'.format(epoch + 1))
        print('Checkpoint {} saved !'.format(epoch + 1))
