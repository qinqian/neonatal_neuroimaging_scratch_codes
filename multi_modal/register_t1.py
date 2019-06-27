#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from skimage import measure

from matplotlib import gridspec
import matplotlib.pyplot as plt
import os
import pandas as pd
import ants
import sys
from skimage.restoration import denoise_wavelet
from skimage import exposure
import numpy as np

def get_crop(mat):
    """ find first slice with non-zero voxels in each dimension borrow from NeBSS"""
    for i in range(mat.shape[0]):
        xmin = i
        if np.sum(mat[i, :, :]) > 0:
                break
    for i in range(mat.shape[0]-1, 0, -1):
        xmax = i
        if np.sum(mat[i, :, :]) > 0:
                break
    for i in range(mat.shape[1]):
        ymin = i
        if np.sum(mat[:, i, :]) > 0:
                break
    for i in range(mat.shape[1]-1, 0, -1):
        ymax = i
        if np.sum(mat[:, i, :]) > 0:
                break
    for i in range(mat.shape[2]):
        zmin = i
        if np.sum(mat[:, :, i]) > 0:
                break
    for i in range(mat.shape[2]-1, 0, -1):
        zmax = i
        if np.sum(mat[:, :, i]) > 0:
                break
    return xmin,xmax,ymin,ymax,zmin,zmax     

def modify_axis(ax):
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def enhan_ants_image(ants_image, mask, clip_limit=0.015):
    imdata = ants_image.numpy()
    imshape = ants_image.shape

    immin = imdata.min()
    imdata = (imdata - immin)

    adapted = exposure.equalize_adapthist(
        denoise_wavelet(imdata.reshape(imshape[0], -1)), clip_limit=clip_limit)

    adapted_image = adapted.reshape(imshape)
    return ants.make_image(mask.shape, voxval=adapted_image,
                           spacing=ants.get_spacing(mask),
                           direction=ants.get_direction(mask),
                           origin=ants.get_origin(mask))

ages = pd.read_table('/data1/qinqian/MRI/data/Atlas/ALBERT/dhcp/ALBERTs/ages.csv', delimiter=' ', header=None)
ages.sort_values(1, inplace=True)
cga = float(sys.argv[3])
templates = ages.iloc[(ages.iloc[:, 1].values - cga).argsort()[:10], ].iloc[:, 0]

b0 = ants.image_read(f'results/processed/{os.path.basename(sys.argv[1])}')
t1 = ants.image_read(f'results/processed/{os.path.basename(sys.argv[2])}')
t1_label = f'results/processed/{os.path.basename(sys.argv[2])}'.replace('.nii.gz', '')

mask = b0.get_mask()
mask_md = ants.iMath(mask, 'MD', 1)

t1 = t1.resample_image_to_target(b0)
t1 = ants.n4_bias_field_correction(t1)
t1 = enhan_ants_image(t1, mask_md)
t1 = ants.iMath(t1, 'Normalize') * 1000
b0 = ants.n4_bias_field_correction(b0)
b0 = ants.iMath(b0, 'Normalize') * 1000

t1 = t1 * mask_md
t1.reg = ants.registration(b0, t1, type_of_transform='Rigid', outprefix=f'{t1_label}_t1_to_b0')
t1 = t1.reg['warpedmovout']

ilist = []
iseg  = []
for template in templates:
    template = f'{template}-session1'
    alberts_t1 = 'results/ALBERTs/{}/T1/{}.nii.gz'.format(template, template)
    alberts_t2 = 'results/ALBERTs/{}/T2/{}.nii.gz'.format(template, template)
    alberts_seg = f'results/ALBERTs/{template}/segmentations/{template}_tissue_labels.nii.gz'
    t1t = ants.image_read(alberts_t1)
    segs = ants.image_read(alberts_seg)
    t1_reg = ants.registration(t1, t1t, type_of_transform='SyNRA', outprefix=f'{t1_label}_albert_to_t1')
    ilist.append(t1_reg['warpedmovout'])
    segs = ants.apply_transforms(t1, segs, transformlist=t1_reg['fwdtransforms'])
    iseg.append(segs)

r = 2
joint_probs = ants.joint_label_fusion(t1, mask_md, ilist, r_search=r, 
                                      label_list=iseg, rad=[r]*t1.dimension)
#seg = ants.eig_seg(mask_md, joint_probs['probabilityimages'])
#csf = ants.threshold_image(seg, 1, 1)
#ventricle = ants.threshold_image(seg, 5, 5)
#cerestem = ants.threshold_image(seg, 6, 6)
#wm = ants.threshold_image(seg, 3, 3)
#hwm = ants.threshold_image(seg, 8, 8)
#lwm = ants.threshold_image(seg, 9, 9)
#wm = ants.threshold_image(wm+hwm+lwm, 1, 3)

csf = ants.threshold_image(joint_probs['probabilityimages'][0], 0.8, 1)
ventricle = ants.threshold_image(joint_probs['probabilityimages'][4], 0.8, 1)
csf = ants.threshold_image(ventricle+csf, 1, 2)
csf = ants.iMath(csf, 'ME', 1)
csf = csf*mask

wm = ants.threshold_image(joint_probs['probabilityimages'][2], 0.8, 1)
hwm = ants.threshold_image(joint_probs['probabilityimages'][8], 0.8, 1)
lwm = ants.threshold_image(joint_probs['probabilityimages'][7], 0.8, 1)
cerestem = ants.threshold_image(joint_probs['probabilityimages'][5], 0.8, 1)
#wm = ants.iMath(wm+hwm+lwm, 'ME', 1)
wm = ants.iMath(ants.threshold_image(wm+lwm, 1, 2), 'ME', 1)
wm = wm*mask
wm_val = wm.numpy()
wm_val[cerestem.numpy()==1] = 0
wm = ants.make_image(mask.shape, voxval=wm_val,
                 spacing=ants.get_spacing(mask),
                 direction=ants.get_direction(mask),
                 origin=ants.get_origin(mask))

#background = ants.threshold_image(joint_probs['probabilityimages'][3], 0.8, 1)
#mask_val = mask.numpy()
#mask_val[csf.numpy()==1] = 0
#mask_val[background.numpy()==1] = 0
#mask_edge = ants.make_image(mask.shape, voxval=mask_val,
#                 spacing=ants.get_spacing(mask),
#                 direction=ants.get_direction(mask),
#                 origin=ants.get_origin(mask))
#edge = ants.iMath(mask_edge, 'ME', 10)
#edge_val = edge.numpy()
#mask_val = mask.numpy()
#mask_val[edge_val==1] = 0
#mask_edge = ants.make_image(mask.shape, voxval=mask_val,
#                 spacing=ants.get_spacing(mask),
#                 direction=ants.get_direction(mask),
#                 origin=ants.get_origin(mask))

ants.image_write(t1, f'{t1_label}_reg.nii.gz')
#ants.image_write(wm, f'{t1_label}_wm.nii.gz')
ants.image_write(wm, f'{t1_label}_wm_nohwm.nii.gz')
ants.image_write(csf, f'{t1_label}_csf.nii.gz')
#ants.image_write(mask_edge, f'{t1_label}_edge.nii.gz')
ants.image_write(b0, f'{t1_label}_b0.nii.gz')

t1 = t1.crop_image(mask_md)
wm = wm.crop_image(mask_md)
csf = csf.crop_image(mask_md)
#edge = mask_edge.crop_image(mask_md)

def plot_contour(img, mask1, mask2):
    ncol=7
    nrow=6
    fig = plt.figure(figsize=((nrow-2.2)*8, (ncol)*5))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    n=15
    cols = ['r', 'b', 'g']
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            ax.imshow(np.rot90(img.numpy()[:, :, n]), cmap=plt.cm.gray)
            for c, m in enumerate([mask1, mask2]):
                contours0 = measure.find_contours(np.rot90(m.numpy()[:, :, n]), 0.8)
                for _, contour in enumerate(contours0):
                    ax.plot(contour[:, 1], contour[:, 0], linestyle='solid', linewidth=3, color=cols[c])
            n+=1
            modify_axis(ax)
            ax.set_aspect('equal')    
            ax.axis('image')
    plt.savefig(f'{t1_label}_seg.pdf')

plot_contour(t1, wm, csf)

