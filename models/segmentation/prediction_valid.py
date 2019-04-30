from skimage import measure

from plot import modify_axis
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import pandas as pd
import numpy as np
from numpy.core.umath_tests import inner1d
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
import nibabel as nib

from skimage.util import view_as_windows
from losses import double_head_loss, sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1

# Hausdorff Distance
def HausdorffDist(A,B):
    # Hausdorf Distance: Compute the Hausdorff distance between two point
    # clouds.
    # Let A and B be subsets of metric space (Z,dZ),
    # The Hausdorff distance between A and B, denoted by dH(A,B),
    # is defined by:
    # dH(A,B) = max(h(A,B),h(B,A)),
    # where h(A,B) = max(min(d(a,b))
    # and d(a,b) is a L2 norm
    # dist_H = hausdorff(A,B)
    # A: First point sets (MxN, with M observations in N dimension)
    # B: Second point sets (MxN, with M observations in N dimension)
    # ** A and B may have different number of rows, but must have the same
    # number of columns.
    #
    # Edward DongBo Cui; Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat,axis=0)),np.max(np.min(D_mat,axis=1))]))
    return(dH)

def ModHausdorffDist(A,B):
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)


def split_patches(image, patch_size, overlap_size):
    """
    :param image:  4D image, last dimension is channel, now support nii
    :param patch_size: sliding window size
    :param overlap_size: overlap between windows
    :return:
    """
    D, H, W, C = image.shape

    overlap_size = int(overlap_size)
    drange = list(range(0, D-patch_size+1, overlap_size))
    hrange = list(range(0, H-patch_size+1, overlap_size))

    if (D-patch_size) % overlap_size != 0:
        drange.append(D-patch_size)
    if (H-patch_size) % overlap_size != 0:
        hrange.append(H-patch_size)

    patches = []
    for d in drange:
        for h in hrange:
            patches.append((d,h))
    return patches

def dice_ratio(pred, label):
    '''Note: pred & label should only contain 0 or 1.
    '''
    #print(pred[label==1])
    return np.sum(pred[label==1])*2.0 / (np.sum(pred) + np.sum(label))

def predict_whole_brain(model, niis, crop_size, labels, npy=False, prob=True, batch=16):
    """
    :param model:
    :return:
    """
    images = []
    for i in niis:
        images.append(nib.load(i).get_data())
    #images = np.array(images)
    images = np.stack(images, axis=0)

    images = np.transpose(images, (1,2,3,0)) # channels last

    for i in range(images.shape[2]):
        images[:, :, i, :] = images[:, :, i, :] / np.max(images[:, :, i, :]) * 255 # norm as image slices, use this
    #images = images/np.max(images)
    images = images/127.5 - 1 ## (norm by image slices) (plus/only) divided by 127.5 then -1, is still working...
    #print(images.max())
    #print(images.min())

    patches = split_patches(images, crop_size, crop_size//2)
    #patches = view_as_windows(images, crop_size, crop_size//2)
    output = np.zeros((images.shape[0], images.shape[1], images.shape[2], 2), dtype=np.float32)
    overlap_count = np.zeros((images.shape[0], images.shape[1], images.shape[2], 2), dtype=np.float32)
    #predictions = {}
    for s in range(images.shape[2]):
        slice_image = []
        for d, h in patches:
            image = images[:, :, s, :]
            #image = np.expand_dims(image[d:(d+crop_size), h:(h+crop_size), :], axis=0)
            image = image[d:(d+crop_size), h:(h+crop_size), :]
            overlap_count[d:(d+crop_size), h:(h+crop_size), s, :] += 1

            slice_image.append(image)

        slice_image = np.stack(slice_image, axis=0)
        #prediction = model.predict(slice_image, batch_size=len(patches))
        prediction = model.predict(slice_image, batch_size=batch)

        i = 0
        for d, h in patches:
            output[d:(d+crop_size), h:(h+crop_size), s, :] += prediction[i, :, :, :]
            i += 1

            #label = np.zeros((crop_size, crop_size, 2), dtype=np.int32)
            #label[:, :, 1] = labels[d: (d+crop_size), h:(h+crop_size), s].astype(np.int32)
            #label[:, :, 0] = (labels[d: (d+crop_size), h:(h+crop_size), s]==0).astype(np.int32)
            #label = np.expand_dims(label, axis=0)
            #print(model.evaluate(image, label, batch_size=1))
            #for i in range(crop_size):
            #    for j in range(crop_size):
            #        key = (d+i, h+j, s)
            #        if key not in predictions.keys():
            #            predictions[key] = []
            #        predictions[key].append(prediction[i, j, :])

    output /= overlap_count
    print(output.mean())
    #for key in predictions.keys():
    #    prediction = np.array(predictions[key])
    #    output[key[0], key[1], key[2], :] = np.mean(prediction, axis=0)

    if prob:
        output = output[:, :, :, 1]
    else:
        output = np.argmax(output, axis=-1)
    if npy:
        np.save(str(niis[0]) + str(crop_size) + '.npy', output)
    return output

def visualize_seg(flair, pred, prob=True, name='test'):
    image = nib.load(flair)
    x_trans, y_trans, z_trans = np.abs(image.affine.diagonal()[:3])
    t2flair_data = image.get_data()
    xmax = t2flair_data.shape[0]
    xmin = 0
    ymax = t2flair_data.shape[1]
    ymin = 0

    nrow = 4
    ncol = 5
    fig = plt.figure(figsize=((ncol-0.4)*2.5, (nrow+0.3)*2.5))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    n = 0
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            ax.set_aspect("equal") # default equal
            if (n >= image.shape[2]):
                ax.imshow(np.rot90(np.zeros(t2flair_data.shape[:2])), interpolation='nearest', cmap=plt.cm.gray,
                          extent=[0, x_trans*(xmax-xmin), 0, y_trans*(ymax-ymin)], origin='lower')
                modify_axis(ax)
                continue

            ax.imshow(np.rot90(t2flair_data[:, :, n]), interpolation='nearest', cmap=plt.cm.gray)

            if not prob:
                contours0 = measure.find_contours(np.rot90(pred[:, :, n]>0), 0.8)
                for _, contour in enumerate(contours0):
                    ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            else:
                ax.imshow(np.rot90(pred[:, :, n]), interpolation='nearest', cmap=nilearn_cmaps['blue_transparent_full_alpha_range'], alpha=0.75)
            modify_axis(ax)
            n += 1
    plt.savefig(f"figures/%s.pdf" % name, dpi=500, facecolor=[0,0,0], bbox_inches=None,
                pad_inches=0)


if __name__ == '__main__':
    from unet import densenet_unet
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #df = pd.read_csv('test_99.csv')
    #df = pd.read_csv('test_9.csv')
    df = pd.read_csv('valid_9.csv')
    print(df.head())
    #df = pd.read_csv('test_5.csv')
    shape = 64
    #model = densenet_unet(2, (shape, shape, 3))
    from unet import edd_net
    model = edd_net(2, (64,64,3), n=2)
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/bce_model_99/best_dense_unet_63_0.00_0.552.h5')
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/double_model_5/best_dense_unet_72_0.18_0.665.h5')
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/bce_model_99_center/best_dense_unet_06_0.44_0.577.h5')
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/bce_model_99_center/best_dense_unet_35_0.29_0.688.h5')
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/double_model_99_center_shape64/best_dense_unet_80_0.38_0.430.h5')
    #model.load_weights('/data1/qinqian/MRI/01_PVL_AI/phaseC_pvl_lesion_seg/models_store/focal_model_99_center/best_dense_unet_80_0.38_0.430.h5')
    #model.load_weights('models_store/double_model_center_32_9/best_dense_unet_53_0.43_0.775.h5')
    model.load_weights('models_store.bak2/edd_seed_bce_00_64_batch16/best_edd_279_0.00020_0.99925.h5')

    adam = tf.keras.optimizers.Adam(lr=2e-4, amsgrad=True)
    model.compile(loss = double_head_loss, optimizer=adam,
                  metrics=[sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1, tf.keras.metrics.binary_crossentropy])

    #df = pd.read_csv('train_9.csv')
    #for i in range(df.shape[0]):
    #    niis = [df.loc[:, 'flair'][i], df.loc[:, 'dwi'][i], df.loc[:, 't1'][i]]
    #    real_label = nib.load(df.loc[:, 'label'][i]).get_data()
    #    predict(model, niis, shape, real_label, npy=True, prob=True)
    #    label = np.load(str(niis[0]) + str(shape) + '.npy')
    #    print(dice_ratio(label>0.3, real_label))

    #    print(label.shape)
    #    #visualize_seg(niis[0], label, prob=True, name='train_pred%s' % i)
    #    #visualize_seg(niis[0], (label>0.5).astype('int'), prob=False, name='train_real%s' % i)
    #    #visualize_seg(niis[0], real_label, prob=False, name='train_real%s' % i)

    for i in range(5, df.shape[0]):
        print(i)
        niis = [df.loc[:, 'flair'][i], df.loc[:, 'dwi'][i], df.loc[:, 't1'][i]]
        print(niis)
        real_label1 = nib.load(df.loc[:, 'label'][i]).get_data()
        real_label2 = nib.load(df.loc[:, 'label2'][i]).get_data()
        real_label = (real_label2 | real_label1).astype(np.int32)
        print(real_label.sum())
        predict_whole_brain(model, niis, shape, real_label, npy=True, prob=True)
        label = np.load(str(niis[0]) + str(shape) + '.npy')
        print(dice_ratio(label>0.5, real_label))

        visualize_seg(niis[0], label, prob=True, name='test_pred_heat%s' % i)
        visualize_seg(niis[0], (label>0.5).astype('int'), prob=False, name='test_pred_mask%s' % i)
        visualize_seg(niis[0], real_label, prob=False, name='test_real%s' % i)

    #df = pd.read_csv('valid_99.csv')
    #for i in range(df.shape[0]):
    #    niis = [df.loc[:, 'flair'][i], df.loc[:, 'dwi'][i], df.loc[:, 't1'][i]]
    #    real_label = nib.load(df.loc[:, 'label'][i]).get_data()
    #    predict(model, niis, shape, real_label)
    #    label = np.load(str(niis[0]) + str(shape) + '.npy')

    #    print(label.shape)
    #    visualize_seg(niis[0], label, 'valid_pred%s' % i)
    #    visualize_seg(niis[0], real_label, 'valid_real%s' % i)
