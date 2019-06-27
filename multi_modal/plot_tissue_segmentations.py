from skimage import measure
import os
from skimage.restoration import denoise_wavelet
from skimage import exposure
import numpy as np
import glob
import ants
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
matplotlib.use('Agg')
def modify_axis(ax):
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

def plot_contour(img, mask1, mask2, label):
    ncol=7
    nrow=6
    fig = plt.figure(figsize=((nrow-2.2)*11, (ncol)*8))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    n=20
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
    plt.savefig(f'{label}_seg.pdf')

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


for seg in glob.glob('/data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/sub-*/segmentations/*tissue_labels.nii.gz'):
    label = os.path.basename(os.path.dirname(os.path.dirname(seg))).replace('-session1', '')
    if not os.path.exists(f'{label}_b0_seg.pdf'):
        b0 = f'/data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/{label}-session1/T2/{label}-session1.nii.gz'
     
        print(label)
        print(b0, seg)
        t1 = ants.image_read(f'results/processed/{label}_T1w_noise_corrected_reoriented_enh_reg.nii.gz')
        segimg = ants.image_read(seg)
        b0 = ants.image_read(b0)
        mask = b0.get_mask()
        print(t1.shape, b0.shape, mask.shape)
        b0 = b0.crop_image(mask)
        segimg = segimg.crop_image(mask)
        t1 = t1.crop_image(mask)
        plot_contour(b0, ants.threshold_image(segimg, 3, 3), 
                         ants.threshold_image(ants.threshold_image(segimg, 5, 5)+ants.threshold_image(segimg, 1, 1), 1, 2), label+'_b0')
        plot_contour(t1, ants.threshold_image(segimg, 3, 3), 
                         ants.threshold_image(ants.threshold_image(segimg, 5, 5)+ants.threshold_image(segimg, 1, 1), 1, 2), label+'_t1')
    else:
        print(label)

