import nibabel
import numpy as np
from matplotlib import gridspec
import matplotlib
matplotlib.use('Agg')
import ants
#from nilearn.plotting import plot_anat, show
#from nilearn import plotting
from ants.utils.convert_nibabel import to_nibabel
import matplotlib.pyplot as plt
from skimage import measure

def enhan(in_file, clip_limit=0.010):
    import numpy as np
    import nibabel as nb
    import os.path as op
    from skimage import exposure, img_as_int
    from nipype.utils import NUMPY_MMAP

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_enh.nii.gz' % fname)

    im = nb.load(in_file, mmap=NUMPY_MMAP)
    imdata = im.get_data()
    imshape = im.shape

    if in_mask is not None:
        msk = nb.load(in_mask, mmap=NUMPY_MMAP).get_data()
        msk[msk > 0] = 1
        msk[msk < 1] = 0
        imdata = imdata * msk

    immin = imdata.min()
    imdata = (imdata - immin).astype(np.uint16)

    adapted = exposure.equalize_adapthist(
        imdata.reshape(imshape[0], -1), clip_limit=clip_limit)
    return adapted

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
    # ax[0, 0].xaxis.set_visible(False)  # Hide only x axis

def plot_with_mask(t2f, name, mask, mask2):
    mask = nibabel.load(mask)
    mask_data = mask.get_data()
    mask2 = nibabel.load(mask2)
    mask_data2 = mask2.get_data()
    
    t2flair = nibabel.load(t2f)
    t2flair_ants = ants.image_read(t2f)
    t2flair_data = t2flair.get_data()
    whole_mask = ants.get_mask(t2flair_ants)
    
    print(t2flair.affine)
    x_trans, y_trans, z_trans = np.abs(t2flair.affine.diagonal()[:3])
    
    # crop the data
    xmin, xmax, ymin, ymax, zmin, zmax = get_crop(whole_mask)
    print(xmin, xmax, ymin, ymax, zmin, zmax)
    print(t2flair_data.shape)
    
    t2flair_data = t2flair_data[xmin:xmax:1, ymin:ymax:1, :]
    mask_data = mask_data[xmin:xmax:1, ymin:ymax:1, :]
    mask_data2 = mask_data2[xmin:xmax:1, ymin:ymax:1, :]
    
    #t2 = nibabel.load(args.t2)
    # subplot layout https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot/41075262
    nrow = 4
    ncol = 5
    fig = plt.figure(figsize=((ncol-0.4)*2.5, (nrow+0.3)*2.5))
    #gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, top=1-0.5/(nrow+1), bottom=0.5/(nrow+1), left=0.5/(ncol+1), right=1-0.5/(ncol+1))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0, hspace=0, top=1, bottom=0, left=0, right=1)
    
    n = 0
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            ax.set_aspect("equal") # default equal
            if (n >= t2flair_data.shape[2]):
                ax.imshow(np.rot90(np.zeros(t2flair_data.shape[:2])), interpolation='nearest', cmap=plt.cm.gray,
                          extent=[0, x_trans*(xmax-xmin), 0, y_trans*(ymax-ymin)], origin='lower')
                modify_axis(ax)
                continue
    
            ax.imshow(np.rot90(t2flair_data[:, :, n]), interpolation='nearest', cmap=plt.cm.gray)

            contours0 = measure.find_contours(np.rot90(mask_data[:, :, n]==1), 0.8)
            contours1 = measure.find_contours(np.rot90(mask_data[:, :, n]==4), 0.8)
            for _, contour in enumerate(contours0):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            for _, contour in enumerate(contours1):
                ax.plot(contour[:, 1], contour[:, 0], linestyle='dashed', linewidth=1, color='r')

            contours2 = measure.find_contours(np.rot90(mask_data2[:, :, n]==1), 0.8)
            contours3 = measure.find_contours(np.rot90(mask_data2[:, :, n]==4), 0.8)
            for _, contour in enumerate(contours2):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='b')
            for _, contour in enumerate(contours3):
                ax.plot(contour[:, 1], contour[:, 0], linestyle='dashed', linewidth=1, color='b')

            modify_axis(ax)
            n += 1
    plt.savefig("%s.pdf" % name, dpi=500, facecolor=[0,0,0], bbox_inches=None,
                pad_inches=0)
    plt.close()

