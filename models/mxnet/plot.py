import ants
import nibabel
import numpy as np
from matplotlib import gridspec
from nilearn.plotting import plot_anat, show
from nilearn import plotting
from ants.utils.convert_nibabel import to_nibabel
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import measure

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

def plot_with_mask(mask, t2f, name):
    mask = nibabel.load(mask)
    mask_data = mask.get_data()
    
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
    
            contours = measure.find_contours(np.rot90(mask_data[:, :, n]), 0.8)
            ax.imshow(np.rot90(t2flair_data[:, :, n]), interpolation='nearest', cmap=plt.cm.gray)
            for _, contour in enumerate(contours):
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1, color='r')
            modify_axis(ax)
            n += 1
    plt.savefig("%s.pdf" % name, dpi=500, facecolor=[0,0,0], bbox_inches=None,
                pad_inches=0)
    plt.close()

