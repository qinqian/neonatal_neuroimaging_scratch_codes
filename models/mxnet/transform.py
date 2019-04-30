import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma):
    """
    Elastic deformation of images as described in [1].
    [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis", in Proc. of the
        International Conference on Document Analysis and Recognition, 2003.
    Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5
    Args:
        image (np.ndarray): image to be deformed
        alpha (list): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (list): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation
    Returns:
        np.ndarray: deformed image
    """

    assert len(alpha) == len(sigma), \
        "Dimensions of alpha and sigma are different"

    channelbool = image.ndim - len(alpha)
    out = np.zeros((len(alpha) + channelbool, ) + image.shape)

    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        array = (np.random.rand(*image.shape) * 2 - 1)
        out[jj] = gaussian_filter(array, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]

    # Map mask to indices
    shapes = list(map(lambda x: slice(0, x, None), image.shape))
    grid = np.broadcast_arrays(*np.ogrid[shapes])
    indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))

    # Transform image based on masked indices
    transformed_image = map_coordinates(image, indices, order=0,
                                        mode='reflect').reshape(image.shape)

    return transformed_image

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

