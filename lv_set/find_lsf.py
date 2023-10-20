import numpy as np
from scipy.ndimage import gaussian_filter

from lv_set.drlse_algo import drlse_edge
from utils.show_fig import show_fig1, show_fig2, draw_all


def find_lsf(cx, cy, crop: np.ndarray, initial_lsf: np.ndarray, timestep=1, iter_inner=10, iter_outer=30,
             alfa=-3, epsilon=1.5, sigma=0.8):
    """
    :param cx: shape[0] of the crop
    :param cy: shape[1] of the crop
    :param crop: Input crop
    :param initial_lsf: Array as same size as the img that contains the seed points for the LSF.
    :param timestep: Time Step
    :param iter_inner: How many iterations to run drlse before showing the output
    :param iter_outer: How many iterations to run the iter_inner
    :param alfa: coefficient of the weighted area term A(phi)
    :param epsilon: parameter that specifies the width of the DiracDelta function
    :param sigma: scale parameter in Gaussian kernal
    """
    if len(crop.shape) != 2:
        raise Exception("Input image should be a gray scale one")

    if len(crop.shape) != len(initial_lsf.shape):
        raise Exception("Input image and the initial LSF should be in the same shape")

    if np.max(crop) <= 1:
        raise Exception("Please make sure the image data is in the range [0, 255]")

    # parameters
    mu = 0.2 / timestep  # coefficient of the distance regularization term R(phi)

    crop = np.array(crop, dtype='float32')
    img_smooth = gaussian_filter(crop, sigma)
    [Iy, Ix] = np.gradient(img_smooth)
    f = np.square(Ix) + np.square(Iy)
    g = 1 / (1 + f)  # edge indicator function

    # initialization
    phi = initial_lsf.copy()

    show_fig1(phi)
    show_fig2(phi, crop)
    print('show fig 2 first time')

    # start level set evolution
    for n in range(iter_outer):
        phi = drlse_edge(cx, cy, crop, phi, g, mu, alfa, epsilon, timestep, iter_inner)
        print('show fig 2 for %i time' % n)
        a = draw_all(phi, crop)

    # refine the zero level contour by further level set evolution with alfa=0
    alfa = 0
    iter_refine = 10
    phi = drlse_edge(cx, cy, crop, phi, g, mu, alfa, epsilon, timestep, iter_refine)
    return phi, a
