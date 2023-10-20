import numpy as np
from scipy.ndimage import laplace

from utils.neighb import neighb3


def in_ex_mean(cx, cy, img, initial_lsf):
    ipts = np.flatnonzero(initial_lsf <= 0)
    epts = np.flatnonzero(initial_lsf > 0)

    nepts = neighb3(cx, cy, epts)

    i = np.sum(img.flat[ipts]) / (len(ipts) + np.finfo(float).eps)
    e = np.sum(img.flat[nepts]) / (len(nepts) + np.finfo(float).eps)

    return i, e


def drlse_edge(cx, cy, crop, phi_0, g, mu, alfa, epsilon, timestep, iters):  # Updated Level Set Function
    """
    :param cx: shape[0] of the crop
    :param cy: shape[1] of the crop
    :param crop: region of interest
    :param phi_0: level set function to be updated by level set evolution
    :param g: edge indicator function
    :param mu: weight of distance regularization term
    :param alfa: weight of the weighted area term
    :param epsilon: width of Dirac Delta function
    :param timestep: time step
    :param iters: number of iterations
    """
    phi = phi_0.copy()
    i_max = 0
    for k in range(iters):
        phi = neumann_bound_cond(phi)
        [phi_y, phi_x] = np.gradient(phi)
        s = np.sqrt(np.square(phi_x) + np.square(phi_y))
        delta = 1e-10
        n_x = phi_x / (s + delta)  # add a small positive number to avoid division by zero
        n_y = phi_y / (s + delta)
        curvature = div(n_x, n_y)

        a = alfa
        i, e = in_ex_mean(cx, cy, crop, phi)
        if i > i_max:
            i_max = i
        else:
            a = -0.1

        dist_reg_term = dist_reg_p2(phi)

        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g
        ed_term = dirac_phi * (1 / ((i - e) ** 2 + delta)) * curvature

        phi += timestep * (mu * dist_reg_term + a * area_term + 10 * delta * ed_term)
    return phi


def dist_reg_p2(phi):
    [phi_y, phi_x] = np.gradient(phi)
    s = np.sqrt(np.square(phi_x) + np.square(phi_y))
    a = (s >= 0) & (s <= 1)
    b = (s > 1)
    ps = a * np.sin(2 * np.pi * s) / (2 * np.pi) + b * (s - 1)  # compute first order derivative of the double-well potential
    dps = ((ps != 0) * ps + (ps == 0)) / ((s != 0) * s + (s == 0))  # compute d_p(s)=p'(s)/s
    return div(dps * phi_x - phi_x, dps * phi_y - phi_y) + laplace(phi, mode='nearest')


def div(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    [_, nxx] = np.gradient(nx)
    [nyy, _] = np.gradient(ny)
    return nxx + nyy


def dirac(x: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    f = (1 / 2 / sigma) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def neumann_bound_cond(f):
    g = f.copy()

    g[np.ix_([0, -1], [0, -1])] = g[np.ix_([2, -3], [2, -3])]
    g[np.ix_([0, -1]), 1:-1] = g[np.ix_([2, -3]), 1:-1]
    g[1:-1, np.ix_([0, -1])] = g[1:-1, np.ix_([2, -3])]
    return g
