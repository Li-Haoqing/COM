import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D

plt.ion()
fig1 = plt.figure(1)
fig2 = plt.figure(2)


def visual_3D(z):

    min_z = np.min(z)
    fig = plt.figure()
    ax = Axes3D(fig)

    x = np.arange(0, z.shape[1], 1)
    y = np.arange(0, z.shape[0], 1)
    x, y = np.meshgrid(x, y)

    ax.plot_surface(y, x, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

    plt.show()


def show_fig1(phi: np.ndarray):
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    y, x = phi.shape
    x = np.arange(0, x, 1)
    y = np.arange(0, y, 1)
    X, Y = np.meshgrid(x, y)
    a = ax1.contour(X, Y, phi, 0, colors='g', linewidths=2)
    return a


def show_fig2(phi: np.ndarray, img: np.ndarray):
    fig2.clf()
    contours = measure.find_contours(phi, 0)
    ax2 = fig2.add_subplot(111)
    ax2.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    for n, contour in enumerate(contours):
        ax2.plot(contour[:, 1], contour[:, 0], linewidth=2)


def draw_all(phi: np.ndarray, img: np.ndarray, pause=0.3):
    show_fig2(phi, img)
    a = show_fig1(phi)
    plt.pause(pause)
    return a
