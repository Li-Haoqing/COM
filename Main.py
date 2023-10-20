import os
import numpy as np
import cv2
from lv_set.find_lsf import find_lsf
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Implement of MY model')

    parser.add_argument('--BIAS', type=int, default=20, help='ROI size / 2')
    parser.add_argument('--RATIO', type=int, default=2, help='A ratio of evolution times of LSF related to BIAS')

    parser.add_argument('--timestep', type=int, default=5, help='time step')
    parser.add_argument('--iter_inner', type=int, default=5, help='iter_inner')

    parser.add_argument('--alfa', type=float, default=1.5, help='coefficient of the weighted area term A(phi)')
    parser.add_argument('--epsilon', type=float, default=1.5, help='parameter that specifies the width of the DiracDelta function')

    args = parser.parse_args()
    return args


def level_set_initial(crop):
    # initialize LSF as binary step function
    c0 = 1
    initial_lsf = c0 * np.ones(crop.shape)
    # initialization
    initial_lsf[crop > 50] = -c0

    return initial_lsf


def flood_fill(img_closed_loop):
    im_floodfill = img_closed_loop.copy()
    mask = np.zeros((img_closed_loop.shape[0] + 2, img_closed_loop.shape[1] + 2), np.uint8)

    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_out = 255 - im_floodfill
    return im_out


def Edge_to_Binari(edge, area):
    x, y = area[0][1] - area[0][0], area[1][1] - area[1][0]
    mask = np.zeros([x, y], dtype='int32')
    for e in edge:
        mask[e[0], e[1]] = 255

    mask = flood_fill(mask)
    return mask


def get_points(img):
    count = 1
    Points_list = []
    img_copy = np.copy(img)

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        global img
        if event == cv2.EVENT_LBUTTONDOWN:
            Point = (x, y)
            Points_list.append(Point)
            img = np.copy(img_copy)
            cv2.drawMarker(img, Point, color=(0, 255, 0))
            cv2.imshow("{}.jpg".format(count), img)
            print('add:{}'.format((x, y)))

        elif event == cv2.EVENT_RBUTTONDOWN:
            delete = Points_list[-1]
            Points_list.pop(-1)
            print('delete:{}'.format(delete))
            img = np.copy(img_copy)
            cv2.imshow("{}.jpg".format(count), img)

    while True:
        cv2.namedWindow("{}.jpg".format(count), cv2.WINDOW_NORMAL)
        cv2.resizeWindow("{}.jpg".format(count), 512, 512)
        cv2.moveWindow("{}.jpg".format(count), 100, 100)
        cv2.setMouseCallback("{}.jpg".format(count), on_EVENT_LBUTTONDOWN)
        cv2.imshow("{}.jpg".format(count), img)
        flag = cv2.waitKey()
        # Enter
        if flag == 13:
            break
    return Points_list


def get_areas(points, max_x, max_y, bias):
    areas = []
    for p in points:
        x = [p[1] - bias if (p[1] - bias) > 0 else 0, p[1] + bias if (p[1] + bias) < max_x else max_x]
        y = [p[0] - bias if (p[0] - bias) > 0 else 0, p[0] + bias if (p[0] + bias) < max_y else max_y]
        areas.append([x, y])
    return areas


def get_crop(img, max_x, max_y, bias):
    Pts_list = get_points(img)
    areas = get_areas(Pts_list, max_x, max_y, bias)

    crops = []
    for area in areas:
        print(area[0][0], area[0][1], area[1][0], area[1][1])
        crop = img[area[0][0]:area[0][1], area[1][0]:area[1][1]]
        crop = crop / np.max(crop)
        crops.append(crop)
    return crops, areas


def get_mask(args, img):
    img = img[:, :, 0]

    max_x, max_y = img.shape[0], img.shape[1]
    crops, areas = get_crop(img, max_x, max_y, args.BIAS)

    masks = []
    for crop, area in zip(crops, areas):
        cx, cy = crop.shape[0], crop.shape[1]
        crop = np.interp(crop, [np.min(crop), np.max(crop)], [0, 255])

        initial_lsf = level_set_initial(crop)
        phi, a = find_lsf(cx, cy, crop, initial_lsf, args.timestep, args.iter_inner, int(args.BIAS/args.RATIO),
                          args.alfa, args.epsilon, sigma=1.5)
        edge = a.allsegs[1][0].astype('int32')
        edge = edge[:, [1, 0]]
        m = Edge_to_Binari(edge, area).astype(np.uint8)
        masks.append(m)
    mask = np.zeros([max_x, max_y])
    for m, a in zip(masks, areas):
        mask[a[0][0]:a[0][1], a[1][0]:a[1][1]] = m

    return mask


if __name__ == '__main__':
    args = parse_args()

    img_path = r"datasets/"

    img = cv2.imread(img_path)
    mask = get_mask(args, img)

    filename = os.path.basename(img_path)

    # Save mask
    save_path = 'COM_results/'
    cv2.imwrite(os.path.join(save_path,  filename), mask)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
