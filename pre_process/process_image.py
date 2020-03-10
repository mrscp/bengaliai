import numpy as np
import cv2


HEIGHT = 137
WIDTH = 236
SIZE = 128


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    ymin, ymax, xmin, xmax = bbox(img0[5:-5,5:-5] > 80)
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    img = cv2.resize(img, (size, size))
    return img


def save_image(location, image):
    cv2.imwrite(location, image)
