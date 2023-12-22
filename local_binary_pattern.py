import numpy as np
import timeit
from numba import jit


@jit(nopython=True)
def get_pixel(image, x, y):
    if 0 <= x < len(image) and 0 <= y < len(image[0]):
        return image[x][y]
    return 0

@jit(nopython=True)
def lbp_calculated_pixel(image, x, y):
    center = image[x][y]
    val_ar = []
    val_ar.append(get_pixel(image, x - 1, y - 1) >= center)
    val_ar.append(get_pixel(image, x - 1, y) >= center)
    val_ar.append(get_pixel(image, x - 1, y + 1) >= center)
    val_ar.append(get_pixel(image, x, y + 1) >= center)
    val_ar.append(get_pixel(image, x + 1, y + 1) >= center)
    val_ar.append(get_pixel(image, x + 1, y) >= center)
    val_ar.append(get_pixel(image, x + 1, y - 1) >= center)
    val_ar.append(get_pixel(image, x, y - 1) >= center)

    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] << i
    return val

@jit(nopython=True)
def lbp(image):
    height, width = len(image), len(image[0])
    lbp_values = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lbp_value = lbp_calculated_pixel(image, i, j)
            lbp_values[i][j] = lbp_value

    return lbp_values
