import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y-1))
    val_ar.append(get_pixel(img, center, x-1, y))
    val_ar.append(get_pixel(img, center, x-1, y + 1))
    val_ar.append(get_pixel(img, center, x, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    val_ar.append(get_pixel(img, center, x + 1, y))
    val_ar.append(get_pixel(img, center, x + 1, y-1))
    val_ar.append(get_pixel(img, center, x, y-1))
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

path = 'test.jpg'
img_bgr = cv2.imread(path, 1)
height, width, _ = img_bgr.shape

# Convert the color image to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Create an empty LBP feature map
lbp_feature_map = np.zeros((height, width), np.uint8)

# Compute the LBP feature for each pixel and populate the feature map
for i in range(0, height):
    for j in range(0, width):
        lbp_feature_map[i, j] = lbp_calculated_pixel(img_gray, i, j)

# Display the LBP feature map
plt.imshow(lbp_feature_map, cmap="gray")
plt.title("LBP Feature Map")
plt.show()

print("LBP Program is finished")
