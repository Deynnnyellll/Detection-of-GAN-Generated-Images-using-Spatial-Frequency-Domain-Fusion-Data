import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing

def get_pixel(image, x, y):
    if 0 <= x < len(image) and 0 <= y < len(image[0]):
        return image[x][y]
    return 0

def lbp_calculated_pixel(image, x, y):
    center = image[x][y]
    val_ar = []
    val_ar.append(get_pixel(image, x - 1, y - 1))
    val_ar.append(get_pixel(image, x - 1, y))
    val_ar.append(get_pixel(image, x - 1, y + 1))
    val_ar.append(get_pixel(image, x, y + 1))
    val_ar.append(get_pixel(image, x + 1, y + 1))
    val_ar.append(get_pixel(image, x + 1, y))
    val_ar.append(get_pixel(image, x + 1, y - 1))
    val_ar.append(get_pixel(image, x, y - 1))

    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] << i
    return val

def lbp(image):
    height, width = len(image), len(image[0])
    lbp_feature_map = [[0] * width for _ in range(height)]
    lbp_values = [[0] * width for _ in range(height)]

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lbp_value = lbp_calculated_pixel(image, i, j)
            lbp_feature_map[i][j] = lbp_value
            lbp_values[i][j] = lbp_value

    return lbp_feature_map, lbp_values

# Load and preprocess the image
image_path = 'test.jpg'
preprocessed_image = preprocessing(image_path)

# Calculate LBP feature map and values
lbp_map, lbp_values = lbp(preprocessed_image)

# Display the LBP feature map
plt.imshow(np.array(lbp_map), cmap="gray")
plt.title("LBP Feature Map")
plt.show()

# Display the LBP values
print("LBP Value:")
print(np.array(lbp_values))
