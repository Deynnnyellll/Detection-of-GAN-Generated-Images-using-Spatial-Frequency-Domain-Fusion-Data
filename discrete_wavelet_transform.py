''' 
This code is for applying Discrete Wavelet Transform
as one of the feature extraction techniques
'''

import numpy as np
import cv2
from preprocessing import preprocessing
import os
import matplotlib.pyplot as plt

# applying haar wavelet
def haar_transform(matrix):
  matrix = matrix.astype(float)
  length = len(matrix)
  temp = np.zeros_like(matrix, dtype=float)

  for i in range(length // 2):
    temp[i] = (matrix[2*i] + matrix[2*i + 1]) / 2
    temp[length // 2 + i] = (matrix[2*i] - matrix[2*i + 1]) / 2

  return temp

# applying 2D dwt
def dwt_2d(image):
  original_height, original_width = image.shape
  height = 2**int(np.ceil(np.log2(original_height)))
  width = 2**int(np.ceil(np.log2(original_width)))

  if original_height != height or original_width != width:
    image = cv2.resize(image, (width, height))

  transformed_image = np.copy(image)


  while height >= 2 and width >= 2:
    for i in range(height):
      transformed_image[i, :width] = haar_transform(transformed_image[i, :width])

    for j in range(width):
      transformed_image[:height, j] = haar_transform(transformed_image[:height, j])

    height //= 2
    width //= 2

  # getting high frequency subband (hh)
  height, width = transformed_image.shape
  hh_subband = transformed_image[:height // 2, width // 2:]
  hh_subband = cv2.resize(hh_subband, dsize=(512, 512))

  # for data visualization only
  print('\nDWT Features:\n', hh_subband)

  return hh_subband