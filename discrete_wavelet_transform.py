''' 
This code is for applying Discrete Wavelet Transform
as one of the feature extraction techniques
'''

import numpy as np
import cv2

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
  height, width = image.shape
  coefficients = []

  for _ in range(2):
      for i in range(height):
          image[i, :width] = haar_transform(image[i, :width])

      for j in range(width):
          image[:height, j] = haar_transform(image[:height, j])

      # extract and store the LL, LH, HL, and HH subbands
      LL = image[:height // 2, :width // 2]
      LH = image[:height // 2, width // 2:]
      HL = image[height // 2:, :width // 2]
      HH = image[height // 2:, width // 2:]

      coefficients.append((LL, LH, HL, HH))

      height //= 2
      width //= 2

  return cv2.resize(coefficients[0][3], dsize=(512, 512))