''' 
This code is for applying Discrete Wavelet Transform
as one of the feature extraction techniques
'''

import numpy as np
import cv2
from preprocessing import preprocessing
import os
import matplotlib.pyplot as plt

#applying haar wavelet
def haar_transform(matrix):
  matrix = matrix.astype(float)
  length = len(matrix)
  temp = np.zeros_like(matrix, dtype=float)

  for i in range(length // 2):
    temp[i] = (matrix[2*i] + matrix[2*i + 1]) / 2
    temp[length // 2 + i] = (matrix[2*i] - matrix[2*i + 1]) / 2

  return temp

#applying 2D dwt
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

  #getting high frequency subband (hh)
  height, width = transformed_image.shape
  hh_subband = transformed_image[:height // 2, width // 2:]
  hh_subband = cv2.resize(hh_subband, dsize=(512, 512))

  #for data visualization only
  print('\nDWT Features:\n', hh_subband)

  # cv2.imshow('DWT Transformed Image', hh_subband)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  return hh_subband

def hh_subband_mean(dwt):
  hh_subband = dwt

  hh_mean = sum(map(sum, hh_subband)) / (len(hh_subband) * len(hh_subband[0]))

  return hh_mean


def visualize(directory):
  images = []

  for filename in os.listdir(directory):
        image = os.path.join(directory, filename)
        if image is not None:
            images.append(image)

  preprocessed = []
  for i in images:
    preprossed_img = preprocessing(i)
    preprocessed.append(preprossed_img)

  dwt = []
  for i in preprocessed:
    dwt_img = dwt_2d(i)
    dwt.append(dwt_img)

  mean = []
  for i in dwt:
    mean_img = hh_subband_mean(i)
    mean.append(mean_img)

  return mean

# directory = "/Users/Danniel/Documents/datasets/real"
# directory1 = "/Users/Danniel/Documents/datasets/gan"


# real = visualize(directory)
# gan = visualize(directory1)

# plt.plot(real, label="real", color="blue")
# plt.plot(gan, label="gan", color="red")

# # Adding labels and a title
# plt.xlabel('Index')
# plt.ylabel('Mean Value')
# plt.title('Mean Values for Two Classes')


# # Display the plot
# plt.show()