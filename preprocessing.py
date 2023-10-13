'''
This code is for preprocessing the image before
going to feature extraction phase

1. The image will be converted to grayscale
2. The image will be resize to 256x256 (as mentioned to the system architecture
   where images will be resize to a common size)
3. The image will have a noice reduction using Gaussian Noice Reduction also known as Gaussian blur or Gaussian smoothing

'''

import cv2
import numpy as np
import time

def preprocessing(image):
    # Load the image in grayscale
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Check the dimensions
    dimension = np.array(image.shape)

    if dimension[0] == 256 and dimension[1] == 256:
        print("\n... Image is already in 256x256\n\n")
        final_image = image
    else:
        print("\nWarning: Image is not resized to 256x256")
        time.sleep(1)
        print('...')
        time.sleep(1)
        print("\nPerforming image resizing (256x256)")
        print('...')
        time.sleep(1)
        final_image = cv2.resize(image, (256, 256))

    # Apply Gaussian noise reduction
    print("\nPerforming Gaussian noise reduction")
    final_image = cv2.GaussianBlur(final_image, (5, 5), 0)  # You can adjust the kernel size and standard deviation as needed

    time.sleep(2)
    return final_image
