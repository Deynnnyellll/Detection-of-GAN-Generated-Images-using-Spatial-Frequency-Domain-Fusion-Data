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
        print("\n\nImage is already in 256x256\n\n")
        final_image = image
    else:
        print("\n\nWarning: Image is not resized")
        print('...')
        print("\nPerforming image resizing")
        print('...')
        final_image = cv2.resize(image, dsize=(512, 512))

    # Noise reduction using Gaussian Blurring
    print("\nApplying Gaussian Noise Reduction")
    noise_reduced_img = cv2.GaussianBlur(final_image, (3, 3), 0)
    
    # noise_reduced_img = cv2.medianBlur(final_image, 5)
    # time.sleep(0.5)

    # compare = np.concatenate((final_image, noise_reduced_img), axis=1)
    # cv2.imshow('final_image', compare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return noise_reduced_img
