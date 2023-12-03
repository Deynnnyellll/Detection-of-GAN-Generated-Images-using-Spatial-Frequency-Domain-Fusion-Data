'''
This code is for preprocessing the image before
going to feature extraction phase

1. The image will be converted to grayscale
2. The image will be resize to 512x512 (as mentioned in the system architecture)
3. The image will have a noise reduction using Gaussian Noise Reduction also known as Gaussian blur or Gaussian smoothing

'''

import cv2
import numpy as np

def preprocessing(image):
    # Load the image in grayscale
    image = cv2.imread(image)

    # Check the dimensions
    dimension = np.array(image.shape)
    if dimension[0] == 512 and dimension[1] == 512:
        print("\n\nImage is already in 512x512\n\n")
        resized_image = image
    else:
        print("\n\nWarning: Image is not resized")
        print('...')
        print("\nPerforming image resizing")
        print('...')
        resized_image = cv2.resize(image, dsize=(512, 512))

    # Grayscale Conversion
    print("\nConverting to grayscale\n")
    grayscaled_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)    

    # Noise reduction using Gaussian Blurring
    print("\nApplying Gaussian Noise Reduction")
    noise_reduced_img = cv2.GaussianBlur(grayscaled_image, (3, 3), 0)

    return noise_reduced_img
