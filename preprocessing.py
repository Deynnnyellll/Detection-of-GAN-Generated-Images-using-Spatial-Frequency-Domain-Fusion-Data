'''
This code is for preprocessing the image before
going to feature extraction phase

1. The image will be converted to grayscale
2. The image will be resize to 256x256 (as mentioned to the system architecture
   where images will be resize to a common size)
'''

import cv2
import numpy as np
import time

def preprocessing(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    dimension = np.array(image.shape)
    if dimension[0] == 256 and dimension[1] == 256:
        print("\n... \tImage is already in 256x256\n\n")
        final_image = image
    else:
        print("\nWarning: Image is not resized to 256x256")
        time.sleep(1)
        print('...')
        time.sleep(1)
        print("\nPerforming image resizing (256x256)")
        print('...')
        time.sleep(1)
        final_image = cv2.resize(image, dsize=(1024, 1024))

    #noise reduction using median filtering
    print("Applying Media Filter to Reduce Noise")
    noise_reduced_img = cv2.medianBlur(final_image, 5)
    time.sleep(2)

    compare = np.concatenate((final_image, noise_reduced_img), axis=1)
    cv2.imshow('final_image', compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return noise_reduced_img