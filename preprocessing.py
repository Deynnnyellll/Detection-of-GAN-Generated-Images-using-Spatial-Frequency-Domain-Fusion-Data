import cv2
import numpy as np
import time

def preprocessing(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    dimension = np.array(image.shape)

    if dimension[0] == 256 and dimension[1] == 256:
        print("\n... tImage is already in 256x256\n\n")
        final_image = image
    else:
        print("\nWarning: Image is not resized to 256x256")
        time.sleep(1)
        print('...')
        time.sleep(1)
        print("\nPerforming image resizing (256x256)\n\n")
        print('...')
        time.sleep(1)
        final_image = cv2.resize(image, (256, 256))

    time.sleep(2)
    return final_image