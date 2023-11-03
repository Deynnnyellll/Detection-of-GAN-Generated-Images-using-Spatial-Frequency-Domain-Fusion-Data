'''
Save preprocessed images to a specific directory

'''

import cv2
import numpy as np
import os
import time
import sys

# path of the folder "Detection-of-GAN-Generated-Images-using-Spatial-Frequency-Domain-Fusion-Data"
sys.path.append("/Users/Danniel/Detection-of-GAN-Generated-Images-using-Spatial-Frequency-Domain-Fusion-Data")
from preprocessing import preprocessing

def load_image(directory):
    images = []

    for filename in os.listdir(directory):
        image = os.path.join(directory, filename)
        if image is not None:
            preprocessed_images = preprocessing(image)
            images.append(preprocessed_images)
            print(f"\n{len(images)} finished\n")

    print("Number of Images: ",len(images))

    return images     

def save_image(image_list, directory):
    for index, image in enumerate(image_list):
        save_name = f"{index+1}_preprocessed.png"
        cv2.imwrite(f"{directory}/{save_name}", image)
        print(save_name)

    if directory is not None:
        print(f"\n------------------{len(image_list)} Images converted successfully-------------------\n")    


# directory for reading
img_real = "/Users/Danniel/Downloads/real"
img_gan = "/Users/Danniel/Downloads/gan"

# directory for saving (must be an empty folder)
save_dir_real = "/Users/Danniel/Downloads/preprocessed_real"
save_dir_gan = "/Users/Danniel/Downloads/preprocessed_gan"

# store the images
image_real = load_image(img_real)
image_gan = load_image(img_gan)

save_image(image_real, save_dir_real)
save_image(image_gan, save_dir_gan)