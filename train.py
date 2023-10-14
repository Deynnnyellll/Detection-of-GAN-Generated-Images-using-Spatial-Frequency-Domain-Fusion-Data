'''
This code is for training GAN-Generated Images and Real Images

you need to install this package by running this command:
pip install -U libsvm-official
'''
import numpy as np
import scipy
from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_predict
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
import time
import cv2
import os


def train(directory):
    # store multiple images into lis
    images = []

    for filename in os.listdir(directory):
        image = os.path.join(directory, filename)
        if image is not None:
            images.append(image)

    # preprocessing
    print("\nPreprocessing the images")
    preprocessed_img = []
    for i in images:
        img = preprocessing(i)

        preprocessed_img.append(img)

    print("Number of Images: ", len(preprocessed_img))    

    # feature extraction
    print("Performing Feature Extraction")
    time.sleep(1)

    # applying discrete wavelet transform
    print("Applying DWT to Images")

    dwt_img_features = []
    for i in preprocessed_img:
        freq_features = dwt_2d(i)
        print('\n\n')
        print(freq_features)

        # store the features in a dwt_img_features list
        dwt_img_features.append(freq_features)
    print("\nDWT application finished\n\n")
    time.sleep(1)

    # applying local binary pattern
    print("Applying Local Binary Pattern")
    lbp_img_features = []
    for i in preprocessed_img:
        texture_features = lbp(i)
        print('\n\n')
        print(texture_features[1])

        # store the features in a lbp_img_features list
        lbp_img_features.append(texture_features[1])
    print('\nLBP application finished\n\n')
    time.sleep(1)


    # svm implementation

directory = "/Users/Danniel/Downloads/FFHQ"
train(directory) 