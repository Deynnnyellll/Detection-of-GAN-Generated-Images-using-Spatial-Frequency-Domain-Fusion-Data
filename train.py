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


#test code

sample_img = "1.jpg"
preprocessed_img = preprocessing(sample_img)

print("\nApplying 2D DWT")
dwt_image = dwt_2d(preprocessed_img)
print("High frequency components obtained")
print(dwt_image)
time.sleep(2)
print("\n....\n")
lbp_image = lbp(preprocessed_img)
print(np.array(lbp_image[1]))

concatenated_feature = np.concatenate((dwt_image, lbp_image[1]), axis=1)
print("\nConcatenated Feature\n")
print(concatenated_feature)