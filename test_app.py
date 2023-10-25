import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from test import concatenate_lbp_dwt
import os


def predict(directory):
    images = []

    # load the images and store in images list
    for filename in os.listdir(directory):
        image = os.path.join(directory, filename)
        if image is not None:
            images.append(image)

    print(len(images))        


    # load the model
    model_file = "/Users/Danniel/Downloads/faces.txt"
    loaded_model = svm_load_model(model_file)

    # preprocessing
    preprocessed_img = []
    for i in images:
        preprocessed_img.append(preprocessing(i))      


    #  discrete wavelet transform
    dwt_feature = []
    for i in preprocessed_img:
        dwt_feature.append(dwt_2d(i))


    # local binary pattern
    lbp_feature = []
    for i in preprocessed_img:
        lbp_feature.append(lbp(i))


    # feature fusion
    fused_vector = []
    for frequency, texture in zip(dwt_feature, lbp_feature):
        fused_vector.append(concatenate_lbp_dwt(texture, frequency))

    print("\n\n",fused_vector)

    # flatten the feature vector
    feature_vector = []
    for i in fused_vector:
        feature_vector.append(i.flatten())


    # predict the result
    predicted_labels, _, _ = svm_predict([], feature_vector, loaded_model, '-q')

    print("------------------------------------------RESULT-----------------------------------\n")
    result = []
    for i in predicted_labels:
        if i == 1.0:
            result.append("Real")
        elif i == 0.0:
            result.append("GAN")

    print(result)

#provide directory for testing dataset
dir = ""

predict(dir)
