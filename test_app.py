import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from test import concatenate_lbp_dwt


def predict(image):
    # load the model
    model_file = "faces.model"
    loaded_model = svm_load_model(model_file)

    preprocessed_img = preprocessing(image)
    dwt_feature = dwt_2d(preprocessed_img)
    lbp_feature = lbp(preprocessed_img)
    fused_vector = concatenate_lbp_dwt(lbp_feature[1], dwt_feature)
    print("\n\n",fused_vector)
    flatten_img = fused_vector.flatten()
    feature_vector = []
    feature_vector.append(flatten_img)

    # predict the result
    predicted_labels, _, _ = svm_predict([], feature_vector, loaded_model)

    print("------------------------------------------RESULT-----------------------------------\n")
    if predicted_labels[0] == 1.0:
        print("Real")
    else: 
        print("GAN")