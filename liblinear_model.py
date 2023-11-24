from preprocessing import preprocessing
from liblinear.liblinearutil import train, predict, load_model, save_model
from train import spatial_frequency_feature_fusion
from prob_estimates import get_prob
import numpy as np
import os

# predict labels calibrated with probability estimates
def linear_predict_proba(images, loaded_model, clf):
    # preprocessing
    preprocessed_img = [preprocessing(image) for image in images]

    # apply spatial frequency feature fusion to the preprocessed images
    fused_features = spatial_frequency_feature_fusion(preprocessed_img)

    feature_vector = [feature.flatten() for feature in fused_features]

    # predict the result
    print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
    _, _, scores = predict([], feature_vector, loaded_model)


    # get the probability estimates using the predicted svm scores of the svm classifier
    predicted_labels, likelihood = get_prob(scores, clf)

    # iterate over the "predicted_labels" list and convert each float value to equivalent label in string
    # in this case, it can either be Real or GAN
    result = ["GAN" if i == 1.0 else "Real" for i in predicted_labels]

    return result, likelihood
