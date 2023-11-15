from preprocessing import preprocessing
from liblinear.liblinearutil import predict
from train import spatial_frequency_feature_fusion
import numpy as np
from prob_estimates import calculate_prob, get_prob

# test the model
def linear_predict(images, loaded_model):
    # preprocessing
    preprocessed_img = [preprocessing(i) for i in images] 

    # apply spatial frequency feature fusion to the preprocessed images
    fused_features = spatial_frequency_feature_fusion(preprocessed_img)

    feature_vector = [i.flatten() for i in fused_features]

    print(feature_vector)


    labels = np.ones((len(feature_vector), 1)) 
    true_label = labels.reshape(labels.shape[0])


    try:
        # predict the result
        print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
        predicted_labels, _, prob_estimates = predict([], feature_vector, loaded_model)

        likelihood = [calculate_prob(i[0]) for i in prob_estimates]


        print("------------------------------------------RESULT-----------------------------------\n")
        result = []
        for i in predicted_labels:
            if i == 1.0:
                result.append("Real")
            elif i == 0.0:
                result.append("GAN")

        return result, likelihood
    except Exception as e:
        print(e)


# still in experimental since prob estimates is trained in another model
def linear_predict_proba(images, loaded_model):
    # preprocessing
    preprocessed_img = [preprocessing(i) for i in images] 

    # apply spatial frequency feature fusion to the preprocessed images
    fused_features = spatial_frequency_feature_fusion(preprocessed_img)

    feature_vector = [i.flatten() for i in fused_features]

    print(feature_vector)


    labels = np.ones((len(feature_vector), 1)) 
    true_label = labels.reshape(labels.shape[0])


    try:
        # predict the result
        print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
        predicted_labels, _, prob_estimates = predict([], feature_vector, loaded_model)
        likelihood = get_prob(feature_vector)


        print("------------------------------------------RESULT-----------------------------------\n")
        result = []
        for i in predicted_labels:
            if i == 1.0:
                result.append("Real")
            elif i == 0.0:
                result.append("GAN")
        return result, likelihood
    except Exception as e:
        print(e)        