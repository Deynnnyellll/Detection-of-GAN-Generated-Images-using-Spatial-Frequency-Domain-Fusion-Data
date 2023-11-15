from preprocessing import preprocessing
from liblinear.liblinearutil import predict
from train import spatial_frequency_feature_fusion
import numpy as np

# test the model
def predict(images, loaded_model):
    # preprocessing
    preprocessed_img = [preprocessing(i) for i in images] 

    # apply spatial frequency feature fusion to the preprocessed images
    fused_features = spatial_frequency_feature_fusion(preprocessed_img)

    feature_vector = [i.flatten() for i in fused_features]


    labels = np.ones((len(feature_vector), 1)) 
    true_label = labels.reshape(labels.shape[0])


    print(len(feature_vector))
    print(len(true_label))


    # predict the result
    print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
    predicted_labels, _, _ = predict([], feature_vector, loaded_model)


    print("------------------------------------------RESULT-----------------------------------\n")
    result = []
    for i in predicted_labels:
        if i == 1.0:
            result.append("Real")
        elif i == 0.0:
            result.append("GAN")

    return result