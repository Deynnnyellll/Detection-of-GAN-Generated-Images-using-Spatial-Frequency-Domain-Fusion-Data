from preprocessing import preprocessing
from libsvm.svmutil import svm_predict
from train import spatial_frequency_feature_fusion

# test the model
def predict(images, loaded_model):
    # preprocessing
    preprocessed_img = []
    for i in images:
        preprocessed_img.append(preprocessing(i))   

    # flatten the feature vector
    fused_features = spatial_frequency_feature_fusion(preprocessed_img)

    feature_vector = []
    for i in fused_features:
        feature_vector.append(i.flatten())

    print(feature_vector)


    # predict the result
    print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
    predicted_labels, _, likelihood = svm_predict([], feature_vector, loaded_model, '-q')


    print("------------------------------------------RESULT-----------------------------------\n")
    result = []
    for i in predicted_labels:
        if i == 1.0:
            result.append("Real")
        elif i == 0.0:
            result.append("GAN")

    # print(likelihood)
    print(result)