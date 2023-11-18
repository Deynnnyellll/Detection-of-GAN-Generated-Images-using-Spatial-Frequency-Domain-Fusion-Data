from preprocessing import preprocessing
from liblinear.liblinearutil import train, predict, save_model, load_model
from train import spatial_frequency_feature_fusion
import numpy as np
from prob_estimates import calculate_prob, get_prob
import os

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


        print("------------------------------------------DISPLAYING THE RESULT-----------------------------------\n")
        result = []
        for i in predicted_labels:
            if i == 1.0:
                result.append("Real")
            elif i == 0.0:
                result.append("GAN")

        return feature_vector, result, likelihood
    except Exception as e:
        print(e)


# still in experimental since prob estimates is trained in another model
def linear_predict_proba(images, loaded_model, clf):
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
        _, _, scores = predict([], feature_vector, loaded_model)

        # get the probability estimates
        predicted_labels, likelihood = get_prob(scores, clf)

        print("------------------------------------------DISPLAYING THE RESULT-----------------------------------\n")
        result = []
        for i in predicted_labels:
            if i == 1.0:
                result.append("Real")
            elif i == 0.0:
                result.append("GAN")     
        return feature_vector, result, likelihood, scores
        print("Hi")
    except Exception as e:
        print(e)        

# application of incremental learning
def adapt(labels, feature_vector, svm_scores, model_file, clf_file): 
    try:
        # reshape true labels
        true_label = labels.reshape(labels.shape[0])

        # initialize problem and param
        model = train(true_label, feature_vector, f'-s 1 -c 1 -B 1 -i {model_file}')
        save_model("/Users/Danniel/Downloads/Model/Validate/faces_old_updated_liblinear.model", model)
        
        os.chdir("/Users/Danniel/Downloads/Model/Platt Scaling")
        platt_scale_file = os.path.basename(clf_file)

        plat = train(true_label, svm_scores, f'-s 0 -c 1 -B 1 -i {platt_scale_file}')
        save_model("/Users/Danniel/Downloads/Model/Platt Scaling/platt_scale_validate_updated_faces.model", plat)
        plat_updated = load_model("/Users/Danniel/Downloads/Model/Platt Scaling/platt_scale_validate_updated_faces.model")

        return model, plat_updated
    except:
        print("Does not support incremental learning")