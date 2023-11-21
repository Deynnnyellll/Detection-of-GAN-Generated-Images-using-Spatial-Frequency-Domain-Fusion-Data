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

    print("SVM Scores Before Incremental: ", scores)

    # get the probability estimates using the predicted svm scores of the svm classifier
    predicted_labels, likelihood = get_prob(scores, clf)

    # iterate over the "predicted_labels" list and convert each float value to equivalent label in string
    # in this case, it can either be Real or GAN
    result = ["Real" if i == 1.0 else "GAN" for i in predicted_labels]

    return result, likelihood

# application of incremental learning
def adapt(images, model_file, clf_file): 
    try:
        real = []
        gan = []
        real_labels = []
        gan_labels = []
        sorted_images = []

        for image in images:
            if "real" in image:
                real.append(image)
                label = np.ones(1)
                real_labels.append(label)
            elif "gan" in image:
                gan.append(image)
                label = np.zeros(1)
                gan_labels.append(label)

        sorted_images.extend(real)
        sorted_images.extend(gan)
            
        # preprocessing
        preprocessed_img = [preprocessing(image) for image in sorted_images]

        # apply spatial frequency feature fusion to the preprocessed images
        fused_features = spatial_frequency_feature_fusion(preprocessed_img)

        feature_vector = [feature.flatten() for feature in fused_features]

        labels = np.vstack((real_labels, gan_labels))
        true_labels = labels.reshape(labels.shape[0])

        # incremental learning of svm
        model = train(true_labels, feature_vector, f'-s 1 -c 1 -B 1 -i {model_file}')
        save_model('faces_updated.model', model)

            # predict new value and get the svm scores to add in the platt scaler
        _, _, svm_scores = predict(true_labels, feature_vector, model)

        print("SVM Scores after Incremental Learning: ", svm_scores)

        os.chdir("/Users/Danniel/Detection-of-GAN-Generated-Images-using-Spatial-Frequency-Domain-Fusion-Data/platt scaler")
        clf_file_final = os.path.basename(clf_file)

        # incremental learning of platt scaler
        plat = train(true_labels, svm_scores, f'-s 0 -c 0.1 -B 1 -i {clf_file_final}')
        save_model('platt_updated.model', plat)
        
        predicted_labels, _, prob_estimates = predict([], svm_scores, plat, '-b 1')



        return model, plat
    except Exception as e:
        print(f"Incremental Learning Error: {e}")
