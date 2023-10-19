'''
This code is for training GAN-Generated Images and Real Images

you need to install this package by running this command:
pip install -U libsvm-official
'''
import numpy as np
import scipy
from libsvm.svmutil import svm_problem, svm_parameter, svm_train, svm_save_model
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from test import concatenate_lbp_dwt
import time
import os


def get_data(directory):
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

    # applying feature fusion
    fused_features = []

    for dwt_features, lbp_features in zip(dwt_img_features, lbp_img_features):
        feature_vector = concatenate_lbp_dwt(lbp_features, dwt_features)
        fused_features.append(feature_vector)

    return fused_features


def prepare_data(real, gan):
    print("----------------------------Preparing the Data-------------------------------\n")  

    #label real  and gan datasets
    real_label = np.ones((len(real), 1))
    gan_label = np.zeros((len(gan), 1))


    # combine the labels and datasets
    dataset_labels = np.vstack((real_label, gan_label))
    print(len(dataset_labels))
    datasets = np.vstack((real, gan))

    # reshape the labels and datasets for svm requirements
    datasets_final = datasets.reshape(datasets.shape[0], -1)
    label_final = dataset_labels.reshape(dataset_labels.shape[0])



    print("----------------------Training Datasets--------------------------\n")

    # initialize parameter
    kernel_type = 2 #rbf (ginawa kong rbf muna same sa gan synthesized na study)
    C = 1.0

    # check if length of datasets is equal to the length of labels
    if len(label_final) == len(datasets_final):
        prob = svm_problem(label_final, datasets_final)
        param = svm_parameter(f'-t {kernel_type} -c {C}')
        model = svm_train(prob, param)
        
        return model
    
    else:
        print("Length of datasets and labels do not match\n")  
        print("Length of Datasets: ", len(datasets))
        print("Lenght of Labels: ", len(dataset_labels))



# provide directory for real and gan 
real_directory = "/Users/Danniel/Documents/datasets/real"
gan_directory = "/Users/Danniel/Documents/datasets/gan"


# run data preparation
real_data = get_data(real_directory)
gan_data = get_data(gan_directory)

# train the data
model = prepare_data(real_data, gan_data)

#save the model
model_file = "faces.model"
svm_save_model(model_file, model)
