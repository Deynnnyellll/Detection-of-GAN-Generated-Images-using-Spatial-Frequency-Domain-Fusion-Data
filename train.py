'''
This code is for training GAN-Generated Images and Real Images

you need to install this package by running this command:
pip install liblinear-official
'''

import numpy as np
from liblinear.liblinearutil import problem, parameter, train
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from feature_fusion import concatenate_lbp_dwt
import os
import matplotlib.pyplot as plt
import cv2
from numba import jit

def get_data(directory):
    # load preprocessed images
    preprocessed_img = []

    for filename in os.listdir(directory):
        image = os.path.join(directory, filename)
        if image is not None:
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            preprocessed_img.append(img)

    print("Preprocessed Images: ", len(preprocessed_img))        

    return preprocessed_img


def spatial_frequency_feature_fusion(images):
    # feature extraction
    print("Performing Feature Extraction")
    # applying local binary pattern
    print("Applying Local Binary Pattern")
    lbp_img_features = []
    for i in images:
        texture_features = lbp(i)
        print('\n\n')
        print(texture_features)


        # store the features in a lbp_img_features list
        lbp_img_features.append(texture_features)
        print(f"\n{len(lbp_img_features)} out of {len(images)} images\nPercentage: {(float(len(lbp_img_features)) / float(len(images)) * 100)}\n")
    print('\nLBP application finished\n\n')
    
    # applying discrete wavelet transform
    print("Applying DWT to Images")

    dwt_img_features = []
    for i in images:
        freq_features = dwt_2d(i)
        print('\n\n')
        print(freq_features)

        # store the features in a dwt_img_features list
        dwt_img_features.append(freq_features)
        print(f"\n{len(dwt_img_features)} out of {len(images)} images\nPercentage: {(float(len(dwt_img_features)) / float(len(images)) * 100)}\n")
    print("\nDWT application finished\n\n")


    # applying feature fusion
    fused_features = []

    for dwt_features, lbp_features in zip(dwt_img_features, lbp_img_features):
        feature_vector = concatenate_lbp_dwt(lbp_features, dwt_features)
        print("Fused Features\n", feature_vector)
        fused_features.append(feature_vector)
        print(f"\n{len(fused_features)} out of {len(images)} images\nPercentage: {(float(len(fused_features)) / float(len(images)) * 100)}\n")

    return fused_features


def prepare_data(gan, real):
    print("----------------------------Preparing the Data-------------------------------\n") 
    #label real  and gan datasets
    gan_label = np.ones((len(gan), 1))
    real_label = np.zeros((len(real), 1))


    # combine the labels and datasets
    dataset_labels = np.vstack((gan_label, real_label))
    datasets = np.vstack((gan, real))

    # reshape the labels and datasets for svm requirements
    datasets_final = [i.flatten() for i in datasets]
    
    label_final = dataset_labels.reshape(dataset_labels.shape[0])

    print("Labels: ", len(label_final))
    print("Datasets: ", len(datasets_final))

    return label_final, datasets_final


def train_linear_model(label, datasets, C):
    print("----------------------Model Training in Liblinear--------------------------\n")

    # check if length of datasets is equal to the length of labels
    if len(label) == len(datasets):
        prob = problem(label, datasets)
        validate = parameter(f'-s 1 -c {C} -v 5 -B 1')
        param = parameter(f'-s 1 -c {C} -B 1')

        validation = train(prob, validate)

        print(validation)

        model = train(prob, param)
    
    else:
        print("Length of datasets and labels do not match\n")  
        print("Length of Datasets: ", len(datasets))
        print("Length of Labels: ", len(label))

    return model


def visualize(real, gan):

    mean1 = [np.mean(features) for features in real]
    mean2 = [np.mean(features) for features in gan]

    plt.plot(mean1, label="real", color="blue")
    plt.plot(mean2, label="gan", color="red")

    # Adding labels and a title
    plt.xlabel('Index')
    plt.ylabel('Mean Value')
    plt.title('Mean Values for Two Classes')


    # Display the plot
    plt.show()

    return mean1, mean2