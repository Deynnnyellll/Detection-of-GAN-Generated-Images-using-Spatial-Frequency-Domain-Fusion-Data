from libsvm.svmutil import svm_predict
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from feature_fusion import concatenate_lbp_dwt

def predict(images, loaded_model):
    # preprocessing
    preprocessed_img = []
    for i in images:
        preprocessed_img.append(preprocessing(i))      


    #  discrete wavelet transform
    dwt_feature = []
    print("\n\n-------------------DWT----------------------------\n")
    for i in preprocessed_img:
        dwt_feature.append(dwt_2d(i))
        print(f"\n{len(dwt_feature)} out of {len(preprocessed_img)} images\nPercentage: {(float(len(dwt_feature)) / float(len(preprocessed_img)) * 100)}\n")


    # local binary pattern
    lbp_feature = []
    print("\n\n-------------------LBP----------------------------\n")
    for i in preprocessed_img:
        print("LBP Features:\n", lbp(i))
        print("\n")
        lbp_feature.append(lbp(i))
        print(f"\n{len(lbp_feature)} out of {len(preprocessed_img)} images\nPercentage: {(float(len(lbp_feature)) / float(len(preprocessed_img)) * 100)}\n")


    # feature fusion
    fused_vector = []
    print("\n\n-------------------FEATURE FUSION----------------------------\n")
    for frequency, texture in zip(dwt_feature, lbp_feature):
        print("Fused Features:\n", concatenate_lbp_dwt(texture, frequency))
        print("\n")
        fused_vector.append(concatenate_lbp_dwt(texture, frequency))
        print(f"\n{len(fused_vector)} out of {len(images)} images\nPercentage: {(float(len(fused_vector)) / float(len(preprocessed_img)) * 100)}\n")

    # flatten the feature vector
    feature_vector = []
    for i in fused_vector:
        feature_vector.append(i.flatten())


    # predict the result
    print("\n\n-------------------THE MODEL IS PREDICTING----------------------------\n")
    predicted_labels, _, _ = svm_predict([], feature_vector, loaded_model, '-q')



    print("------------------------------------------RESULT-----------------------------------\n")
    result = []
    for i in predicted_labels:
        if i == 1.0:
            result.append("Real")
        else:
            result.append("GAN")    

    return result