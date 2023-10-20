import numpy as np
from libsvm.svmutil import svm_load_model, svm_predict
from preprocessing import preprocessing
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from test import concatenate_lbp_dwt

model_file = "faces.model"
loaded_model = svm_load_model(model_file)


image = "/Users/Danniel/Documents/datasets/real/12458.png"
image1 = preprocessing(image)
dwt_image = dwt_2d(image1)
lbp_image = lbp(image1)
fuse = concatenate_lbp_dwt(lbp_image[1], dwt_image)
print("Before Reshape:\n", fuse)
final = fuse.reshape(fuse.shape[0], -1)
print("After Reshape:\n", final)


predicted_labels, _, _ = svm_predict([], final, loaded_model)
print(predicted_labels)

if 0 in predicted_labels:
    print("GAN")
else:
    print("Real")