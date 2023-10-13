'''
This code is for feature fusion
The features obtained from dwt and lbp
will be fused using feature concatenation
'''

import numpy as np


def feature_fusion(frequency_spec, texture_desc):
    fused_features = np.concatenate(frequency_spec, texture_desc)