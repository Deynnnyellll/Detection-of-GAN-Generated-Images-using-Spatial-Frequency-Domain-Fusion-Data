#initial implementation of DWT using pywt library

import numpy as np
import pywt

def dwt(image):
    wavelet_coeff = pywt.dwt2(image, wavelet='haar')

    frequency_features = []

    for subband in wavelet_coeff:
        mean = np.mean(subband)
        std = np.mean(subband)

        frequency_features.append(mean)
        frequency_features.append(std)

    return frequency_features