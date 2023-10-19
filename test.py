import numpy as np
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from preprocessing import preprocessing
import matplotlib.pyplot as plt



def concatenate_lbp_dwt(lbp_values, dwt_features):
  """Concatenates the LBP and DWT values into a single fused vector.

  Args:
    lbp_values: A numpy array containing the LBP values.
    dwt_features: A numpy array containing the DWT features.

  Returns:
    A numpy array containing the fused vector.
  """


  # Concatenate the flattened arrays.
  fused_vector = np.concatenate((lbp_values, dwt_features), axis=1)

  return fused_vector
