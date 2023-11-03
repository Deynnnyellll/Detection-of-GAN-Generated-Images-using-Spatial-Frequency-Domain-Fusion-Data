import numpy as np
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp



def concatenate_lbp_dwt(lbp_values, dwt_features):
  # Concatenate the flattened arrays.
  fused_vector = np.concatenate((lbp_values, dwt_features), axis=1)

  return fused_vector
