import numpy as np

def concatenate_lbp_dwt(lbp_values, dwt_features):
  # Concatenate the flattened arrays.
  fused_vector = np.concatenate((lbp_values, dwt_features), axis=1)

  return fused_vector
