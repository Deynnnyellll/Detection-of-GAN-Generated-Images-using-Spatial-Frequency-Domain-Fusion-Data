import numpy as np

def concatenate_lbp_dwt(lbp_values, dwt_features):
  # Concatenate the flattened arrays.
  # w1 = 0.5
  # w2 = 0.5

  # normalized_dwt = (dwt_features - np.min(dwt_features)) / (np.max(dwt_features) - np.min(dwt_features))
  # normalized_lbp = (lbp_values - np.min(lbp_values)) / (np.max(lbp_values) - np.min(lbp_values))
  fused_vector = np.concatenate((lbp_values, dwt_features), axis=1)

  # fused_vector = w1 * normalized_dwt + w2 * normalized_lbp

  return fused_vector
