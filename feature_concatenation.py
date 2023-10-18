import numpy as np
from discrete_wavelet_transform import dwt_2d
from local_binary_pattern import lbp
from preprocessing import preprocessing



def concatenate_lbp_dwt(lbp_values, dwt_features):
  """Concatenates the LBP and DWT values into a single fused vector.

  Args:
    lbp_values: A numpy array containing the LBP values.
    dwt_features: A numpy array containing the DWT features.

  Returns:
    A numpy array containing the fused vector.
  """

  # Flatten the LBP and DWT arrays.
  lbp_values_flat = (lbp_values[1]).flatten()
  dwt_features_flat = dwt_features.flatten()

  # Concatenate the flattened arrays.
  fused_vector = np.concatenate([lbp_values_flat, dwt_features_flat])

  return fused_vector

image = '1.jpg'
image1 = preprocessing(image)
dwt_features = dwt_2d(image1)
lbp_values = lbp(image1)


fused_vector = concatenate_lbp_dwt(np.array(lbp_values[1]), np.array(dwt_features))


print('Fused vector value:')
print(fused_vector)