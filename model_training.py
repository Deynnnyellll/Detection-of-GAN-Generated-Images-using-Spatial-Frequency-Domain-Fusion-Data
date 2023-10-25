import numpy as np

# Your CSV-like data as a multi-line string
csv_data = [[  0,   0,   0, ...,   0,   0,   0,],
 [  0, 255, 255, ...,   0,   0,   0,],
 [  0, 255, 255, ...,   0,  0,   0,],
 ...,
 [  0, 143, 143, ...,  95,  32,   0,],
 [  0, 255, 255, ...,  95,  32,   0,],
 [  0,   0,   0, ...,  95,  32,   0,]]

data_array = np.array(csv_data, dtype=np.uint8)

# Convert the NumPy array to a hexadecimal string
string_numpy = data_array.tobytes().hex()


print(string_numpy)


