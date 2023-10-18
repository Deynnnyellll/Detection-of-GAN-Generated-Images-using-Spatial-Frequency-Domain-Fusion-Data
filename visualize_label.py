import numpy as np
import matplotlib.pyplot as plt

# Assuming three datasets with random feature vectors
data1 = np.random.rand(20, 2)  # 20 feature vectors for dataset 1
data2 = np.random.rand(20, 2) + 1  # 20 feature vectors for dataset 2
data3 = np.random.rand(20, 2) + 2  # 20 feature vectors for dataset 3

# Assign labels to each dataset
data1_labels = np.zeros(len(data1))  # Label 0 for dataset 1
data2_labels = np.ones(len(data2))   # Label 1 for dataset 2
data3_labels = 2 * np.ones(len(data3))  # Label 2 for dataset 3

# Concatenate features and labels
features = np.vstack((data1, data2, data3))
labels = np.hstack((data1_labels, data2_labels, data3_labels))

# Plot the feature vectors with different colors for each label
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Visualization of Concatenated Feature Vectors with Labels')
plt.colorbar(label='Assigned Label')
plt.show()
