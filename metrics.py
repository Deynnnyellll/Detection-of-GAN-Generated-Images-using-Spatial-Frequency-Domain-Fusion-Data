# Assign values to TP, TN, FP, FN variables
TP = 187 
TN = 294  
FP = 6 
FN = 113 

# Calculate Precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0

# Calculate Recall
recall = TP / (TP + FN) if (TP + FN) != 0 else 0

# Calculate F1 Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)
