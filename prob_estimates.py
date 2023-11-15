import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_prob(decision_value):
    probability = sigmoid(decision_value)

    probA = probability
    probB = 1 - probA

    return probA, probB