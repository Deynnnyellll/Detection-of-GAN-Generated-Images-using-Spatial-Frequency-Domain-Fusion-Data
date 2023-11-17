import numpy as np
from liblinear.liblinearutil import predict

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_prob(decision_value):
    try:
        probability = sigmoid(decision_value)

        probA = probability
        probB = 1 - probA

        return probA, probB
    except Exception as e:
        print(e)

# prob estimates
def get_prob(scores, clf):
    try:
        a, _, y_prob = predict([], scores, clf, '-b 1')
        return y_prob
    except Exception as e:
        print(e)        