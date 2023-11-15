import numpy as np
import pickle

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

# prob estimates implementation using calibrated classifier
def get_prob(data):
    with open("/Users/Danniel/Downloads/Model/calibrated_classifier.pkl", 'rb') as model_file:
        clf = pickle.load(model_file)

    try:
        y_prob = clf.predict_proba(data)
        return y_prob
    except Exception as e:
        print(e)        