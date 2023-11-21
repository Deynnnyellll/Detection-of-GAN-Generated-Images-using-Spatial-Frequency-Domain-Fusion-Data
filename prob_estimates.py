from liblinear.liblinearutil import predict

# prob estimates
def get_prob(scores, clf):
    try:
        predicted_labels, _, y_prob = predict([], scores, clf, '-b 1')
        return predicted_labels, y_prob
    except Exception as e:
        print(e)        