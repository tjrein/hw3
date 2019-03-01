import numpy as np
from bayes_functions import classify, train
from data_operations import filter_low_std, handle_data

def display_performance(labels, test_y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x, prediction in enumerate(labels):
        target = test_y[x]

        if prediction == 1 and target == 1:
            tp += 1
        if prediction == 1 and target == 0:
            fp += 1
        if prediction == 0 and target == 0:
            tn += 1
        if prediction == 0 and target == 1:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print ("Precision:", precision)
    print ("Recall:", recall)
    print ("F-measure:", (2 * precision * recall) / (precision + recall))
    print ("Accuracy:", (tp + tn) / (tp + tn + fp + fn))

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_performance(labels, test_y)

if __name__ == "__main__":
    main()
