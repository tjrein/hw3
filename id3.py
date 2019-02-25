import numpy as np
import math
from math import ceil

np.set_printoptions(suppress=True)

def isolate_sets(training, testing):
    train_y, train_x = separate_data(training)
    test_y, test_x = separate_data(testing)

    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0, ddof=1)

    train_x = standardize(train_x, mean, std)
    test_x = standardize(test_x, mean, std)

    return (train_x, train_y, test_x, test_y)

def separate_data(data):
    targets = data[:, 57:]
    features = data[:, :57]
    return (targets, features)

def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    return features

def handle_data(data):
    np.random.seed(0)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    testing = data[range:]

    return isolate_sets(training, testing)

def choose_best(features, labels, used=None):
    e0 = { 'pos': 0, 'neg': 0 }
    e1 = { 'pos': 0, 'neg': 0 }

    for i, feature in enumerate(features):
        if feature < 0:
            if labels[i] == 1:
                e0['pos'] += 1
            else:
                e1['neg'] += 1
        else:
            if labels[i] == 1:
                e1['pos'] += 1
            else:
                e1['neg'] += 1

    print("e0", e0)
    print("e1", e1)


def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    train_x, train_y, test_x, test_y  = handle_data(data)

    x = np.array([["green", 2, True], ["yellow", 4, True]])

    choose_best(train_x.T[0], train_y)



if __name__ == '__main__':
    main()
