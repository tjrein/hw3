import numpy as np
from math import ceil

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

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',', dtype="uint16")
    train_x, train_y, test_x, test_y  = handle_data(data)

    spam = []
    not_spam = []
    for i, obs in enumerate(train_x):
        if train_y[i] == 1:
            not_spam.append(obs)
        else:
            spam.append(obs)


if __name__ == '__main__':
    main()

