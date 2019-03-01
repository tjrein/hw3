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
    cutoff = data.shape[1] - 1
    targets = data[:, cutoff:]
    features = data[:, :cutoff]
    return (targets, features)


def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    return features

def handle_data(data):
    np.random.seed(2)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    testing = data[range:]

    return isolate_sets(training, testing)

def filter_low_std(data):
    std = data.std(axis=0, ddof=1)
    remove = [ i for i, val in enumerate(std) if val < 0.1 ]
    data = np.delete(data, remove, 1)

    return data
