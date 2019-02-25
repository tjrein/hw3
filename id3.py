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

def entropy_helper(frac):
    if frac == 0:
        return 0
    return -frac * math.log(frac)

def calculate_entropy(spam_subsets, not_spam_subsets, total):
    avg_entropy = 0

    for i in range(0, 2):
        p = spam_subsets[i]
        n = not_spam_subsets[i]

        subset_len = len(p) + len(n)

        entropy = entropy_helper(len(p)/subset_len) + entropy_helper(len(n)/subset_len)
        avg_entropy += subset_len / total * entropy

    print(avg_entropy)


def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    train_x, train_y, test_x, test_y  = handle_data(data)

    spam = []
    not_spam = []
    for i, obs in enumerate(train_x):
        if train_y[i] == 1:
            spam.append(obs)
        else:
            not_spam.append(obs)

    spam = np.array(spam)
    not_spam = np.array(not_spam)

    print(spam.shape)

    for i in range(0, train_x.shape[1]):
        spam_features = spam[:, i]
        not_spam_features = not_spam[:, i]

        #create subsets
        spam_subsets = [ [], [] ]
        for feature in spam_features:
            if feature < 0:
                spam_subsets[0].append(feature)
            else:
                spam_subsets[1].append(feature)

        not_spam_subsets = [ [], [] ]
        for feature in not_spam_features:
            if feature < 0:
                not_spam_subsets[0].append(feature)
            else:
                not_spam_subsets[1].append(feature)

        total = len(spam_features) + len(not_spam_features)
        calculate_entropy(spam_subsets, not_spam_subsets, total)


        #return
    #for i, features in enumerate(train_x.T):
    #    choose_best(features, train_y)
    #    return
        #for
    #choose_best(train_x.T[4], train_y)



if __name__ == '__main__':
    main()
