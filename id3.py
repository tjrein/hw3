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

    return -(frac) * math.log(frac, 2)

def calculate_entropy(spam_subsets, not_spam_subsets, total, k=2):
    avg_entropy = 0

    for i in range(0, k):
        p = spam_subsets[i]
        n = not_spam_subsets[i]

        subset_len = len(p) + len(n)
        entropy = entropy_helper(len(p)/subset_len) + entropy_helper(len(n)/subset_len)
        avg_entropy += (subset_len / total) * entropy

    return avg_entropy


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


    #entropy calculation test
    #spam_subsets = [ [11, 13], [22], []]
    #not_spam_subsets = [ [12], [22], [32]]

    #calculate_entropy(spam_subsets, not_spam_subsets, 6, k=)

    node = (1, None, [], [])
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
        feature_entropy = calculate_entropy(spam_subsets, not_spam_subsets, total)

        if feature_entropy < node[0]:
            p0 = spam_subsets[0]
            p1 = spam_subsets[1]
            n0 = not_spam_subsets[0]
            n1 = not_spam_subsets[1]

            print("\n", i, "\n")
            print("spam0", len(p0))
            print("spam1", len(p1))
            print("not_spam0", len(n0))
            print("not_spam1", len(n1))

            total_with_feature = len(p1) + len(n1)
            total_without_feature = len(p0) + len(n0)

            print("total_with_feature", total_with_feature)
            print("total_without_feature", total_without_feature)
            print("total samples", len(train_x))


            node = (feature_entropy, i, spam_subsets, not_spam_subsets)

    root_node = node




if __name__ == '__main__':
    main()
