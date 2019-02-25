import numpy as np
import math
from math import ceil

np.set_printoptions(suppress=True)

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

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

        print("p", len(p))
        print("n", len(n))


        subset_len = len(p) + len(n)
        print("subset_len", subset_len)

        entropy = 0

        if subset_len:
            entropy = entropy_helper(len(p)/subset_len) + entropy_helper(len(n)/subset_len)

        avg_entropy += (subset_len / total) * entropy

    return avg_entropy

def choose_best(observations, features):
    spam = observations[1]
    not_spam = observations[0]

    node = (1, None, [], [])
    for i in features:
        spam_features = spam[:, i]
        not_spam_features = not_spam[:, i]

        #create subsets
        spam_subsets = [ [], [] ]
        for j, feature in enumerate(spam_features):
            if feature < 0:
                spam_subsets[0].append(spam[j])
            else:
                spam_subsets[1].append(spam[j])

        not_spam_subsets = [ [], [] ]
        for j, feature in enumerate(not_spam_features):
            if feature < 0:
                not_spam_subsets[0].append(not_spam[j])
            else:
                not_spam_subsets[1].append(not_spam[j])


        spam_subsets = np.array(spam_subsets)
        not_spam_subsets = np.array(not_spam_subsets)

        total = len(spam_features) + len(not_spam_features)
        print("I", i)
        feature_entropy = calculate_entropy(spam_subsets, not_spam_subsets, total)

        if feature_entropy < node[0]:
            node = (feature_entropy, i, spam_subsets, not_spam_subsets)

    return node


def dtl(observations, features, default='Spam'):
    print("features", features)
    print("observations0", len(observations[0]))
    print("observations1", len(observations[1]))

    if not len(observations):
        return default
    if not len(observations[0]):
        return 1
    if not len(observations[1]):
        return 0
    if not len(features):
        return 1

    best_attribute = choose_best(observations, features)
    feature_ind = best_attribute[1]

    print(feature_ind)
    tree = Node(feature_ind)
    features.remove(feature_ind)

    spam_subsets, not_spam_subsets = best_attribute[2], best_attribute[3]

    p0 = spam_subsets[0]
    p1 = spam_subsets[1]
    n0 = not_spam_subsets[0]
    n1 = not_spam_subsets[1]

    #print("\n", feature_ind, "\n")
    #print("spam0", len(p0))
    #print("spam1", len(p1))
    #print("not_spam0", len(n0))
    #print("not_spam1", len(n1))

    total_with_feature = len(p1) + len(n1)
    total_without_feature = len(p0) + len(n0)

    #print("total_with_feature", total_with_feature)
    #print("total_without_feature", total_without_feature)

    for i in range(0, 2):
        spam = np.array(spam_subsets[i])
        not_spam = np.array(not_spam_subsets[i])

        observations = [not_spam, spam]

        return dtl(observations, features)

    return "shit"


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

    observations = [not_spam, spam]

    dtl(observations, list(range(0,57)))

if __name__ == '__main__':
    main()
