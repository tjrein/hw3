import numpy as np
import math
import random
from math import ceil

np.set_printoptions(suppress=True)

class Node(object):
    def __init__(self, data):
        self.data = data
        self.left_child = None
        self.right_child = None

    def add_left_child(self, obj):
        self.left_child = obj

    def add_right_child(self, obj):
        self.right_child = obj

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
        feature_entropy = calculate_entropy(spam_subsets, not_spam_subsets, total)

        if feature_entropy <= node[0]:
            node = (feature_entropy, i, spam_subsets, not_spam_subsets)

    return node

def generate_guess(observations):
    not_spam = len(observations[0])
    spam = len(observations[1])
    percent_spam = spam / (not_spam + spam)
    percent_not_spam = not_spam / (not_spam + spam)
    #random.seed(0)
    choice = random.choices(population=[0, 1], weights=[percent_not_spam, percent_spam], k=1)[0]
    return choice

def dtl(observations, features, default=0):
    if not len(observations):
        return Node(default)
    if not len(observations[0]):
        return Node(1)
    if not len(observations[1]):
        return Node(0)
    if not len(features):
        if len(observations[0]) > len(observations[1]):
            return Node(0)
        else:
            return Node(1)

    best_attribute = choose_best(observations, features)
    feature_ind = best_attribute[1]

    tree = Node(feature_ind)

    spam_subsets, not_spam_subsets = best_attribute[2], best_attribute[3]

    p0 = spam_subsets[0]
    p1 = spam_subsets[1]
    n0 = not_spam_subsets[0]
    n1 = not_spam_subsets[1]

    total_with_feature = len(p1) + len(n1)
    total_without_feature = len(p0) + len(n0)

    for i in (0, 1):
        new_features = features.copy()
        new_features.remove(feature_ind)
        spam = np.array(spam_subsets[i])
        not_spam = np.array(not_spam_subsets[i])

        observations = [not_spam, spam]

        if i == 0:
            tree.add_left_child(dtl(observations, new_features))
        else:
            tree.add_right_child(dtl(observations, new_features))

    return tree

def traverse_tree(tree, obs):
    if tree.left_child is None and tree.right_child is None:
        return tree.data

    if obs[tree.data] < 0:
        return traverse_tree(tree.left_child, obs)
    else:
        return traverse_tree(tree.right_child, obs)

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
    tree = dtl(observations, list(range(0,57)))

    labels = []
    for i, obs in enumerate(test_x):
        labels.append(traverse_tree(tree, obs))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    labels = np.array(labels)
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

    print("tp", tp)
    print("tn", tn)
    print("fp", fp)
    print("fn", fn)

    print ("precision", tp / (tp + fp))
    print ("recall", tp / (tp + fn))
    print ((tp + tn) / (tp + tn + fp + fn))

if __name__ == '__main__':
    main()
