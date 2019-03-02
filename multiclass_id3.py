import numpy as np
import math
import random
from math import ceil
from data_operations import handle_data, filter_low_std

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

def entropy_helper(frac):
    if frac == 0:
        return 0
    return -(frac) * math.log(frac, 2)

def calculate_entropy_multi(subsets, total):
    avg_entropy = 0

    for letter in subsets:
        branch = subsets[letter]
        subset_len = 0
        entropy = 0

        for key, val in branch.items():
            subset_len += len(val)

        if subset_len:
            for key, val in branch.items():
                entropy += entropy_helper(len(val) / subset_len)

        avg_entropy += (subset_len / total ) * entropy
    return avg_entropy

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

def initialize_subsets(groups):
    subsets = { 'T': {}, 'F': {}}

    for key in groups:
        subsets['T'][key] = []
        subsets['F'][key] = []

    return subsets


def choose_best_multi(groups, feature_list):
    node = (1, None, {})
    for i in feature_list:
        subsets = initialize_subsets(groups)
        total = 0
        for key, val in groups.items():
            class_features = val[:, i]
            for j, feature in enumerate(class_features):
                if feature < 0:
                    subsets['F'][key].append(val[j])
                else:
                    subsets['T'][key].append(val[j])

            total += len(class_features)
            subsets['F'][key] = np.array(subsets['F'][key])
            subsets['T'][key] = np.array(subsets['T'][key])

        feature_entropy = calculate_entropy_multi(subsets, total)
        if feature_entropy <= node[0]:
            node = (feature_entropy, i, subsets)
    return node


def choose_best(observations, feature_list):
    spam = observations[1]
    not_spam = observations[0]

    node = (1, None, [], [])
    for i in feature_list:
        spam_features = spam[:, i]
        not_spam_features = not_spam[:, i]

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
        print("feature entropy", feature_entropy)

        if feature_entropy <= node[0]:
            node = (feature_entropy, i, spam_subsets, not_spam_subsets)

    return node

def dtl(groups, features, default=0):
    observations = [groups[0], groups[1]]
    #observations = []


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

    best_multi_attribute = choose_best_multi(groups, features)
    feature_ind = best_multi_attribute[1]

    tree = Node(feature_ind)
    subsets = best_multi_attribute[2]

    for i in ('F', 'T'):
        new_features = features.copy()
        new_features.remove(feature_ind)

        groups = subsets[i]

        if i == 'F':
            tree.add_left_child(dtl(groups, new_features))
        else:
            tree.add_right_child(dtl(groups, new_features))

    return tree

def traverse_tree(tree, obs):
    if tree.left_child is None and tree.right_child is None:
        return tree.data

    if obs[tree.data] < 0:
        return traverse_tree(tree.left_child, obs)
    else:
        return traverse_tree(tree.right_child, obs)

def train(train_x, train_y):
    groups = {}
    for i, obs in enumerate(train_x):
        label = train_y[i][0]
        if not label in groups:
            groups[label] = []
        groups[label].append(obs)

    for key in groups:
        np_obs = np.array(groups[key])
        groups[key] = np_obs

    return groups

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    train_x, train_y, test_x, test_y  = handle_data(data)

    groups = train(train_x, train_y)

    spam = []
    not_spam = []
    for i, obs in enumerate(train_x):
        if train_y[i] == 1:
            spam.append(obs)
        else:
            not_spam.append(obs)

    spam = np.array(spam)
    not_spam = np.array(not_spam)


    #using groups
    subsets = initialize_subsets(groups)

    observations = [not_spam, spam]
    #tree = dtl(groups, [0])
    tree = dtl(groups, list(range(0,57)))

    #observations = [not_spam, spam]
    #tree = dtl(observations, list(range(0,57)))

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
