import numpy as np
import math

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

def calculate_entropy(subsets, total):
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

def initialize_subsets(groups):
    subsets = { 'T': {}, 'F': {}}

    for key in groups:
        subsets['T'][key] = []
        subsets['F'][key] = []

    return subsets

def choose_best(groups, feature_list):
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

        feature_entropy = calculate_entropy(subsets, total)
        if feature_entropy <= node[0]:
            node = (feature_entropy, i, subsets)
    return node

def classes_with_examples(groups):
    classes = []
    for key, val in groups.items():
        if len(val):
            classes.append(key)

    return classes

def mode_class(groups):
    class_counts = {}
    for key, val in groups.items():
        if len(val):
            class_counts[key] = len(val)

    return max(class_counts, key=class_counts.get)

def dtl(groups, features, default=0):
    classes = classes_with_examples(groups)

    if not len(classes):
        return Node(default)

    if len(classes) == 1:
        return Node(int(classes[0]))

    if not len(features):
        return Node(int(mode_class(groups)))

    best_multi_attribute = choose_best(groups, features)
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
