import numpy as np
import math
from math import ceil

def initialize_probs(groups):
    class_probs = {}
    for key, val in groups.items():
        class_probs[key] = val['prior']
    return class_probs

def pdf(x, mean, std):
    pdf = (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-( (x - mean) ** 2 ) / (2 * std ** 2))
    return pdf

def train(train_x, train_y):
    groups = {}
    for i, obs in enumerate(train_x):
        label = train_y[i][0]
        if not label in groups:
            groups[label] = { 'observations': [] }
        groups[label]['observations'].append(obs)

    for key in groups:
        np_obs = np.array(groups[key]['observations'])
        groups[key]['observations'] = np_obs
        groups[key]['mean'] = np_obs.mean(axis=0)
        groups[key]['std'] = np_obs.std(axis=0)
        groups[key]['prior'] = len(np_obs) / len(train_x)

    return groups

def classify(groups, test_x):
    labels = []

    for obs in test_x:
        class_probs = initialize_probs(groups)
        for j, feature in enumerate(obs):
            for key, val in groups.items():
                mean = val['mean'][j]
                std = val['std'][j]
                class_probs[key] *= pdf(feature, mean, std)
        labels.append(max(class_probs, key=class_probs.get))

    return labels
