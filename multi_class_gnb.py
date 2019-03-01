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
    cutoff = data.shape[1] - 1
    targets = data[:, cutoff:]
    features = data[:, :cutoff]
    return (targets, features)

def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    return features

def handle_data(data):
    np.random.seed(3)
    np.random.shuffle(data)

    range = ceil(len(data) * 2/3)

    training = data[0:range]
    testing = data[range:]

    return isolate_sets(training, testing)

def initialize_probs(groups):
    class_probs = {}

    for key, val in groups.items():
        class_probs[key] = val['prior']

    return class_probs

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def main():
    data = np.genfromtxt('./CTG.csv', delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)

    #data = np.genfromtxt('./spambase.data', delimiter=',')

    train_x, train_y, test_x, test_y = handle_data(data)

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

    labels = []
    for i, obs in enumerate(test_x):

        class_probs = initialize_probs(groups)

        for j, feature in enumerate(obs):
            for key, val in groups.items():

                mean = val['mean'][j]
                std = val['std'][j]
                class_probs[key] *= normpdf(feature, mean, std)

        labels.append(max(class_probs, key=class_probs.get))

    labels = np.array(labels)
    correct = 0
    incorrect = 0
    for x, prediction in enumerate(labels):
        target = test_y[x]

        if target == prediction:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (incorrect + correct)

    print("accuracy: ", accuracy)


if __name__ == '__main__':
    main()
