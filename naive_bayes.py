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

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',', dtype="uint16")
    train_x, train_y, test_x, test_y  = handle_data(data)

    spam = []
    not_spam = []
    for i, obs in enumerate(train_x):
        if train_y[i] == 1:
            spam.append(obs)
        else:
            not_spam.append(obs)
    
    spam_prior = len(spam) / len(train_x)
    not_spam_prior = len(not_spam) / len(train_x)

    spam = np.array(spam)
    not_spam = np.array(not_spam)

    spam_mean = np.mean(spam, axis=0)
    spam_std = np.std(spam, axis=0)

    not_spam_mean = np.mean(not_spam, axis=0)
    not_spam_std = np.std(not_spam, axis=0)

    labels = []
    for i, obs in enumerate(test_x):

        pos_prob = spam_prior
        neg_prob = not_spam_prior

        for j, feature in enumerate(obs):

            pos_mean = spam_mean[j]
            pos_std = spam_std[j]
            neg_mean = not_spam_mean[j]
            neg_std = not_spam_std[j]

            pos_prob *= normpdf(feature, pos_mean, pos_std)
            neg_prob *= normpdf(feature, neg_mean, neg_std)

        if pos_prob > neg_prob:
            labels.append(1)
        else:
            labels.append(0)

    print("all done")
    labels = np.array(labels)
    for x, label in enumerate(labels):
        print(label, test_y[x])

if __name__ == '__main__':
    main()

