import numpy as np
from bayes_functions import classify, train
from data_operations import filter_low_std, handle_data

def display_performance(labels, test_y):
    correct = 0
    incorrect = 0
    for x, prediction in enumerate(labels):
        target = test_y[x]

        if target == prediction:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (incorrect + correct)

    print("Accuracy: ", accuracy)

def main():
    data = np.genfromtxt('./CTG.csv', delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_performance(labels, test_y)

if __name__ == "__main__":
    main()
