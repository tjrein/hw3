import numpy as np
from bayes_operations import classify, train
from data_operations import filter_low_std, handle_data
from display_operations import display_binary_performance

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_binary_performance(labels, test_y)

if __name__ == "__main__":
    main()
