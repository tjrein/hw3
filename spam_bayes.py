import numpy as np
import sys
from bayes_operations import classify, train
from data_operations import filter_low_std, handle_data, get_data_file
from display_operations import display_binary_performance

def main():
    args = sys.argv
    filename = get_data_file(args, './spambase.data')
    data = np.genfromtxt(filename, delimiter=',')
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_binary_performance(labels, test_y)

if __name__ == "__main__":
    main()
