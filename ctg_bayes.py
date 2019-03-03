import numpy as np
import sys
from bayes_operations import classify, train
from data_operations import filter_low_std, handle_data, get_data_file
from display_operations import display_multiclass_performance

def main():
    args = sys.argv
    filename = get_data_file(args, './CTG.csv')
    data = np.genfromtxt(filename, delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_multiclass_performance(labels, test_y)

if __name__ == "__main__":
    main()
