import numpy as np
import sys
from data_operations import filter_low_std, handle_data, get_data_file
from id3_operations import train, dtl, traverse_tree, mode_class
from display_operations import display_multiclass_performance

def main():
    args = sys.argv
    filename = get_data_file(args, './CTG.csv')
    data = np.genfromtxt(filename, delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)
    limit = train_x.shape[1]

    groups = train(train_x, train_y)
    default = mode_class(groups)
    tree = dtl(groups, list(range(0,limit)), default)

    labels = []
    for i, obs in enumerate(test_x):
        labels.append(traverse_tree(tree, obs))

    display_multiclass_performance(labels, test_y)

if __name__ == "__main__":
    main()
