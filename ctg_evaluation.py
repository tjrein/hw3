import numpy as np
import sys
from data_operations import filter_low_std, handle_data, get_data_file
from display_operations import display_multiclass_performance, display_binary_performance
import id3_operations as id3
import bayes_operations as bayes

def perform_gnb(train_x, train_y, test_x, test_y):
    groups = bayes.train(train_x, train_y)
    labels = np.array(bayes.classify(groups, test_x))
    display_multiclass_performance(labels, test_y, "Naive Bayes")

def perform_id3(train_x, train_y, test_x, test_y):
    limit = train_x.shape[1]

    groups = id3.train(train_x, train_y)
    default = id3.mode_class(groups)
    tree = id3.dtl(groups, list(range(0,limit)), default)

    labels = []
    for i, obs in enumerate(test_x):
        labels.append(id3.traverse_tree(tree, obs))

    display_multiclass_performance(labels, test_y, "ID3")

def main():
    args = sys.argv
    filename = get_data_file(args, './CTG.csv')
    data = np.genfromtxt(filename, delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)
    perform_gnb(train_x, train_y, test_x, test_y)
    perform_id3(train_x, train_y, test_x, test_y)

if __name__ == "__main__":
    main()
