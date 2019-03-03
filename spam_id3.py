import numpy as np
from data_operations import handle_data, filter_low_std
from id3_operations import train, dtl, traverse_tree
from display_operations import display_binary_performance

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    data = filter_low_std(data)
    train_x, train_y, test_x, test_y  = handle_data(data)

    groups = train(train_x, train_y)
    limit = train_x.shape[1]

    tree = dtl(groups, list(range(0,limit)))

    labels = []
    for i, obs in enumerate(test_x):
        labels.append(traverse_tree(tree, obs))

    display_binary_performance(labels, test_y)

if __name__ == '__main__':
    main()
