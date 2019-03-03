import numpy as np
from bayes_operations import classify, train
from data_operations import filter_low_std, handle_data
from display_operations import display_multiclass_performance

def main():
    data = np.genfromtxt('./CTG.csv', delimiter=',', skip_header=2)
    second_last_col = data.shape[1] - 2
    data = np.delete(data, second_last_col, 1)
    data = filter_low_std(data)

    train_x, train_y, test_x, test_y = handle_data(data)

    groups = train(train_x, train_y)
    labels = np.array(classify(groups, test_x))
    display_multiclass_performance(labels, test_y)

if __name__ == "__main__":
    main()
