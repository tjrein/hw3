import numpy as np
from data_operations import handle_data
from id3_functions import train, dtl, traverse_tree

def main():
    data = np.genfromtxt('./spambase.data', delimiter=',')
    train_x, train_y, test_x, test_y  = handle_data(data)

    groups = train(train_x, train_y)

    tree = dtl(groups, list(range(0,57)))

    labels = []
    for i, obs in enumerate(test_x):
        labels.append(traverse_tree(tree, obs))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    labels = np.array(labels)
    for x, prediction in enumerate(labels):
        target = test_y[x]

        if prediction == 1 and target == 1:
            tp += 1

        if prediction == 1 and target == 0:
            fp += 1

        if prediction == 0 and target == 0:
            tn += 1

        if prediction == 0 and target == 1:
            fn += 1

    print("tp", tp)
    print("tn", tn)
    print("fp", fp)
    print("fn", fn)

    print ("precision", tp / (tp + fp))
    print ("recall", tp / (tp + fn))
    print ((tp + tn) / (tp + tn + fp + fn))

if __name__ == '__main__':
    main()
