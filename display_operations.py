def display_binary_performance(labels, test_y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

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

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print ("Precision:", precision)
    print ("Recall:", recall)
    print ("F-measure:", (2 * precision * recall) / (precision + recall))
    print ("Accuracy:", (tp + tn) / (tp + tn + fp + fn))

def display_multiclass_performance(labels, test_y, classifier):
    correct = 0
    incorrect = 0
    for x, prediction in enumerate(labels):
        target = test_y[x]

        if target == prediction:
            correct += 1
        else:
            incorrect += 1

    accuracy = correct / (incorrect + correct)

    print(classifier + " Accuracy:", accuracy)
