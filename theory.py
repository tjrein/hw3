import math
import numpy as np

np.set_printoptions(suppress=True)

def standardize(features, mean=None, std=None):
    features = (features - mean) / std
    return features

def standardize_data(matrix):
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0, ddof=1)
    return (matrix - mean) / std

def entropy_helper(frac):
    if frac == 0:
        return 0

    return -(frac) * math.log(frac, 2)


entropy = 3/6 * (entropy_helper(2/3) + entropy_helper(1/3)) + 2/6

starting_entropy = entropy_helper(4/7) + entropy_helper(3/7)

x1_entropy = 8/21 * (entropy_helper(7/8) + entropy_helper(1/8)) + 13/21 * (entropy_helper(5/13) + entropy_helper(8/13))
x2_entropy = 10/21 * (entropy_helper(7/10) + entropy_helper(3/10)) + 11/21 * (entropy_helper(5/11) + entropy_helper(6/11))

#print("starting_entropy", starting_entropy)
#print("x1 entropy", x1_entropy)
#print("x2 entropy", x2_entropy)

x = np.array( [
    [216, 5.68],
    [69, 4.78],
    [302, 2.31],
    [60, 3.16],
    [393, 4.2]
])

orig_mean = x.mean(axis=0)

print(orig_mean)
orig_std = x.std(axis=0, ddof=1)
print(orig_std)

x = standardize_data(x)

test = np.array([242, 4.56])
test = standardize(test, orig_mean, orig_std)

print(test)


yes = []
no = []
labels = [1, 1, 0, 1, 0]

for i, label in enumerate(labels):
    if label is 1:
        yes.append(x[i])
    else:
        no.append(x[i])

yes = np.array(yes)
no = np.array(no)

yes_mean = yes.mean(axis=0)
yes_std = yes.std(axis=0)

no_mean = no.mean(axis=0)
no_std = no.std(axis=0)
