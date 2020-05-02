import numpy as np
# without Regularization
hist = []
# Compute the cost


def compute_noreg(x1, y1, theta1):
    m = len(y1)
    w = np.dot(x1, theta1)
    c = np.sum((w-y1)*(w-y1))  # cost = sum( loss - Y)^2 the loss calculated
    cost = c / (2 * m)
    return cost

# Gradient Descent


def gradient_noreg(x, y, theta1, alpha, num_iter):
    m = len(y)  # no. of training examples
    x1 = np.array(x)
    y1 = np.array(y)
    for i in range(0, num_iter):
        w = np.dot(x1, theta1)
        cost = np.sum(w - y1, axis=1, keepdims=True)
        term = np.dot(x1.transpose(), cost)
        # Now the updating of gradients
        dw = alpha*(1/m)*term
        theta1 = theta1 - dw
        temp = compute_noreg(x1, y1, theta1)
        hist.append(temp)

    return hist, theta1

# Predict Function


def predict(x, y, theta):
    x1 = np.array(x)
    y1 = np.array(y)
    res = np.dot(x1, theta)
    res1 = np.sum(res) / np.sum(y)
    return res1
