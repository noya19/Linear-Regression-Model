import test
import test1
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# -------First is to Read the data-------------------------


my_data = np.genfromtxt('train.csv', delimiter=",")  # Test Set
my_valid = np.genfromtxt('valid.csv', delimiter=",")  # Valid Set

# -------Normalize the data---------------------------------
# For Train Set
scalar = MinMaxScaler()
scalar.fit(my_data)
new_data = scalar.transform(my_data)

# For Valid Set
scal = MinMaxScaler()
scal.fit(my_valid)
new_valid = scal.transform(my_valid)
# -----------------------------------------------------------
# ----------assign new arrays as x and y---------------------
x = [[s[j] for j in range(0, len(s) - 1)] for s in new_data]  # features
y = [[s[j] for j in range(0, len(s)) if j == len(s) - 1] for s in new_data]  # value
x_val = [[s[j] for j in range(0, len(s) - 1)] for s in new_valid]
y_val = [[s[j] for j in range(0, len(s)) if j == len(s) - 1] for s in new_valid]

# ------------------------------------------------------------

# weights matrix and initialize values------------------------
hist = []
theta = [[0] for _ in range(len(x[0]))]
num_iter = 200
alpha = 0.01
lam = 0.001  # Regularization
theta1 = np.array(theta)
# ------------------------------------------------------------

# ---Compute with regularization--------------------------
history, theta_reg = test.gradient(x, y, theta1, alpha, num_iter, lam)

# ---Compute without regularization--------------------------
history1, theta_noreg = test1.gradient_noreg(x, y, theta1, alpha, num_iter)

# -----Accuracy Test------------------------------------------
print('\n------Accuracy without Regularization-------\n')
train_acc = test1.predict(x, y, theta_noreg)
print("Training Accuracy", train_acc)
valid_acc = test1.predict(x_val, y_val, theta_noreg)
print("Validation Accuracy", valid_acc)

print('\n------Accuracy with Regularization-------\n')
train_acc = test.predict(x, y, theta_reg)
print("Training Accuracy", train_acc)
valid_acc = test1.predict(x_val, y_val, theta_reg)
print("Validation Accuracy", valid_acc)

print("\nRun the Model on Test data?(y/n):")
res = input()
if res == "y":
    # -----For Testing Phaze-------------
    # Read Data
    my_test = np.genfromtxt('test.csv', delimiter=",")  # Valid Set

    # For Test Set (Normalize data)
    scal1 = MinMaxScaler()
    scal1.fit(my_test)
    new_test = scal1.transform(my_test)

    # ----------assign new arrays as x and y---------------------
    x_test = [[s[j] for j in range(0, len(s) - 1)] for s in new_test]  # features
    y_test = [[s[j] for j in range(0, len(s)) if j == len(s) - 1] for s in new_test]  # value
    x_test1 = np.array(x_test)
    y_test1 = np.array(y_test)
    print('\n------Testing-------\n')
    test_acc = test.predict(x_test1, y_test1, theta_reg)
    print("Test Accuracy", test_acc)

# --------Display the graph of loss over iterations-----------
x_par = [i for i in range(0, num_iter)]
y_par = history
plt.figure("Loss vs no. of iterations")
plt.plot(x_par, y_par, color='b')  # Regularization
plt.xlabel("Total iteration")
plt.ylabel("loss")
plt.show()

# -- no. of iterations


