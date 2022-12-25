import cvxopt.solvers
import numpy
import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
from numpy import double


def calc_A21_matrix(trainX, trainY):
    m, d = trainX.shape  # m = 100, d = 784
    A_2_1 = np.zeros([m, d])
    for i in range(m):
        for j in range(d):
            A_2_1[i][j] = trainX[i][j] * trainY[i]

    return A_2_1


"""
:param l: the parameter lambda of the soft SVM algorithm
:param trainX: numpy array of size (m, d) containing the training sample
:param trainy: numpy array of size (m, 1) containing the labels of the training sample
:return: linear predictor w, a numpy array of size (d, 1)
"""


def softsvm(l, trainX: np.array, trainy: np.array):
    # Calculate the number of examples and dimensions
    m, d = trainX.shape  # m = 100, d = 784

    # For preventing H eigenvalues to be negative
    epsilon = 0.001
    epsilon_matrix = np.eye(d + m, dtype=double) * epsilon

    # Create H as blocking matrix in size of (d + m) X (d + m)
    H_11 = 2 * l * np.eye(d, dtype=double)
    H_12 = np.zeros([d, m])
    H_21 = np.zeros([m, d])
    H_22 = np.zeros([m, m])
    H = np.block([[H_11, H_12],
                  [H_21, H_22]])
    H_cvxopt = cvxopt.matrix(H + epsilon_matrix)

    # Create vector u = [0...0, 1/m...1/m] zeros are first d enters and 1/m are m next enters (size of m + d)
    u = np.concatenate((np.zeros(d), np.ones(m) * 1 / m))
    u_cvxopt = cvxopt.matrix(u)

    # Create vector v = [0...0, 1...1] zeros are first m enters and 1 are m next enters (size of 2m)
    v = np.concatenate((np.zeros(m), np.ones(m)))
    v_cvxopt = cvxopt.matrix(v)

    # Create A as blocking matrix in size of (d + m) X 2m
    A_11 = np.zeros([m, d])
    A_12 = np.eye(m, dtype=double)
    A_21 = calc_A21_matrix(trainX, trainy)
    A_22 = np.eye(m, dtype=double)
    A = np.block([[A_11, A_12],
                  [A_21, A_22]])
    epsilon_matrix = np.eye(2 * m, d + m, dtype=double) * epsilon
    A_cvxopt = cvxopt.matrix(A + epsilon_matrix)

    sol = solvers.qp(H_cvxopt, u_cvxopt, -A_cvxopt, -v_cvxopt)
    w = np.array(sol["x"])[:d]

    return w


def q2():
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainY = data['Ytrain']
    testY = data['Ytest']

    # Experiment 1
    n_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_of_tests = 10
    m = 100
    # Run the softsvm algorithm
    sample_error_train_1 = np.zeros(
        (len(n_1), num_of_tests))  # For each lamda we make 10 tests so we get a matrix of 10 x 10
    sample_error_test_1 = np.zeros(
        (len(n_1), num_of_tests))  # For each lamda we make 10 tests so we get a matrix of 10 x 10
    for i in range(len(n_1)):
        l = 10 ** n_1[i]
        for k in range(num_of_tests):
            # Get a random m training examples from the training set
            indices_permutation_train = np.random.permutation(trainX.shape[0])  # This is a permutation over the index's
            _trainX = trainX[indices_permutation_train[:m]]
            _trainY = trainY[indices_permutation_train[:m]]

            w = softsvm(l, _trainX, _trainY)
            train_preds = np.sign(_trainX @ w)  # Equivalent to inner product of <w, x_i> one be one
            sample_error_train_1[i][k] = np.mean(
                _trainY != list(np.concatenate(train_preds).flat))

            test_preds = np.sign(testX @ w)  # Equivalent to inner product of <w, x_i> one be one
            sample_error_test_1[i][k] = np.mean(
                testY != list(np.concatenate(test_preds).flat))

    sample_statistics_train = {"avg": [sum(error) / len(error) for error in sample_error_train_1],
                               "max": [max(error) for error in sample_error_train_1],
                               "min": [min(error) for error in sample_error_train_1]}
    sample_statistics_test = {"avg": [sum(error) / len(error) for error in sample_error_test_1],
                              "max": [max(error) for error in sample_error_test_1],
                              "min": [min(error) for error in sample_error_test_1]}

    # Experiment 2
    m = 1000
    # Get a random m training examples from the training set
    n_2 = [1, 3, 5, 8]
    # run the softsvm algorithm
    sample_error_train_2 = np.zeros(len(n_2))
    sample_error_test_2 = np.zeros(len(n_2))
    for i in range(len(n_2)):
        l = 10 ** n_2[i]
        # Get a random m training examples from the training set
        indices_permutation_train = np.random.permutation(trainX.shape[0])  # This is a permutation over the index's
        _trainX = trainX[indices_permutation_train[:m]]
        _trainY = trainY[indices_permutation_train[:m]]

        w = softsvm(l, _trainX, _trainY)
        train_preds = np.sign(_trainX @ w)
        sample_error_train_2[i] = np.mean(
            _trainY != list(np.concatenate(train_preds).flat))  # Equivalent to inner product of <w, x_i> one be one

        test_preds = np.sign(testX @ w)
        sample_error_test_2[i] = np.mean(
            testY != list(np.concatenate(test_preds).flat))  # Equivalent to inner product of <w, x_i> one be one

    # Plotting the graph
    plt.scatter(n_1, sample_statistics_train["max"], color="orange")
    plt.scatter(n_1, sample_statistics_train["min"], color="green")
    plt.scatter(n_1, sample_statistics_train["avg"], color="blue")
    plt.scatter(n_1, sample_statistics_test["max"], color="gray")
    plt.scatter(n_1, sample_statistics_test["min"], color="purple")
    plt.scatter(n_1, sample_statistics_test["avg"], color="black")
    x = n_1
    y_train = sample_statistics_train["avg"]
    y_test = sample_statistics_test["avg"]
    up = list()
    down = list()
    for item1, item2 in zip(sample_statistics_train["max"], sample_statistics_train["avg"]):
        up.append(item1 - item2)
    for item1, item2 in zip(sample_statistics_train["avg"], sample_statistics_train["min"]):
        down.append(item1 - item2)
    yerr_train = [down, up]
    up_test = list()
    down_test = list()
    for item1, item2 in zip(sample_statistics_test["max"], sample_statistics_test["avg"]):
        up_test.append(item1 - item2)
    for item1, item2 in zip(sample_statistics_test["avg"], sample_statistics_test["min"]):
        down_test.append(item1 - item2)
    yerr_test = [down_test, up_test]
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.xlim((0, 10.5))
    plt.ylim((0, 0.7))
    plt.errorbar(x, y_train, yerr=yerr_train, capsize=3, fmt="r--o", color="black", ecolor="black",
                 label="Train Error Experiment1")
    plt.errorbar(x, y_test, yerr=yerr_test, capsize=3, fmt="r--o", ecolor="black", label="Test Error Experiment1")
    plt.scatter(n_2, sample_error_train_2, label="Train Error Experiment2", color="#8B0000")
    plt.scatter(n_2, sample_error_test_2, label="Test Error Experiment2", color="yellow")
    plt.legend()
    plt.show()


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100
    d = trainX.shape[1]  # 784 X 1

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])  # This is a permutation over the index's
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvm algorithm
    w = softsvm(10, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert w.shape[0] == d and w.shape[1] == 1, f"The shape of the output should be ({d}, 1)"

    # get a random example from the test set, and classify it
    i = np.random.randint(0, testX.shape[0])
    predicty = np.sign(testX[i] @ w)

    # this line should print the classification of the i'th test sample (1 or -1).
    print(f"The {i}'th test sample was classified as {predicty}")


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    q2()
