import cvxopt
import matplotlib.pyplot as plt
import numpy as np
from cvxopt import solvers
from numpy import double

import softsvm


def kernel(x1, x2, k):
    return (1 + np.dot(x1, x2)) ** k


def calculate_gramM_G(trainX, m, k):
    G = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            G[i, j] = kernel(trainX[i], trainX[j], k)
    return G


def calc_A21_matrix(trainY, G):
    m, n = G.shape  # m = 100
    A_2_1 = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            A_2_1[i][j] = G[i][j] * trainY[i]

    return A_2_1


"""

    :param l: the parameter lambda of the soft SVM algorithm
    :param sigma: the bandwidth parameter sigma of the RBF kernel.
    :param trainX: numpy array of size (m, d) containing the training sample
    :param trainy: numpy array of size (m, 1) containing the labels of the training sample
    :return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """


def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    # Calculate the number of examples and dimensions
    m, d = trainX.shape  # m = 100, d = 784

    # Create G as gram matrix in size of m x m
    G = calculate_gramM_G(trainX, m, k)

    # For preventing H eigenvalues to be negative
    epsilon = 0.001
    epsilon_matrix = np.eye(2 * m, dtype=double) * epsilon

    # Create H as blocking matrix in size of 2m x 2m
    H_11 = 2 * l * G
    H_12 = np.zeros([m, m])
    H_21 = np.zeros([m, m])
    H_22 = np.zeros([m, m])
    H = np.block([[H_11, H_12],
                  [H_21, H_22]])

    H_cvxopt = cvxopt.matrix(H + epsilon_matrix)

    # Create vector u = [0...0, 1/m...1/m] zeros are first d enters and 1/m are m next enters (size of m + d)
    u = np.concatenate((np.zeros(m), np.ones(m) * 1 / m))
    u_cvxopt = cvxopt.matrix(u)

    # Create vector v = [0...0, 1...1] zeros are first m enters and 1 are m next enters (size of 2m)
    v = np.concatenate((np.zeros(m), np.ones(m)))
    v_cvxopt = cvxopt.matrix(v)

    # Create A as blocking matrix in size of 2m X 2m
    A_11 = np.zeros([m, m])
    A_12 = np.eye(m, dtype=double)
    A_21 = calc_A21_matrix(trainy, G)
    A_22 = np.eye(m, dtype=double)
    A = np.block([[A_11, A_12],
                  [A_21, A_22]])
    A_cvxopt = cvxopt.matrix(A + epsilon_matrix)

    sol = solvers.qp(H_cvxopt, u_cvxopt, -A_cvxopt, -v_cvxopt)
    alpha = np.array(sol["x"])[:m]

    return alpha


def q4_a():
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainY = data['Ytrain']

    fig, ax = plt.subplots()

    # Plot the points in the training set
    ax.scatter(trainX[:, 0], trainX[:, 1], c=trainY, cmap='coolwarm')

    # Show the plot
    plt.show()


def calculate_kernel_for_hw(x, trainX, k):
    """
    Returns an array with calculated kernels - [K(x0, x) , ... , K(xm, x)]
    """
    return [kernel(trainX_element, x, k) for trainX_element in trainX]


def hw(x, trainX, alphas, karnel_k):
    """
    Returns the output of h on x > h(x)
    """
    return np.sign(np.inner(alphas.flatten(), calculate_kernel_for_hw(x, trainX, karnel_k)))


def compute_softsvmpoly_error(alpahs, X_validation_examples, Y_validation_labels, train_X, karnel_k):
    sample_length = len(X_validation_examples)
    hw_outputs = np.zeros(shape=len(X_validation_examples))
    for i in range(sample_length):
        hw_outputs[i] = hw(X_validation_examples[i], train_X, alpahs, karnel_k)
    return np.mean(hw_outputs != Y_validation_labels)


def compute_softsvm_error(w, X_validation_examples, Y_validation_labels):
    train_preds = np.sign(X_validation_examples @ w)
    return np.mean(Y_validation_labels != list(np.concatenate(train_preds).flat))


def k_fold_cross_validation(number_of_folds: int, trainX: np.array, trainY: np.array,
                            possible_values_for_parm: np.array, karnel_k: int | None):
    # Step 1: Split the training sample into k equal parts
    folds_X = np.array(np.split(trainX, number_of_folds))
    folds_Y = np.array(np.split(trainY, number_of_folds))

    # Step 2: Iterate over the possible values for the parameter α
    best_param = None
    min_error = float("inf")
    for param in possible_values_for_parm:
        # Step 3: Iterate over the folds
        total_error = 0
        for i in range(1, number_of_folds):
            # Step 4: Set the validation set and the training set
            validation_X = folds_X[i]
            validation_Y = folds_Y[i]
            current_train_X = np.concatenate(np.delete(folds_X, i, axis=0))
            current_train_Y = np.concatenate(np.delete(folds_Y, i, axis=0))

            # Use softsvmpoly
            if karnel_k is not None:
                # Step 5: Train the model on the training set
                alpahs = softsvmpoly(param, karnel_k, current_train_X, current_train_Y)

                # Step 6: Compute the error on the validation set
                error = compute_softsvmpoly_error(alpahs, validation_X, validation_Y, current_train_X, karnel_k)

            # Use softsvm
            else:
                # Step 5: Train the model on the training set
                w = softsvm.softsvm(param, current_train_X, current_train_Y)

                # Step 6: Compute the error on the validation set
                error = compute_softsvm_error(w, validation_X, validation_Y)

            # Step 7: Add the error to the total error
            total_error += error

        # Step 8: Compute the average error for the current value of α
        avg_error = total_error / number_of_folds

        # Step 9: If the average error is the lowest so far, update the best parameter value
        if avg_error < min_error:
            min_error = avg_error
            best_param = param

    # Step 10: Best param
    return min_error


def find_min_error(err_list):
    min_error = float("inf")
    optima_lamda = None
    optima_kernel_k = None
    for i in range(len(err_list)):
        if err_list[i][0] < min_error:
            min_error = err_list[i][0]
            optima_lamda = err_list[i][1]
        if len(err_list[i]) > 2:
            optima_kernel_k = err_list[i][2]
    if len(err_list[i]) > 2:
        return min_error, optima_lamda, optima_kernel_k
    else:
        return min_error, optima_lamda


def q4_b():
    # load question 4 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainY = data['Ytrain']

    # A finite set of possible values for
    lamdas_options = np.array([1, 10, 100])
    kernel_k_options = np.array([2, 5, 8])
    number_of_folds_for_cross_validation = 5
    errors = {}
    err_list = np.zeros([9, 3])
    possible_values_for_parm = np.zeros(1)
    ind = 0
    for i in range(len(lamdas_options)):
        for j in range(len(kernel_k_options)):
            possible_values_for_parm[0] = lamdas_options[i]
            current_error = k_fold_cross_validation(number_of_folds_for_cross_validation, trainX, trainY,
                                                    possible_values_for_parm, kernel_k_options[j])
            errors['[' + str(lamdas_options[i]) + ',' + str(kernel_k_options[j]) + ']'] = current_error
            err_list[ind][0] = current_error
            err_list[ind][1] = lamdas_options[i]
            err_list[ind][2] = kernel_k_options[j]
            ind = ind + 1

    # Report the 9 average validation error values for each of the pairs (λ, k) and which pair was selected by the cross validation
    print("Cross Validation average errors: ")
    for key, value in errors.items():
        print(str(key) + ": Error = " + str(value))
    min_error, optimal_lamda, optimal_kernel_k = find_min_error(err_list)  # [error, lamda, k]
    print("Polynomial SVM optimal error is: " + str(min_error))
    print("Cross Validation chose Hyper parameters: " + "lamda: " + str(optimal_lamda) + " k: " + str(optimal_kernel_k))

    # Report the test error of the resulting classifier
    testX = data['Xtest']
    testY = data['Ytest']
    alphas = softsvmpoly(optimal_lamda, optimal_kernel_k, trainX, trainY)
    testError = compute_softsvmpoly_error(alphas, testX, testY, trainX, optimal_kernel_k)
    print("The error on the test sample of Polynomial soft SVM is: " + str(testError))

    # Repeat the procedure above using the linear (non-kernel) soft SVM code from Q1 on the given training set
    errors = {}
    err_list = np.zeros([3, 2])
    possible_values_for_parm = np.zeros(1)
    ind = 0
    for i in range(len(lamdas_options)):
        possible_values_for_parm[0] = lamdas_options[i]
        current_error = k_fold_cross_validation(number_of_folds_for_cross_validation, trainX, trainY,
                                                possible_values_for_parm, None)
        errors['[' + str(lamdas_options[i]) + ']'] = current_error
        err_list[ind][0] = current_error
        err_list[ind][1] = lamdas_options[i]
        ind = ind + 1

    # Report the 9 average validation error values for each of the pairs (λ, k) and which pair was selected by the cross validation
    print("Cross Validation average errors: ")
    for key, value in errors.items():
        print(str(key) + ": Error = " + str(value))
    min_error, optimal_lamda = find_min_error(err_list)  # [error, lamda]
    print("Linear SVM optimal error is: " + str(min_error))
    print("Cross Validation chose Hyper parameters: " + "lamda: " + str(optimal_lamda))

    # Report the test error of the resulting classifier
    w = softsvm.softsvm(optimal_lamda, trainX, trainY)
    testError = compute_softsvm_error(w, testX, testY)
    print("The error on the test sample of linear soft SVM is: " + str(testError))


# Plot data with given x and y values for two classes, with a given title
def plot_data(x_1, y_1, x_2, y_2, title):
    plt.scatter(x_1, y_1, color="red")
    plt.scatter(x_2, y_2, color="blue")
    plt.title(title)
    plt.legend()
    plt.show()


# Clear given lists of data
def clear_lists(x_1, x_2, y_1, y_2):
    x_1.clear()
    x_2.clear()
    y_1.clear()
    y_2.clear()


# Function to perform soft SVM with polynomial kernel and plot results
def q4e():
    # Load training data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']

    # Initialize matrix for storing results of polynomial kernel function
    my_matrix = np.eye(50)

    # Initialize lists for storing data to plot
    x_1 = list()
    x_2 = list()
    y_1 = list()
    y_2 = list()

    # Loop through different hyperparameter values
    for k in enumerate([3, 5, 8]):
        # Calculate alpha values for SVM with given hyperparameters
        alphas = softsvmpoly(100, k[1], trainX, trainy)
        # Clear lists for storing data to plot
        clear_lists(x_1, x_2, y_1, y_2)
        # Loop through grid of points in range [-1, 1]
        for i in range(50):
            for j in range(50):
                # Calculate polynomial kernel function for current point
                my_matrix[i][j] = hw([1 - i / 25, -1 + j / 25], trainX, alphas, 8)
                # Store point in appropriate list based on kernel function result
                if my_matrix[i][j] == 1:
                    x_1.append(1 - i / 25)
                    y_1.append(-1 + j / 25)
                else:
                    x_2.append(1 - i / 25)
                    y_2.append(-1 + j / 25)
        # Plot data for current hyperparameter values
        plot_data(x_1, y_1, x_2, y_2, "With hyperparameters: l=100, k=" + str(k[1]))

    # Clear lists for storing data to plot
    clear_lists(x_1, x_2, y_1, y_2)
    # Loop through training data
    for i in range(len(trainX)):
        # Store data in appropriate list based on class label
        if trainy[i] == 1:
            x_1.append(trainX[i][0])
            y_1.append(trainX[i][1])
        else:
            x_2.append(trainX[i][0])
            y_2.append(trainX[i][1])

    plot_data(x_1, y_1, x_2, y_2, "Real Values")


def simple_test():
    # load question 4 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def pci(x):
    x1 = x[0]
    x2 = x[1]
    #  The return value is the formula for q_4_f_i
    return np.array([1, x1, x2, x1 ** 2, x1 * x2, x2 ** 2, x1 ** 3, x1 ** 2 * x2, x1 * x2 ** 2, x2 ** 3, x1 ** 4,
                     x1 ** 3 * x2, x1 ** 2 * x2 ** 2, x1 * x2 ** 3, x2 ** 4, x1 ** 5, x1 ** 4 * x2, x1 ** 3 * x2 ** 2,
                     x1 ** 2 * x2 ** 3, x1 * x2 ** 4, x2 ** 5])


def calculate_w():
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    trainy = data['Ytrain']
    alphas = softsvmpoly(1, 5, trainX, trainy)
    w = alphas[0] * pci(trainX[0])
    for i in range(1, trainX.shape[0]):
        w = w + alphas[i] * pci(trainX[i])
    print(w)  # The answer for q_4_f_ii
    return w


def h_w(w, x):
    return np.sign(np.inner(w, pci(x)))


def q_4_f_iii():
    pcai = np.array(
        ["1", "x1", "x2", "x1**2", "x1 * x2", "x2 ** 2", "x1 ** 3", "x1 ** 2 * x2", "x1 * x2 ** 2", "x2 ** 3",
         "x1 ** 4",
         "x1 ** 3 * x2", "x1 ** 2 * x2 ** 2", "x1 * x2 ** 3", "x2 ** 4, x1 ** 5", "x1 ** 4 * x2", "x1 ** 3 * x2 ** 2",
         "x1 ** 2 * x2 ** 3", "x1 * x2 ** 4", "x2 ** 5"])
    w = calculate_w()
    res = "f(x1,x2) = "
    for pcai_i, w_i in zip(pcai, w):
        res = res + str(w_i) + " * " + pcai_i + " + \n "
    print(res)
    print(res[: len(res) - 4])


def q4_f():
    # Load data from ex2q4_data.npz file
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']

    # Initialize lists to store data points
    x_1 = list()
    x_2 = list()
    y_1 = list()
    y_2 = list()

    # Calculate w using calculate_w function
    w = calculate_w()

    # Loop through each data point in trainX
    for i in range(len(trainX)):
        # If h_w function returns 1, append the x and y values to x_1 and y_1 respectively
        if h_w(w, trainX[i]) == 1:
            x_1.append(trainX[i][0])
            y_1.append(trainX[i][1])
        # If h_w function returns 0, append the x and y values to x_2 and y_2 respectively
        else:
            x_2.append(trainX[i][0])
            y_2.append(trainX[i][1])

    # Print the inner product of w and pci(trainX)
    print(np.inner(w, pci(trainX)))

    # Create a scatter plot using the x and y values from x_1 and y_1, with the points colored red
    plt.scatter(x_1, y_1, color="red")
    # Create a scatter plot using the x and y values from x_2 and y_2, with the points colored blue
    plt.scatter(x_2, y_2, color="blue")
    plt.title("Real Values")  # Set the title of the plot
    plt.legend()  # Display the plot legend
    plt.show()  # The answer for q_4_f_iv


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()
    q4_a()
    q4_b()
    q4_f()

    # here you may add any code that uses the above functions to solve question 4
