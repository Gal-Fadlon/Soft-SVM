# Soft-SVM
This project provides an implementation of the Soft Support Vector Machine (Soft-SVM) algorithm. Soft-SVM is a variant of the Support Vector Machine (SVM) algorithm, that handles non-linearly separable data through the introduction of slack variables and a regularization parameter, often denoted as lambda or l. These elements allow for misclassifications with a certain penalty, thus "softening" the decision boundary.

# Project Structure
The implementation is a two Python script which consists of several functions for handling the core computations required for Soft-SVM, as well as some additional functions for testing, experiment setup, and results visualization.

calc_A21_matrix(trainX, trainY): This function calculates a specific matrix used in the quadratic programming problem, which is a crucial part of the SVM algorithm.

softsvm(l, trainX: np.array, trainy: np.array): The main function implementing the Soft-SVM algorithm. This function takes as input the regularization parameter l, training data trainX, and corresponding labels trainy, and returns the weight vector of the linear predictor.

q2(): A function running multiple experiments on the Soft-SVM algorithm with different parameters and visualizing the results.

simple_test(): A simple testing function which verifies that the softsvm function behaves as expected.

# How to Run
To run this project:

Ensure you have numpy, cvxopt, and matplotlib installed in your Python environment. You can install these packages using pip:

pip install numpy cvxopt matplotlib

# Contributions
Contributions are welcome! Please make sure that your pull requests are well-documented.

# License
This project is licensed under the terms of the MIT license.

# Contact
If you have any questions, feel free to reach out or raise an issue in the repository.
