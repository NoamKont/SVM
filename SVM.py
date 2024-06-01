import qpsolvers as qps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import itertools


class SVM:
    def __init__(self, kernel, degree, C=1, gamma=1):
        self.kernel = None
        self.set_kernel(kernel)
        self.degree = degree
        self.C = C
        self.gamma = gamma
        self.alpha = None
        self.support_vectors_ = None
        self.support_vectors_predict = None
        self.threshold = 0.0001
        self.weights_ = None

    def kernel_poly(self, x, y):
        return (1 + x @ y.T) ** self.degree

    def kernel_RBF(self, x, y):
        return np.e ** (-self.gamma * np.linalg.norm(x - y))

    def kernel_sigmoid(self, x, y):
        return np.tanh(self.gamma * np.dot(x, y.T))

    def set_kernel(self, kernel_name):
        if kernel_name == 'poly':
            self.kernel = self.kernel_poly
        elif kernel_name == 'RBF':
            self.kernel = self.kernel_RBF
        elif kernel_name == 'sigmoid':
            self.kernel = self.kernel_sigmoid
        elif kernel_name == 'linear':
            self.kernel = self.kernel_linear
        else:
            raise ValueError('Unknown kernel type')


    def fit(self, X, y):
        N = X.shape[0]
        P = np.empty((N, N))
        for i, j in itertools.product(range(N), range(N)):
            P[i, j] = y[i] * y[j] * self.kernel(X[i, :], X[j, :])
        P = 0.5 * (P + P.T)
        P = 0.5 * P
        q = -np.ones(N)
        GG = -np.eye(N)
        h = np.zeros(N)
        lb = np.zeros(N)
        ub = self.C * np.ones(N)
        self.alpha = qps.solve_qp(P, q, GG, h, lb=lb, ub=ub, solver='osqp')
        index_support_vectors = np.where((np.abs(self.alpha) >= self.threshold))
        self.support_vectors_ = X[index_support_vectors]
        self.support_vectors_predict = y[index_support_vectors]

    def decision_function(self, X_test):
        z = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            z[i] = sum([self.alpha[k] * self.support_vectors_predict[k] * self.kernel(self.support_vectors_[k, :].T, X_test[i, :]) for k in range(len(self.support_vectors_))])
        return z

    def predict(self, X):
        z = self.decision_function(X)
        return np.sign(z)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        res = 0
        for pred, real in zip(y_pred, y_test):
            if pred == real:
                res += 1
        return res / len(y_pred)

    @staticmethod
    def train_test_split(X, y, train_size=0.8):
        train_part = int(X.shape[0] * train_size)

        X_train = X[:train_part].values
        X_test = X[train_part:].values
        y_train = y[:train_part].values
        y_test = y[train_part:].values
        return X_train, X_test, y_train, y_test

    @staticmethod
    def error_plot(error, title):
        error_plot = np.array(error)
        plt.plot(error_plot[:, 1], error_plot[:, 0], marker='o', linestyle='-')
        plt.title(f"Error Rate of {title} Kernel")
        plt.grid(False)
        plt.show()