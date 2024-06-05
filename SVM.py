import qpsolvers as qps
import numpy as np
import matplotlib.pyplot as plt
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
        self.threshold = 0

    def kernel_poly(self, x, y):
        return (1 + x @ y.T) ** self.degree

    def kernel_RBF(self, x, y):
        return np.e ** (-self.gamma * np.linalg.norm(x - y))

    def set_kernel(self, kernel_name):
        if kernel_name == 'poly':
            self.kernel = self.kernel_poly
        elif kernel_name == 'RBF':
            self.kernel = self.kernel_RBF
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
            z[i] = sum([self.alpha[k] * self.support_vectors_predict[k] * self.kernel(self.support_vectors_[k, :].T, X_test[i, :]) for k in
                        range(len(self.support_vectors_))])
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
    def error_heatmap_plot(error, title):
        scores = np.array(error)[:, 0]
        gammas = np.array(list(sorted(set(row[1] for row in error))))
        C = np.array(list(sorted(set(row[2] for row in error))))
        scores_matrix = scores.reshape(len(gammas), len(C))
        plt.imshow(scores_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.xticks(np.arange(len(C)), labels=C)
        plt.yticks(np.arange(len(gammas)), labels=gammas)
        for i in range(len(gammas)):
            for j in range(len(C)):
                plt.text(j, i, f'{scores_matrix[i, j]:.2f}', weight='bold', ha='center', va='center', color='w')

        plt.xlabel('C')
        if title == "poly":
            plt.ylabel('Degrees')
        else:
            plt.ylabel('Gamma')
        plt.title('Heatmap of Errors')
        plt.show()
