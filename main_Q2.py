import qpsolvers as qps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import itertools


def plot_classifier_z_kernel(alpha, X, y, ker,threshold=0.1):
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    red = np.where(y < 0)
    blue = np.where(y > 0)
    plt.plot(X[red, 0], X[red, 1], 'o', color='red')
    plt.plot(X[blue, 0], X[blue, 1], 'o', color='blue')

    xx = np.linspace(x_min, x_max)
    yy = np.linspace(y_min, y_max)

    xx, yy = np.meshgrid(xx, yy)
    N = X.shape[0]
    z = np.zeros(xx.shape)
    for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
        z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]])) for k in range(N)])

    support_vectors = np.where((np.abs(alpha) >= threshold))
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='black')

    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
    plt.show()

def RBF_kernel(x, y, gamma: float = 1.0):
    return np.e ** (-gamma * np.linalg.norm(x - y))


def polynomial_kernel(x, y, gamma: float = 1.0, degree: int = 2.0):
    return (gamma * np.dot(x, y.T) + 1) ** degree


def svm_dual_kernel(X, y, ker,threshold=0.1):
    N = X.shape[0]
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :])
    P = 0.5 * (P + P.T)
    P = 0.5 * P
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)
    alpha = qps.solve_qp(P, q, GG, h, solver='osqp')

    return alpha


if __name__ == "__main__":
    with open("..\simple_nonlin_classification.csv") as file:
        df = pd.read_csv(file)
    X = df.drop(columns=['y']).values
    #X = np.c_[X, np.ones(X.shape[0])]
    y = df['y'].values

    alpha = svm_dual_kernel(X, y,RBF_kernel)
    plot_classifier_z_kernel(alpha, X, y, RBF_kernel)
