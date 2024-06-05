import qpsolvers as qps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import itertools


def RBF_kernel(x, y, gamma: float = 1.0):
    return np.e ** (-gamma * np.linalg.norm(x - y))


def polynomial_kernel(x, y, degree: int = 2):
    return (x @ y.T + 1) ** degree


def svm_dual_kernel(X, y, ker, arg):
    N = X.shape[0]
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :], arg)
    P = 0.5 * (P + P.T)
    P = 0.5 * P
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)
    alpha = qps.solve_qp(P, q, GG, h, solver='osqp')

    return alpha


def decision_function(alpha, X_train, y_train, X_test, ker, arg):
    N, n = X_train.shape
    z = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        z[i] = sum([alpha[k] * y_train[k] * ker(X_train[k, :].T, X_test[i, :], arg) for k in range(N)])
    return z


def predict(alpha, X_train, y_train, X_test, ker, arg):
    z = decision_function(alpha, X_train, y_train, X_test, ker, arg)
    return np.sign(z)


def errorScore(y_pred, y_test):
    res = 0
    for pred, real in zip(y_pred, y_test):
        if pred == real:
            res += 1
    return 1 - (res / len(y_pred))


def error_plot(error, title):
    error_plot = np.array(error)
    plt.plot(error_plot[:, 0], error_plot[:, 1], marker='o', linestyle='-')
    plt.title(f"Error Rate of {title} Kernel")
    plt.grid(False)
    plt.show()


def plot_classifier_z_kernel(alpha, X, y, ker, arg, threshold=0.1):
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
        z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]]), arg) for k in range(N)])
    threshold = np.mean(alpha)
    support_vectors = np.where((np.abs(alpha) >= threshold))
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='black')
    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
    plt.show()



if __name__ == "__main__":
    with open("..\simple_nonlin_classification.csv") as file:
        df = pd.read_csv(file)
    X = df.drop(columns=['y']).values  # Drop the "y" column from features
    y = df['y'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) # split the dataset train(80%) test(20%)

    gamma_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 20, 30, 50]

    error_RBF = []
    best_gamma = 0
    best_accuracy = 1
    for gamma in gamma_list:
        alpha = svm_dual_kernel(X_train, y_train, RBF_kernel, gamma)
        pred = predict(alpha, X_train, y_train, X_test, RBF_kernel, gamma)
        accur = errorScore(pred, y_test)
        if (accur < best_accuracy):
            best_accuracy = accur
            best_gamma = gamma

        error_RBF.append([gamma, accur])
        print("The Gamma is: ", gamma, "and score is ", 1 - accur)
    error_plot(error_RBF, f"RBF")

    alpha = svm_dual_kernel(X_train, y_train, RBF_kernel, best_gamma)
    plot_classifier_z_kernel(alpha, X_train, y_train, RBF_kernel, best_gamma)

    error_poly = []
    best_degree = 0
    best_accuracy_poly = 1
    for degree in range(2, 11):
        alpha = svm_dual_kernel(X_train, y_train, polynomial_kernel, degree)
        pred = predict(alpha, X_train, y_train, X_test, polynomial_kernel, degree)
        accur = errorScore(pred, y_test)
        if (accur < best_accuracy_poly):
            best_accuracy_poly = accur
            best_degree = degree

        error_poly.append([degree, accur])
        print("The Degree is: ", degree, "and score is ", 1 - accur)
    error_plot(error_poly, f"Poly")

    alpha = svm_dual_kernel(X_train, y_train, polynomial_kernel, best_degree)
    plot_classifier_z_kernel(alpha, X_train, y_train, polynomial_kernel, best_degree)
