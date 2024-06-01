import qpsolvers as qps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plotSVM(X, y, w=None, support_vectors=None, title:str = None):
    red = np.where(y < 0)
    blue = np.where(y > 0)

    plt.plot(X[red, 0], X[red, 1], 'o', color='red')
    plt.plot(X[blue, 0], X[blue, 1], 'o', color='blue')

    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])  # set size of axis
    lx = np.linspace(x_min, x_max)

    if w is not None:  # Classifier visualization
        ly = [(-w[-1] - w[0] * p) / w[1] for p in
              lx]  # Seperator Y coordinates from the equation w_1*x + w_2*y + w_3 = 0
        ly1 = [(-w[-1] - w[0] * p - 1) / w[1] for p in lx]  # Margin Red line
        ly2 = [(-w[-1] - w[0] * p + 1) / w[1] for p in lx]  # Margin Blue line
        plt.plot(lx, ly, color='black')
        plt.plot(lx, ly1, "--", color='red')
        plt.plot(lx, ly2, "--", color='blue')

    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=100, facecolors='none', edgecolors='black')
    plt.title(title)
    plt.show()

def DualSVM(X, y,threshold = 0.1):
    N, n = X.shape
    X = np.c_[X, np.ones(N)]
    old_G = np.diag(y) @ X
    P = old_G@old_G.T
    q = -np.ones(N)
    G = -np.eye(N)
    h = np.zeros(N)
    alpha = qps.solve_qp(P, q, G, h, solver='osqp')
    support_Index = np.where((np.abs(alpha) >= threshold))

    w = old_G.T @ alpha
    return w , support_Index



def QuadSVM(X, y):
    N, n = X.shape
    X = np.c_[X, np.ones(N)]
    P = 2 * np.eye(n + 1)
    q = np.zeros(n + 1)
    G = -np.diag(y) @ X
    h = -np.ones(N)
    w = qps.solve_qp(P, q, G, h, solver='osqp')
    return w



if __name__ == "__main__":
    with open("..\simple_classification.csv") as file:
        df = pd.read_csv(file)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = 2 * y - 1

    w = QuadSVM(X, y)
    print("Quadratic weights:", w)
    plotSVM(X, y, w, title = "Quadratic weights")

    w_d, support_vector = DualSVM(X, y)
    print("Dual weights:", w_d)
    plotSVM(X, y, w_d, support_vector,"Dual Program")
