import pandas as pd
from SVM import SVM

if __name__ == '__main__':
    with open("..\Processed Wisconsin Diagnostic Breast Cancer.csv") as file:
        df = pd.read_csv(file)
    y = df['diagnosis']
    X = df.drop('diagnosis', axis=1)
    y = y * 2 - 1  # Converting the tags to 1, -1

    # Lists of parameters to check
    degree_array = [3, 4, 5]
    #degree_array = [1, 2, 3, 4, 5]
    gamma_array = [0.001, 0.01, 0.1, 1, 10]
    C = [0.01, 0.1, 1, 10, 100]

    X_train, X_test, y_train, y_test = SVM.train_test_split(X, y)  # split the dataset train(80%) test(20%)

    poly_error = []
    for degree in degree_array:
        for c in C:
            svm = SVM('poly', degree, c)
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            poly_error.append([(1 - score), degree, c])
            print(len(svm.support_vectors_), X_train.shape[0], degree, c)
    SVM.error_heatmap_plot(poly_error, "poly")

    RBF_error = []
    for gamma in gamma_array:
        for c in C:
            svm = SVM("RBF", 1, c, gamma)
            svm.fit(X_train, y_train)
            score = svm.score(X_test, y_test)
            RBF_error.append([(1 - score), gamma, c])
    SVM.error_heatmap_plot(RBF_error, "RBF")
