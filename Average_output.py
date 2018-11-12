
# from data_preProcessing_normalize import *
from sklearn.model_selection import StratifiedKFold
from data_preProcessing_interval_normalize import *

K = 5

CV = StratifiedKFold(K, shuffle=False)

mean_errors = np.zeros(K)
k = 0
for train_index, test_index in CV.split(X, y):
    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    strong_cement = np.sum(y==1)
    weak_cement = np.sum(y==0)
    if strong_cement > weak_cement:
        y_mean_est = 1
    else:
        y_mean_est = 0

    mis_classified = np.sum(y_mean_est != y_test)
    relative_error = mis_classified / len(y_test)
    mean_errors[k] = relative_error
    # y_mean_est = np.mean(y_test)
    # mean_errors[k] = np.square(y_test - y_mean_est).sum().astype(float) / y_test.shape[0]

    k += 1

print("Gen Error is {0}".format(np.mean(mean_errors)))