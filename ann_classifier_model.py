# New classifier model with two layered cross validation

from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import dbplotf
import numpy as np
import sklearn.neural_network as nn
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import figure, plot, subplot, title, show, bar, xticks, tight_layout, xlabel, ylabel, legend
from toolbox_02450 import confmatplot
from scipy.stats import zscore

from data_preProcessing_interval_normalize import *

Y = zscore(X,0)
# Normalize and compute PCA (UNCOMMENT to experiment with PCA preprocessing)
U,S,V = np.linalg.svd(Y,full_matrices=False)
V = V.T
#Components to be included as features
k_pca = 2
X = X @ V[:,0:k_pca]
N, M = X.shape


# CV folds
# Outer loop
K = 5
CV_1 = StratifiedKFold(K, shuffle=False)

# Inner loop
J = 10
CV_2 = StratifiedKFold(J, shuffle=False)

# Parameters for neural network
n_hidden_neurons = 2
n_train = 8

k = 0
l = 0
train_errors_inner = np.ones(J*K)*np.inf
errors_inner = np.zeros(J*K)*np.nan
errors_outer = np.zeros(K)*np.nan
best_clf_inner = list()
best_clf_outer = list()
hidden_neurons = list()
best_hidden_neurons = list()
y_est_outer = list()
y_test_outer = list()
X_test_outer = list()
for train_index, test_index in CV_1.split(X, y):
    print('\nCrossvalidation fold one: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    j = 0
    for train_index_2, test_index_2 in CV_2.split(X_train, y_train):
        print('\nCrossvalidation fold one: {0}/{1}'.format(j + 1, J))

        X_train_2 = X_train[train_index_2, :]
        y_train_2 = y_train[train_index_2]
        X_test_2 = X_train[test_index_2, :]
        y_test_2 = y_train[test_index_2]


        for i in range(n_train):
            clf = nn.MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(n_hidden_neurons+i, ), random_state=1)
            clf.fit(X_train_2, y_train_2)

            mis_classified = np.sum(clf.predict(X_train_2) != y_test_2)
            relative_error = mis_classified/len(y_test_2)
            # print("The relative error in misclassification made {0}% with {1} hidden neurons".format(relative_error ,n_hidden_neurons+i))

            if i == 0:
                best_clf_inner.append(clf)
                hidden_neurons.append(n_hidden_neurons+i)

            if relative_error < train_errors_inner[l]:
                train_errors_inner[l] = relative_error
                best_clf_inner[l] = clf
                hidden_neurons[l] = n_hidden_neurons+i

        best_index_inner = train_errors_inner.argmin()
        mis_classified = np.sum(best_clf_inner[l].predict(X_test_2) != y_test_2)
        relative_error = mis_classified / len(y_test_2)
        print("Relative test error for the best fitted model with {0} hidden neurons".format(hidden_neurons[best_index_inner]))
        errors_inner[l] = relative_error
        j += 1
        l += 1

    # Refitting the model from the inner loop
    best_model_index = errors_inner[:2*(k+1)].argmin()

    clf_best = nn.MLPClassifier(solver='lbfgs', alpha=1e-4,
                                hidden_layer_sizes=(hidden_neurons[best_model_index], ), random_state=1)
    clf_best.fit(X_train, y_train)
    y_est = clf.predict(X_test)
    y_est_outer.append(y_est)
    y_test_outer.append(y_test)
    X_test_outer.append(X_test)
    mis_classified = np.sum(clf.predict(X_test) != y_test)
    relative_error = mis_classified / len(y_test)
    errors_outer[k] = relative_error
    best_clf_outer.append(clf_best)
    best_hidden_neurons.append(hidden_neurons[best_model_index])

    k += 1

best_index_total = errors_outer.argmin()

gen_error = np.mean(errors_outer).round(3)
print("The generalization error is {0}% misclassification".format(gen_error*100))

figure(figsize=(6,7));
plot; bar(range(0,K),errors_outer); title('Square Errors for best performing models from the inner loop');
xticks(np.arange(K), (best_hidden_neurons[0],
                      best_hidden_neurons[1],
                      best_hidden_neurons[2]))
xlabel('Hidden Neurons')
ylabel('Square Error')

# The best confusion matrix for the ann selection
figure();
confmatplot(y_test_outer[best_index_total], y_est_outer[best_index_total])

# neval = np.argmax(best_clf_outer[best_index_total].predict_proba(X_test_outer[best_index_total]),1)
# Doing the decision boundaries
def neval(xval):
    return np.argmax(best_clf_outer[best_index_total].predict_proba(xval), 1)

if k_pca == 2:
    figure();
    dbplotf(X_test_outer[best_index_total], y_test_outer[best_index_total], neval, 'auto')





