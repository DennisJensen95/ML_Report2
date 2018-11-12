"""
This script is for making an Artificial Neural Network for regression model of
compressive strength
"""

from matplotlib.pyplot import figure, plot, subplot, title, show, bar, xticks, tight_layout, xlabel, ylabel, legend
import numpy as np
from scipy.io import loadmat
import neurolab as nl
from sklearn import model_selection
from scipy import stats
import random

from data_preProcessing_normalize import *

# Parameters for neural network classifier
n_hidden_units = 1     # number of hidden units
n_train = 8           # number of networks trained in each k-fold
learning_goal = 1       # stop criterion 1 (train mse to be reached)
max_epochs = 300         # stop criterion 2 (max epochs in training)
show_error_freq = 50     # frequency of training status updates

# Getting the min max range for every attribute and the y vector
# minMaxRange = [[min(X[:, 0]), max (X[:, 0])], [min(X[:, 1]), max(X[:, 1])], [min(X[:, 2]), max(X[:,2])],
#                [min(X[:, 3]), max (X[:, 3])], [min(X[:, 4]), max(X[:, 4])], [min(X[:, 5]), max(X[:,5])],
#                [min(X[:, 6]), max(X[:, 6])], [min(X[:, 7]), max(X[:, 7])]]
minMaxRange = [[0, 1]]*M

# K-fold crossvalidation
# Outer loop
K = 5
# Inner Loop
J = 3
CV_1 = model_selection.KFold(K, shuffle=False)
CV_2 = model_selection.KFold(J, shuffle=False)

# Variable for classification error
errors = np.zeros((J, K))*np.nan
gen_errors = np.zeros(K)*np.nan
error_hist = np.zeros((max_epochs,K))*np.nan
bestnet = list()
best_bestnet = list()
k = 0
bestnet_hidden_units = np.zeros((J, K))*np.nan
y_best_est = []
best_performing_anns = []
best_hidden_neurons_outer = []
mean_errors = np.zeros(K)*np.nan
for train_index, test_index in CV_1.split(X, y):
    print('\nCrossvalidation fold one: {0}/{1}'.format(k + 1, K))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    j = 0
    for train_index_2, test_index_2 in CV_2.split(X_train, y_train):
        print('\nCrossvalidation fold two: {0}/{1}'.format(j + 1, J))

        # print(test_index_2)
        X_train_2 = X_train[train_index_2, :]
        y_train_2 = y_train[train_index_2]
        X_test_2 = X_train[test_index_2, :]
        y_test_2 = y_train[test_index_2]

        best_train_error = np.inf
        for i in range(n_train):
            print('Training network {0}/{1}...'.format(i + 1, n_train))

            ann = nl.net.newff(minMaxRange, [n_hidden_units+i, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
            # ann = nl.net.newff([[-3, 3]] * M, [n_hidden_units + i, j], [nl.trans.TanSig(), nl.trans.PureLin()])
            if i==0:
                bestnet.append(ann)

            # Train network
            train_error = ann.train(X_train_2, y_train_2.reshape(-1,1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
            if train_error[-1] < best_train_error:
                bestnet[j] = ann
                best_train_error = train_error[-1]
                error_hist[range(len(train_error)), k] = train_error
                bestnet_hidden_units[j, k] = n_hidden_units+i

        print('Best train error: {0}. ..'.format(best_train_error))
        y_est = bestnet[j].sim(X_test).squeeze()
        y_mean_est = np.mean(y_test)
        errors[j, k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]

        if j == 0:
            best_bestnet.append(bestnet[j])

        if errors[j, k] < min(errors[j, :]):
            best_bestnet[k] = bestnet[j]

        j += 1

    print('Best train error: {0}...'.format(best_train_error))

    #Index for the lowest error in each k fold
    best_index = errors[:, k].argmin()

    # Best peforming model from the inner loop
    ann = nl.net.newff(minMaxRange, [int(bestnet_hidden_units[best_index, k]), 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    best_hidden_neurons_outer.append(bestnet_hidden_units[best_index, k])
    # Train network
    train_error = ann.train(X_train, y_train.reshape(-1, 1), goal=learning_goal, epochs=max_epochs, show=show_error_freq)
    best_performing_anns.append(ann)
    y_est = ann.sim(X_test).squeeze()
    y_best_est.append(y_est)
    gen_errors[k] = np.power(y_est - y_test, 2).sum().astype(float) / y_test.shape[0]
    y_mean_est = np.mean(y_test)
    mean_errors[k] = np.power(y_mean_est - y_test, 2).sum().astype(float) / y_test.shape[0]
    k += 1

# Print the average least squares error
print('Mean-square error: {0}'.format(np.mean(gen_errors)))

best_index = gen_errors.argmin()

figure(figsize=(6,7));
plot; bar(range(0,K),gen_errors); title('Square Errors for best performing models from the inner loop');
xticks(np.arange(K), (best_hidden_neurons_outer[0],
                      best_hidden_neurons_outer[1],
                      best_hidden_neurons_outer[2],
                      best_hidden_neurons_outer[3],
                      best_hidden_neurons_outer[4],))
xlabel('Hidden Neurons')
ylabel('Square Error')

# bestnet_hidden_units[3],
# bestnet_hidden_units[4],
# bestnet_hidden_units[5],
# bestnet_hidden_units[6],
# bestnet_hidden_units[7],
# bestnet_hidden_units[8],
# bestnet_hidden_units[9])


# plot(error_hist); title('Training error as function of BP iterations'); xlabel('Epochs training')
# ylabel('Training Error')
# legend(['hidden neurons {0}'.format(bestnet_hidden_units[0]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[1]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[2]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[3]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[4]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[5]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[6]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[7]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[8]),
#         'hidden neurons {0}'.format(bestnet_hidden_units[9])])
# figure(figsize=(9,10));
plot(np.array(y_best_est[best_index])); plot(y_test); title('Best performing ANN with {0} hidden neurons: est_y vs. test_y'.format(bestnet_hidden_units[1, 3]));
xlabel('Observations, predicted')
ylabel('Compressive strength')
legend(['Predicted results', 'Test results'])
#
# plot((y_best_est[best_index]-y_test)); title('Best performing ANN with {0} hidden neurons: prediction error (est_y-test_y)'.format(bestnet_hidden_units[best_index]));
# xlabel('Observations, predicted')
# ylabel('Prediction Error Compressive Strength')
# tight_layout()
#
# show()
#
# print('Ran ann_regression_model')
#
# # % The weights if the network can be extracted via
# generalization_error = np.mean(errors)
# print("The total generalization error is {0}".format(generalization_error))
# v_first_layer = bestnet[best_index].layers[0].np['w'] # Get the weights of the first layer
# v_bias_first_layer = bestnet[best_index].layers[0].np['b'] # Get the bias of the first layer
