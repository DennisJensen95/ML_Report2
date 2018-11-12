# Structure the concrete data

import re
import math
import numpy as np
from xlrd import open_workbook
from sklearn.preprocessing import normalize
from scipy.stats import zscore

# Reading the excel sheet
data = open_workbook('../Concrete_Data.xls').sheet_by_index(0)

# Extracted attributes
attributes = data.row_values(0)
attributes = [re.findall('[\w\s]+', x)[0] for x in attributes][:-1]

# Result vector of compressive strength
y = np.asarray((data.col_values(-1, 1)))
y_int = np.zeros((len(y), 1))

intervals = ['Weak', 'Strong']
classNames = ['Weak', 'Strong']
interval_value = math.ceil(max(y)/len(intervals))

for j in range(len(intervals)):
    # print('Interval: ' + str(j*interval_value) + ':' + str((j+1)*interval_value))
    for i in range(len(y)):
        if (y[i] >=j*interval_value and y[i] <= (j+1)*interval_value):
            y_int[i, 0] = j
            y[i] =  j


# Attributes matrix
attributeNames = list([re.findall('[\w\s]+', x)[0] for x in attributes])
X = np.empty((1030, 8))
for i in range(len(attributeNames)):
    X[:, i] = np.asarray(data.col_values(i, 1))

# Normalize data
X = zscore(X)

# Want y to be a matrix
# y = y_int

# Compute values of N, M
N = len(y)
M = len(attributeNames)
C = len(intervals)

