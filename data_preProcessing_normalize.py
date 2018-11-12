# Structure the concrete data

import re
import numpy as np
from xlrd import open_workbook
from scipy.stats import zscore
from sklearn.preprocessing import normalize

# Reading the excel sheet
data = open_workbook('../Concrete_Data.xls').sheet_by_index(0)

# Extracted attributes
attributes = data.row_values(0)
attributeNames = [re.findall('[\w\s]+', x)[0] for x in attributes][:-1]

y = np.asarray(data.col_values(-1, 1))
X = np.empty((1030, 8))
for i in range(len(attributeNames)):
    X[:, i] = np.asarray(data.col_values(i, 1))

X = zscore(X)
# X = normalize(X)

# Compute values of N, M
N = len(X[:, 0])
M = len(attributeNames)
C = len(y)