from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
from load_datesets import load_datasets

X, y = load_datasets()
X_test = X[650:]
y_test = y[650:]
X = X[0:650]
y = y[0:650]


def load_test_data():
    return X_test, y_test


def load_train_data(test_split=None, use_cross_validation=None, k_fold=None):
    if use_cross_validation:
        data = []
        sfolder = StratifiedKFold(n_splits=k_fold, random_state=1)
        y_label = np.argmax(y, axis=1)
        for train_index, valid_index in sfolder.split(X, y_label):
            X_train, X_valid, y_train, y_valid = X[train_index], X[valid_index], y[train_index], y[valid_index]
            data_tmp = (X_train, X_valid, y_train, y_valid)
            data.append(data_tmp)
        return data
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_split, random_state=1)

        return X_train, X_valid, y_train, y_valid
