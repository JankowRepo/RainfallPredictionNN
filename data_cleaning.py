# File prepared from data visualization work
# Serves as a module to perform data for machine learning algorithms

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, LabelEncoder
from imblearn.combine import SMOTETomek
from scipy import stats


def choose_data(balanced, split_size):
    df = pd.read_csv('weatherAUS.csv')

    df = df.drop(['Location', 'Temp9am', 'Date', 'Sunshine',
                  'Cloud9am', 'Cloud3pm', 'Evaporation'],
                 axis=1).dropna()

    le = LabelEncoder()
    df = df.apply(le.fit_transform)

    df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]

    X = df.iloc[:, :-1]
    y = df['RainTomorrow']

    splitted_data = data_split(X, y, balanced, split_size)

    return splitted_data


def data_split(X, y, balanced, split_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    if balanced:
        X_train, y_train = SMOTETomek(sampling_strategy={0: np.count_nonzero(y_train.values == 0),
                                                         1: int(np.count_nonzero(y_train.values == 0)/2)
                                                         }).fit_resample(X_train, y_train)

    print("Train data:", len(X_train))
    print("Test data:", len(X_test))

    return [X_train, X_test, y_train, y_test]
