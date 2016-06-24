# MinMaxScaler
# own implementation to be able to keep NaNs
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class MinMaxNA(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.col_min = dict()
        self.col_max = dict()

    @staticmethod
    def __get_minmax(column):
        col_min = np.nanmin(column, axis=None)[0]
        col_max = np.nanmax(column, axis=None)[0]
        return col_min, col_max

    @staticmethod
    def __apply_scale(column, col_min, col_max):
        if (col_max - col_min) == 0:
            return np.zeros(len(column))
        else:
            std = [((X - col_min) / (col_max - col_min)) for X in column]

            std_min = np.nanmin(std, axis=None)
            std_max = np.nanmax(std, axis=None)

            # applying scaling to each entry
            return [(X * (std_max - std_min) + std_min) for X in std]

    def transform(self, X, y=None, *_):
        X_scaled = X.copy()
        for col in X.columns:
            X_scaled[col] = self.__apply_scale(X[col], self.col_min[col], self.col_max[col])
        return X_scaled

    def fit(self, X, y=None, *_):
        for col in X.columns:
            mi, ma = self.__get_minmax(X[col])
            self.col_min[col] = mi
            self.col_max[col] = ma
        return self
