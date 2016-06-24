from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class LogTransform(TransformerMixin, BaseEstimator):
    def __init__(self, test_skewness=True):
        self.test_skewness = test_skewness
        self.skewness = pd.Series()
        self.to_add_one = []

    def fit(self, X, y=None, *_):
        if self.test_skewness:
            self.skewness = X.skew()
        for col in X.columns:
            if 0 in X[col]:
                self.to_add_one.append(col)
        return self

    def transform(self, X, y=None, *_):
        temp = X.copy()
        temp[self.to_add_one] += 1
        temp = temp.apply(np.absolute).apply(np.log10)
        if not self.test_skewness:
            return temp
        else:
            temp_skewness = temp.skew()
            for s in self.skewness.index:
                if np.absolute(self.skewness[s]) >= np.absolute(temp_skewness[s]):
                    X.loc[:, s] = temp.loc[:, s]
            return X
