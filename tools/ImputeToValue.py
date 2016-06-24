from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd


class ImputeToValue(TransformerMixin, BaseEstimator):
    def __init__(self, value=0):
        self.value = value

    def fit(self, *_):
        return self

    def transform(self, X, y=None, *_):
        return X.fillna(self.value)
