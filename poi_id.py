#!/usr/bin/python
import pickle
import sys

sys.path.append("./tools/")

from tester import dump_classifier_and_data
from prepareData import prepare_data
from tools import ImputeToValue, LogTransform, MinMaxNA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# load data
with open("./data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# preprocess
df = prepare_data(data_dict)

steps = [('log transforming', LogTransform.LogTransform()),
         ('impute nans', ImputeToValue.ImputeToValue()),
         ('minmax scaling', MinMaxNA.MinMaxNA()),
         ('kbest', SelectKBest(k=14)),
         ('pca', PCA(n_components='mle', whiten=False)),
         ('classify', LogisticRegression(tol=1e-16, class_weight='balanced', C=1))]

clf = Pipeline(steps)

dump_classifier_and_data(clf, df)