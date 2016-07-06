#!/usr/bin/python
import pickle
from tester import dump_classifier_and_data
from tools.prepareData import prepare_data
from tools.customTransformers import ImputeToValue, LogTransform, MinMaxNA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# load data
with open("./data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# preprocess
df = prepare_data(data_dict)

steps_logit = [('impute nans', ImputeToValue()),
               ('log transforming', LogTransform(True)),
               ('minmax scaling', MinMaxNA()),
               ('kbest', SelectKBest(k=15)),
               ('pca', PCA(n_components=8)),
               ('classify', LogisticRegression(C=1, class_weight='balanced', penalty='l1'))]

clf = Pipeline(steps_logit)
'''
'''
# Tested pipelines that were npt chosen as the final classifier
'''
from sklearn.linear_model import SGDClassifier
steps_sgd = [('impute nans', ImputeToValue()),
             ('log transforming', LogTransform(True)),
             ('minmax scaling', MinMaxNA()),
             ('kbest', SelectKBest(k=16)),
             ('pca', PCA(n_components=15)),
             ('classify', SGDClassifier(alpha=0.01, class_weight='balanced', loss='log', penalty='l1', n_iter=6))]

clf = Pipeline(steps_sgd)

from sklearn.ensemble import AdaBoostClassifier
steps_adaboost = [('impute nans', ImputeToValue()),
                  ('log transforming', LogTransform(True)),
                  ('minmax scaling', MinMaxNA()),
                  ('kbest', SelectKBest(k=3)),
                  ('pca', PCA(n_components=None)),
                  ('classify', AdaBoostClassifier(n_estimators=160))]

clf = Pipeline(steps_adaboost)
'''

# save the classifier pipeline and preprocessed data
dump_classifier_and_data(clf, df)
