# -*- coding: utf-8 -*-

# Imports

# system file handling
import pickle
import sys

# data structure
import pandas as pd

# mathematical operations
import numpy as np

# machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from tools import ImputeToValue, LogTransform, MinMaxNA

# path to support scripts
sys.path.append("./tools/")

from prepareData import prepare_data

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def main():
    # load data
    with open("./data/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # preprocess
    df = prepare_data(data_dict)

    # split into training and test set
    cv = StratifiedShuffleSplit(df['poi'], 100, random_state=42)

    # create classifiers
    svm = {'classifier': SVC(),
           'parameters': {'classify__kernel': ('linear',
                                               'rbf',
                                               'poly'),
                          'classify__C': [1e-05, 1e-2, 1e-1, 1],
                          'kbest__k': np.arange(5, 19, 1).tolist(),
                          'pca__n_components': [.25, .5, .75, 1, 2],
                          'pca__whiten': [True, False]}}
    randomforest = {'classifier': RandomForestClassifier(),
                    'parameters': {'classify__n_estimators': np.arange(20, 200, 30).tolist(),
                                   'classify__min_samples_split': [2, 8, 32, 128],
                                   'kbest__k': np.arange(5, 19, 1).tolist(),
                                   'pca__n_components': [.25, .5, .75, 1, 2],
                                   'pca__whiten': [True, False]}}
    adaboost = {'classifier': AdaBoostClassifier(),
                'parameters': {'classify__n_estimators': np.arange(20, 200, 20).tolist(),
                               'classify__base_estimator': [DecisionTreeClassifier(),
                                                            RandomForestClassifier()],
                               'kbest__k': np.arange(5, 19, 1).tolist(),
                               'pca__n_components': [.25, .5, .75, 1, 2],
                               'pca__whiten': [True, False]}}

    logisticregr = {'classifier': LogisticRegression(),
                    'parameters': {'classify__C': [1e-05, 1e-2, 1e-1, 1],
                                   'classify__class_weight': ['balanced'],
                                   'classify__tol': [1e-2, 1e-8, 1e-16, 1e-64, 1e-256],
                                   'kbest__k': np.arange(5, 19, 1).tolist(),
                                   'pca__n_components': [.25, .5, .75, 1, 2],
                                   'pca__whiten': [True, False]}}

    # Add classifiers to be tested into the dictionary
    classifiers = {'clf_logreg_final_precision': logisticregr}

    # split features and labels
    features = df.drop(['poi'], 1)
    labels = df['poi']

    # apply a gridsearch for each classifier
    for c in classifiers:
        print(c)
        steps = [('log transforming', LogTransform.LogTransform()),
                 ('impute nans', ImputeToValue.ImputeToValue()),
                 ('minmax scaling', MinMaxNA.MinMaxNA()),
                 ('kbest', SelectKBest()),
                 ('pca', PCA()),
                 ('classify', classifiers[c]['classifier'])]

        pipeline = Pipeline(steps)

        clf = GridSearchCV(pipeline,
                           classifiers[c]['parameters'],
                           scoring='precision',
                           cv=cv,
                           verbose=10,
                           n_jobs=4)

        clf.fit(features, labels)
        print('best')
        print(clf.best_estimator_)
        print(clf.best_score_)

        clf_outfile = './data/clf_' + c + '.pkl'
        with open(clf_outfile, 'wb') as f:
            pickle.dump(clf, f)

if __name__ == '__main__':
    main()
