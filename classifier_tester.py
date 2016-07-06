# -*- coding: utf-8 -*-

# Imports

# system file handling
import pickle
import sys

# mathematical operations
import numpy as np

# machine learning and data preprocessing
from tools.customTransformers import ImputeToValue, LogTransform, MinMaxNA
from tools.prepareData import prepare_data
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


def main():
    # load data
    with open("./data/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    # preprocess
    df = prepare_data(data_dict)

    # split into training and test set
    cv = StratifiedShuffleSplit(df['poi'], 100)

    '''
    For each classifier two pipelines were created. That was necessary because without knowing the number of features
    after applying SelectKBest the maximum of n_components for pca cannot be set. Thus in a first grid search the n_components
    parameter of the PCA is set to None to include all components, while the other parameters are optimized. In the second
    following grid search the parameters found in the first run were set and the n_components parameter optimized to this set
    of parameters.
    '''

    # create classifiers and parameter grids
    sgd = {'classifier': SGDClassifier(n_jobs=4),
           'parameters': {'kbest__k': np.arange(1, 20, 1).tolist(),
                          'pca__n_components': [None],
                          'classify__class_weight': ['balanced', None],
                          'classify__loss': ['log', 'hinge'],
                          'classify__penalty': ['l2', 'l1', 'elasticnet', 'none'],
                          'classify__alpha': [0.0001, 0.001, 0.01, 0.1]}}

    sgd_pca = {'classifier': SGDClassifier(n_jobs=4),
               'parameters': {'kbest__k': [16],
                              'pca__n_components': [None] + np.arange(1, 16, 1).tolist(),
                              'classify__class_weight': ['balanced'],
                              'classify__loss': ['log'],
                              'classify__penalty': ['l1'],
                              'classify__alpha': [0.001, 0.01, 0.1]}}

    ada = {'classifier': AdaBoostClassifier(),
           'parameters': {'kbest__k': np.arange(1, 20, 1).tolist(),
                          'pca__n_components': [None],
                          'classify__n_estimators': np.arange(20, 200, 20).tolist(),
                          'classify__base_estimator': [DecisionTreeClassifier(),
                                                       RandomForestClassifier()]}}

    ada_pca = {'classifier': AdaBoostClassifier(),
               'parameters': {'kbest__k': [3],
                              'pca__n_components': [None] + np.arange(1, 3, 1).tolist(),
                              'classify__n_estimators': [160, 180, 200],
                              'classify__base_estimator': [DecisionTreeClassifier()]}}

    logit = {'classifier': LogisticRegression(),
             'parameters': {'kbest__k': np.arange(1, 20, 1).tolist(),
                            'pca__n_components': [None],
                            'classify__class_weight': ['balanced', None],
                            'classify__penalty': ['l2', 'l1'],
                            'classify__C': [0.001, 0.01, 0.1, 1, 10]}}

    logit_pca = {'classifier': LogisticRegression(),
                 'parameters': {'kbest__k': [15],
                                'pca__n_components': [None] + np.arange(1, 15, 1).tolist(),
                                'classify__class_weight': ['balanced'],
                                'classify__penalty': ['l1'],
                                'classify__C': [0.1, 1, 10]}}

    # list of classifiers to optimize
    classifiers = {'sgd_finalpca': sgd_pca,
                   'ada_finalpca': ada_pca,
                   'logit_finalpca': logit_pca}

    # split features and labels
    features = df.drop(['poi'], 1)
    labels = df['poi']

    # apply a gridsearch for each classifier
    for c in classifiers:
        print(c)
        steps = [('impute nans', ImputeToValue()),
                 ('log transforming', LogTransform(True)),
                 ('minmax scaling', MinMaxNA()),
                 ('kbest', SelectKBest()),
                 ('pca', PCA()),
                 ('classify', classifiers[c]['classifier'])]

        pipeline = Pipeline(steps)

        clf = GridSearchCV(pipeline,
                           classifiers[c]['parameters'],
                           scoring='precision',
                           cv=cv,
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
