"""
    Script to load the classifier created by the classifier_tester.py script and print the parameters found by the grid search.
"""

import pickle
from pprint import pprint

# enter the filename of the pickle-dumped classifier here
filename = 'data/clf_logit_finalpca.pkl'

with open(filename, "r") as clf_infile:
    clf = pickle.load(clf_infile)

pprint(clf.best_estimator_)
print(clf.scorer_)
print(clf.best_score_)
pprint(clf.best_params_)
