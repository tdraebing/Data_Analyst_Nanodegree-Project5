import pickle

filename = '../data/gridsearch results/logisticregression/clf_logRegsmall-wOut.pkl'

with open(filename, "r") as clf_infile:
    clf = pickle.load(clf_infile)

print(clf.best_estimator_)
print(clf.best_score_)
print(clf.best_params_)