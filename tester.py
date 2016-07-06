#!/usr/bin/pickle

"""
    a basic script for importing student's POI identifier,
    and checking the results that they get from it 
 
    requires that the algorithm, dataset, and features list
    be written to my_classifier.pkl, my_dataset.pkl, and
    my_feature_list.pkl, respectively

    that process should happen at the end of poi_id.py

    modified to be able to deal with pandas data frames and use the classifier pipeline.
"""

import pickle
import sys
from sklearn.cross_validation import StratifiedShuffleSplit

sys.path.append("./tools/")

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def test_classifier(clf, dataset, folds=1000):
    labels = dataset['poi']
    features = dataset.drop(['poi'], 1)
    cv = StratifiedShuffleSplit(labels, folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for train_idx, test_idx in cv:
        features_train = features.iloc[train_idx]
        features_test = features.iloc[test_idx]
        labels_train = labels.iloc[train_idx]
        labels_test = labels.iloc[test_idx]

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)

        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                           true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


CLF_PICKLE_FILENAME = "./data/my_classifier.pkl"
DATASET_PICKLE_FILENAME = "./data/my_dataset.pkl"
FEATURE_LIST_FILENAME = "./data/my_feature_list.pkl"


def dump_classifier_and_data(clf, dataset):
    with open(CLF_PICKLE_FILENAME, "w") as clf_outfile:
        pickle.dump(clf, clf_outfile)
    with open(DATASET_PICKLE_FILENAME, "w") as dataset_outfile:
        pickle.dump(dataset, dataset_outfile)


def load_classifier_and_data():
    with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
        clf = pickle.load(clf_infile)
    with open(DATASET_PICKLE_FILENAME, "r") as dataset_infile:
        dataset = pickle.load(dataset_infile)
    return clf, dataset


def main():
    # load up student's classifier, dataset, and feature_list
    clf, dataset = load_classifier_and_data()
    # Run testing script
    test_classifier(clf, dataset)


if __name__ == '__main__':
    main()
