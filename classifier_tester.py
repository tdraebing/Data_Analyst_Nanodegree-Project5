# -*- coding: utf-8 -*-

#Imports

#system file handling
import os
import sys
import pickle

#path to support scripts
sys.path.append("./tools/")

#data structure
import pandas as pd

#mathmatical operations
import numpy as np

#machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import  StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC


PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

#MinMaxScaler
#own implementaion to be able to keep NaNs
def minmax(features_train, features_test):
    features_train_scaled = features_train.copy()
    features_test_scaled = features_test.copy()
    for col in features_train.columns:
        #calculating constants for each feature
        col_min = np.nanmin(features_train[col], axis = None)[0]
        col_max = np.nanmax(features_train[col], axis = None)[0]
        if (col_max - col_min) == 0:
            train_scaled = features_train[col]
        else:
            train_std = [((X - col_min) / (col_max - col_min)) for X in features_train[col]]
            test_std = [((X - col_min) / (col_max - col_min)) for X in features_test[col]]
            
            train_std_min = np.nanmin(train_std, axis = None)
            train_std_max = np.nanmax(train_std, axis = None)
            
            test_std_min = np.nanmin(test_std, axis = None)
            test_std_max = np.nanmax(test_std, axis = None)
            #applying scaling to each entry
            train_scaled = [(X * (train_std_max - train_std_min) + train_std_min) for X in train_std] 
            test_scaled = [(X * (test_std_max - test_std_min) + test_std_min) for X in test_std]
        
        features_train_scaled[col] = train_scaled
        features_test_scaled[col] = test_scaled
    return features_train_scaled, features_test_scaled

#data preprocessing
def prepare_data(data_dict):
    #delete TOTAL-entry
    del data_dict['TOTAL']
    #create dataframe
    df = pd.DataFrame(data_dict).transpose()
    #delete email _adress variable
    del df['email_address']
    #Transform values to float
    df = df.replace('%','',regex=True).astype('float')
    #create binary variable for restricted_stock_deferred
    df['restricted_stock_deferred_bin'] = np.zeros((len(df.index), 1)) + 1
    df.loc[df['restricted_stock_deferred'].isnull(), 
           'restricted_stock_deferred_bin'] = 0
    #delete variables with large amounts of NaN
    df = df.drop(['restricted_stock_deferred'], 1)
    #remove outliers
    col_ol = ['shared_receipt_with_poi',
                'to_messages']
    for col in col_ol:
        index = df[np.abs(df[col] - df[col].mean()) \
              > 3 * df[col].std()][col].index.tolist()
        for i in index:
            df[i,col] = np.NaN
    #reduce variables
    features_list = ['poi', 
                'bonus',
                'deferred_income',
                'exercised_stock_options',
                'expenses',
                'long_term_incentive',
                'other',
                'restricted_stock',
                'salary',
                'restricted_stock_deferred_bin',
                'shared_receipt_with_poi',
                'to_messages']
    df_reduced = df.copy()[features_list]
    return df_reduced

#data preprocessing    
def preprocess_data(features_train, features_test):
    #transformation
    to_add_one = ['deferred_income',
                'shared_receipt_with_poi']
    features_train[to_add_one] = features_train[to_add_one] + 1
    features_test[to_add_one] = features_test[to_add_one] + 1
    to_transform = ['bonus',
                    'deferred_income',
                    'exercised_stock_options', 
                    'long_term_incentive', 
                    'other',
                    'restricted_stock',
                    'shared_receipt_with_poi',
                    'to_messages']
    features_train[to_transform] = features_train[to_transform].apply(np.absolute).apply(np.log10)
    features_test[to_transform] = features_test[to_transform].apply(np.absolute).apply(np.log10)
    
    #scaling
    features_train, features_test = minmax(features_train, features_test)
    
    #Imputating NaNs
    features_train = features_train.fillna(0)
    features_test = features_test.fillna(0)
    return pd.DataFrame(features_train), pd.DataFrame(features_test)
    
def evaluate_clf(df, clf_list, cv): 
    features = df.copy().drop('poi', axis=1)
    labels = df['poi']
    eval_dict = {}
    for c in clf_list:
        eval_dict[c] = {}
        print(c)
        clf = clf_list[c]
        true_negatives = 0
        false_negatives = 0
        true_positives = 0
        false_positives = 0
        for train_index, test_index in cv:
            features_train = features.iloc[train_index] 
            features_test = features.iloc[test_index]
            labels_train = labels[train_index]
            labels_test = labels[test_index]
            features_train_pre, features_test_pre = preprocess_data(features_train, features_test)
            ### fit the classifier using training set, and test on test set
            clf.fit(features_train_pre, labels_train)
            predictions = clf.predict(features_test_pre)
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
            accuracy = 1.0*(true_positives + true_negatives)/total_predictions
            precision = 1.0*true_positives/(true_positives+false_positives)
            recall = 1.0*true_positives/(true_positives+false_negatives)
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
            eval_dict[c]['accuracy'] = accuracy
            eval_dict[c]['precision'] = precision
            eval_dict[c]['recall'] = recall
            eval_dict[c]['f1'] = f1
            eval_dict[c]['f2'] = f2
            print clf
            print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
            print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
            print ""
        except:
            print "Got a divide by zero when trying out:", clf
            print "Precision or recall may be undefined due to a lack of true positive predicitons."
        
    return eval_dict        
        
def main():
    #load data
    with open("./data/final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        
    #preprocess
    df_preproc = prepare_data(data_dict)
    
    #split into training and test set
    cv = StratifiedShuffleSplit(df_preproc['poi'], 1000, random_state = 42)
    
    
    #create classifiers
    clf_list = {}
#    clf_list['GaussianNB'] = GaussianNB()
#    clf_list['RandomForestVanilla'] = RandomForestClassifier()
#    clf_list['AdaBoostVanilla'] = AdaBoostClassifier()
#    clf_list['svmVanilla'] = SVC()
#    clf_list['RandomForest'] = RandomForestClassifier(n_estimators = 100, 
#                                                        criterion = 'gini', 
#                                                        max_features = 'auto')
#    clf_list['RandomForest_mss10'] = RandomForestClassifier(n_estimators = 100, 
#                                                        criterion = 'gini', 
#                                                        max_features = 'auto',
#                                                        min_samples_split = 10)
#    clf_list['RandomForest_mss20'] = RandomForestClassifier(n_estimators = 100, 
#                                                        criterion = 'gini', 
#                                                        max_features = 'auto',
#                                                        min_samples_split = 20)
#    clf_list['RandomForest_mss40'] = RandomForestClassifier(n_estimators = 100, 
#                                                        criterion = 'gini', 
#                                                        max_features = 'auto',
#                                                        min_samples_split = 40)
#    clf_list['RandomForest_mss80'] = RandomForestClassifier(n_estimators = 100, 
#                                                        criterion = 'gini', 
#                                                        max_features = 'auto',
#                                                        min_samples_split = 80)
#                                                                                                
#    clf_list['svm'] = SVC(C = 1.0,
#                        kernel = 'poly')
#                        
#    
#    clf_list['AdaBoost025'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=0.25)
#    clf_list['AdaBoost05'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=0.5)
#    clf_list['AdaBoost075'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=0.75)  
#    clf_list['AdaBoost1'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost125'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=1.25)
#    clf_list['AdaBoost15'] = AdaBoostClassifier(n_estimators=100,
#                                                learning_rate=1.5)
#    clf_list['AdaBoost075-10'] = AdaBoostClassifier(n_estimators=10,
#                                                learning_rate=0.75)
#    clf_list['AdaBoost075-50'] = AdaBoostClassifier(n_estimators=50,
#                                                learning_rate=0.75)
#    clf_list['AdaBoost075-250'] = AdaBoostClassifier(n_estimators=250,
#                                                learning_rate=0.75)  
#    clf_list['AdaBoost075-125'] = AdaBoostClassifier(n_estimators=125,
#                                                learning_rate=0.75)
#    clf_list['AdaBoost075-150'] = AdaBoostClassifier(n_estimators=150,
#                                                learning_rate=0.75)  
#    clf_list['AdaBoost075-175'] = AdaBoostClassifier(n_estimators=175,
#                                                learning_rate=0.75)  
#    clf_list['AdaBoost'] = AdaBoostClassifier(n_estimators=100,
#                                          learning_rate=0.75,
#                                          algorithm="SAMME") 
#    clf_list['AdaBoost1-10'] = AdaBoostClassifier(n_estimators=10,
#                                                learning_rate=1)
#    clf_list['AdaBoost1-50'] = AdaBoostClassifier(n_estimators=50,
#                                                learning_rate=1)
#    clf_list['AdaBoost1-250'] = AdaBoostClassifier(n_estimators=250,
#                                                learning_rate=1)  
#    clf_list['AdaBoost1-125'] = AdaBoostClassifier(n_estimators=125,
#                                                learning_rate=1)
#    clf_list['AdaBoost1-150'] = AdaBoostClassifier(n_estimators=150,
#                                                learning_rate=1)  
#    clf_list['AdaBoost1-175'] = AdaBoostClassifier(n_estimators=175,
#                                                learning_rate=1)   
#    clf_list['AdaBoost1'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost2'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost4'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost8'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost16'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=16),
#                                                n_estimators=100,
#                                                learning_rate=1) 
#    clf_list['AdaBoost24'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=24),
#                                                n_estimators=100,
#                                                learning_rate=1) 
#    clf_list['AdaBoost32'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=32),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoost64'] = AdaBoostClassifier(DecisionTreeClassifier(max_depth=64),
#                                                n_estimators=100,
#                                                learning_rate=1)                                                
#    clf_list['AdaBoostmsl2'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=2),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoostmsl4'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=4),
#                                                n_estimators=100,
#                                                learning_rate=1)                                            
#    clf_list['AdaBoostmsl6'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=6),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoostmsl8'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=8),
#                                                n_estimators=100,
#                                                learning_rate=1)                                            
#    clf_list['AdaBoostmsl10'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=10),
#                                                n_estimators=100,
#                                                learning_rate=1)
#    clf_list['AdaBoostmsl25'] = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=25),
#                                                n_estimators=100,
#                                                learning_rate=1)    

    clf_list['final'] = AdaBoostClassifier(n_estimators=100,
                                            learning_rate=1)                                        
    eval_list = evaluate_clf(df_preproc, clf_list, cv)                       
    print(eval_list)
    
    

if __name__ == '__main__':
    main()
    



