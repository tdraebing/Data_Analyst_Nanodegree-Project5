#!/usr/bin/python

import sys
import pickle
sys.path.append("./tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
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

### Load the dictionary containing the dataset
with open("./data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#create dataframe
df = pd.DataFrame(data_dict).transpose()
#delete email _adress variable
del df['email_address']
#Transform values to float
df = df.replace('%','',regex=True).astype('float')

### Task 2: Remove outliers
#delete TOTAL-entry
del data_dict['TOTAL']
#defining columns of which to remove outliers
#only columns defined in the features_list are mentioned here
col_ol = ['shared_receipt_with_poi',
          'to_messages']
for col in col_ol:
    index = df[np.abs(df[col] - df[col].mean()) \
          > 3 * df[col].std()][col].index.tolist()
    for i in index:
        df[i,col] = np.NaN
        
### Task 3: Create new feature(s)
# create binary feature of restricted_stock_deferred
df['restricted_stock_deferred_bin'] = np.zeros((len(df.index), 1)) + 1
df.loc[df['restricted_stock_deferred'].isnull(), 
       'restricted_stock_deferred_bin'] = 0
       
###Data Transformation and Scaling
#MinMaxScaler
#own implementaion to be able to keep NaNs
def minmax(features):
    features_scaled = features.copy()
    for col in features.columns:
        #calculating constants for each feature
        col_min = np.nanmin(features[col], axis = None)[0]
        col_max = np.nanmax(features[col], axis = None)[0]
        if (col_max - col_min) == 0:
            f_scaled = features[col]
        else:
            features_std = [((X - col_min) / (col_max - col_min)) for X in features[col]]
            
            train_std_min = np.nanmin(features_std, axis = None)
            train_std_max = np.nanmax(features_std, axis = None)
            
            #applying scaling to each entry
            f_scaled = [(X * (train_std_max - train_std_min) + train_std_min) for X in features_std] 
        features_scaled[col] = f_scaled
    return features_scaled
    
#transformation
to_add_one = ['deferred_income',
              'shared_receipt_with_poi']
df[to_add_one] = df[to_add_one] + 1
to_transform = ['bonus',
                'deferred_income',
                'exercised_stock_options', 
                'long_term_incentive', 
                'other',
                'restricted_stock',
                'shared_receipt_with_poi',
                'to_messages']
df[to_transform] = df[to_transform].apply(np.absolute).apply(np.log10)

#scaling
df = minmax(df)

#Imputating NaNs
df = df.fillna(0)
   
### Store to my_dataset for easy export below.
my_dataset = df.transpose().to_dict()

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)