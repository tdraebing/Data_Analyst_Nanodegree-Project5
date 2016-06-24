import pandas as pd
import numpy as np


# data preprocessing
def prepare_data(data_dict):
    # delete TOTAL-entry
    print(data_dict)
    del data_dict['TOTAL']
    # create dataframe
    df = pd.DataFrame(data_dict).transpose()
    # delete email_address variable
    del df['email_address']
    # Transform values to float
    df = df.replace('%', '', regex=True).astype('float')
    # create binary variable for restricted_stock_deferred
    df['loan_advances_bin'] = np.zeros((len(df.index), 1)) + 1
    df.loc[df['loan_advances'].isnull(), 'loan_advances_bin'] = 0
    df['director_fees_bin'] = np.zeros((len(df.index), 1)) + 1
    df.loc[df['director_fees'].isnull(), 'director_fees_bin'] = 0
    df['restricted_stock_deferred_bin'] = np.zeros((len(df.index), 1)) + 1
    df.loc[df['restricted_stock_deferred'].isnull(),
           'restricted_stock_deferred_bin'] = 0
    # delete variables with large amounts of NaN
    df = df.drop(['loan_advances',
                  'director_fees',
                  'restricted_stock_deferred'], 1)
    # reduce variables
    features_list = ['bonus',
                     'deferral_payments',
                     'deferred_income',
                     'director_fees_bin',
                     'exercised_stock_options',
                     'expenses',
                     'from_messages',
                     'from_poi_to_this_person',
                     'from_this_person_to_poi',
                     'loan_advances_bin',
                     'long_term_incentive',
                     'other',
                     'poi',
                     'restricted_stock',
                     'restricted_stock_deferred_bin',
                     'salary',
                     'shared_receipt_with_poi',
                     'to_messages',
                     'total_payments',
                     'total_stock_value']
    df_reduced = df.copy()[features_list]
    return df_reduced
