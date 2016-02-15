# -*- coding: utf-8 -*-
import os
import pickle
import re
import sys
from nltk.stem.snowball import SnowballStemmer
import string


def parse_signature(f):
    f.seek(0)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    from_email = ""
    subject = ""
    if len(content) > 1:
        try:
            from_meta = re.search('\nFrom:.+\n', content[0])
            subject_meta = re.search('\nSubject:.+\n', content[0])
            from_email = re.search('[A-Za-z0-9._-]+[@].+[.].+$', from_meta.group(0)).group(0)
            subject = re.search('.+$', subject_meta.group(0)).group(0)
        except:
            return None, None
            
    ### remove punctuation
    text_string = subject.translate(string.maketrans("", ""), string.punctuation)

    words = text_string.split()

    ### split the text string into individual words, stem each word,
    ### and append the stemmed word to words (make sure there's a single
    ### space between each stemmed word)

    stemmer = SnowballStemmer("english")
    words = [stemmer.stem(word) for word in words]
    words = ' '.join(words[1:])
    return from_email, words
        

def create_corpus():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    del data_dict['TOTAL']
    
    from_word_data = {'poi':{}, 'nonpoi':{}}
    
    poi_emails = [data_dict[d]['email_address'] \
                    if data_dict[d]['poi'] == 1 \
                    else 'NaN' \
                    for d in data_dict]
    poi_emails = [x for x in poi_emails if x != 'NaN']
    
    nonpoi_emails = [data_dict[d]['email_address'] \
                    if data_dict[d]['poi'] == 0 \
                    else 'NaN' \
                    for d in data_dict]
    nonpoi_emails = [x for x in nonpoi_emails if x != 'NaN']
    
    path="D:/Documents/GoogleDrive/Udacity - data Analyst/ud120-projects/maildir"
    
    for path, dirs, files in os.walk(path):
        print(path)
        for filename in files:
            fullpath = os.path.join(path, filename)
            with open(fullpath, 'r') as email:
                from_email, text = parse_signature(email)
                if from_email in poi_emails:
                    try:
                        from_word_data['poi'][from_email].append(text)
                    except:
                        from_word_data['poi'][from_email] = []
                        from_word_data['poi'][from_email].append(text)
                elif from_email in nonpoi_emails:
                    try:
                        from_word_data['nonpoi'][from_email].append(text)
                    except:
                        from_word_data['nonpoi'][from_email] = []
                        from_word_data['nonpoi'][from_email].append(text)
        
    
    pickle.dump( from_word_data, open("from_mail_word_data.pkl", "w") )
    
#    for i in from_word_data:
#        from_word_data[i] = ' '.join(from_word_data[i])
#        
#    for i in to_word_data:
#        to_word_data[i] = ' '.join(to_word_data[i])
#    
#    pickle.dump( from_word_data, open("j-from_mail_word_data.pkl", "w") )
#    pickle.dump( to_word_data, open("j-to_mail_word_data.pkl", "w") )
#    print(from_word_data['ben.glisan@enron.com'])
    
def split_poi(email_dict, out_prefix):
    print('Pickle load')
    file_handler = open(email_dict, "r")
    text = pickle.load(file_handler)
    file_handler.close()
    
    print('reform data')
    features_poi = [' '.join(text['poi'][sublist]) for sublist in text['poi']]
    features_nonpoi = [' '.join(text['nonpoi'][sublist]) for sublist in text['nonpoi']]

    features = features_poi + features_nonpoi
    labels = [1]*len(features_poi) + [0]*len(features_nonpoi)
    
    print(len(features))
    print(len(labels))
    pickle.dump( features, open(out_prefix + "_text_features.pkl", "w") )
    pickle.dump( labels, open(out_prefix + "_text_labels.pkl", "w") )
    
def create_features(email_dict, vectorizer, selector):
    print('Pickle load')
    file_handler = open(email_dict, "r")
    text = pickle.load(file_handler)
    file_handler.close()
    
    text_full = text['poi'].copy()
    text_full.update(text['nonpoi'])
    
    text_full_vect = vectorizer.transform(text_full)
    text_full_select = selector.transform(text_full_vect)
    return text_full_select
    
def train_tfidf(feat, lab):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    from sklearn.metrics import f1_score
    from sklearn import cross_validation
    
    from sklearn.tree import DecisionTreeClassifier
    
    print('Pickle load')
    file_handler = open(feat, "r")
    features = pickle.load(file_handler)
    file_handler.close()
    file_handler = open(lab, "r")
    labels = pickle.load(file_handler)
    file_handler.close()
    
    print('split data')
    features_train, features_test, labels_train, labels_test = \
                    cross_validation.train_test_split(features,
                                                      labels,
                                                      test_size=0.4)
    
    print('train tfidf')                                                  
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 max_df=0.5,
                                 stop_words='english')

    features_train = vectorizer.fit_transform(features_train)
    features_test = vectorizer.transform(features_test)
    
    print('selector')
    selector = SelectKBest(k=20)
    selector.fit(features_train, labels_train)
    features_transformed = selector.transform(features_train)
    features_test_transformed  = selector.transform(features_test)
    
    print('decision tree')
    clf = DecisionTreeClassifier()
    clf = clf.fit(features_transformed, labels_train)
    pred = clf.predict(features_test_transformed)
    acc = clf.score(features_test_transformed, labels_test)
    
    print('Accuracy', acc)
    print('F1: ', f1_score(labels_test, pred))
    
    return vectorizer, selector

#create_corpus()
split_poi('from_mail_word_data.pkl', 'from')
vectorizer, selector = train_tfidf("from_text_features.pkl", "from_text_labels.pkl")
create_features('from_mail_word_data.pkl', vectorizer, selector)