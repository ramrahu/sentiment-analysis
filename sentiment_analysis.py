# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:10:29 2020

@author: Aspire V5
"""

# Sentiment Analysis

# Importing the libraries
import numpy as np
import pandas as pd
import re

# Importing the dataset
train_dataset = pd.read_csv('train_F3WbcTw.csv')
train_X = train_dataset.iloc[:, 1:3].values
dfX = pd.DataFrame(train_X);
train_Y = train_dataset.iloc[:, 3].values
dfY = pd.DataFrame(train_Y);

test_dataset = pd.read_csv('test_tOlRoBf.csv')
test_X = test_dataset.iloc[:, 1].values
dfY_test = pd.DataFrame(test_X)

def preprocess_review(words):
    words = words.lower()
    words = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', words)
    words = words.strip(' "\'')
    words = re.sub(r'\s+', ' ', words)
    words = words.split()
    
    processed_review = []
    for word in words:
        word = str(porter_stemmer.stem(word))
        processed_review.append(word)
    
    return ' '.join(processed_review)

# Cleaning the data
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split

X_train, Y_train, X_val, Y_val = train_test_split(dfX[0], dfY[0], test_size=0.20, random_state = np.random.randint(1,1000, 1)[0])

porter_stemmer = PorterStemmer()
review = []
review_test = []

for i in X_train:
    #text = dfX[0][i]
    text = i
    review.append(preprocess_review(text))

for i in Y_train:
    #text = dfY_test[0][i]
    text = i
    review_test.append(preprocess_review(text))

# Extract features and perform classification using SVM
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, 
                            sublinear_tf=True, use_idf=True)
train_features_tfidf = vectorizer.fit_transform(review)

classifier_liblinear = svm.LinearSVC()
classifier_liblinear.fit(train_features_tfidf, X_val)

test_features_tfidf = vectorizer.transform(review_test)

prediction_liblinear = classifier_liblinear.predict(test_features_tfidf)

score_svm = f1_score(prediction_liblinear, Y_val, average = None)

# Perform classification using Logistic Regression
from xgboost import XGBClassifier

xgb = XGBClassifier(max_depth=6,n_estimators=1000,random_state=0)
xgb.fit(train_features_tfidf,X_val)

prediction_xgb = xgb.predict(test_features_tfidf)

score_xgb = f1_score(prediction_xgb, Y_val, average = None)