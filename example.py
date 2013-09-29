"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p


loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')


print "loading data.."
# get only the boilerplate text (column 2)
traindata = list(np.array(p.read_table('data/train.tsv'))[:,2])
testdata = list(np.array(p.read_table('data/test.tsv'))[:,2])
# get the label (1 for evergreen)
y = np.array(p.read_table('data/train.tsv'))[:,-1]

# the r'\w{1,}' string gets rid of the json stuff. The format is '{"title": "<content>"}'
# tfv is a vectorizer object

# TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
# CountVectorizer gets the counts for each word in each document, and will return an
# N x vocab_size sparse matrix.

# The TfidfTransformer will turn the word counts into tfidf score.
# tfidf_score = function(word, document, corpus) {
#    return (count(word) in document)/(count(word in document) in corpus)
#}

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
      analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                           C=1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)

X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline"
tfv.fit(X_all)
print "transforming data"
# this will return an N x vocab_size sparse matrix
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20,
        scoring="roc_auc"))

