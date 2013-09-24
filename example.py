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

def main():

    print "loading data.."
    traindata = list(np.array(p.read_table('data/train.tsv'))[:,2])
    testdata = list(np.array(p.read_table('data/test.tsv'))[:,2])
    y = np.array(p.read_table('data/train.tsv'))[:,-1]

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
    X_all = tfv.transform(X_all)

    X = X_all[:lentrain]
    X_test = X_all[lentrain:]

    print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20,
            scoring="roc_auc"))

if __name__=="__main__":
    main()
