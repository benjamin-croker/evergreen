import numpy as np
import pandas as pd

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm


class TFIDF(object):
    _X_cols = ["boilerplate"]
    _y_col = "label"
    _replacements = {}

    def __init__(self, trainDF, testDF):

        print("Transforming data for TFIDF...")

        # perform transforms on columns
        for df in [trainDF, testDF]:
            for k in self._replacements:
                df[k] = df[k].replace(*replacements[k])

        tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
              analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

        self._lr_model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                                   C=1, fit_intercept=True, intercept_scaling=1.0, 
                                   class_weight=None, random_state=None)
        
        self._y = trainDF[self._y_col]
        X_train = list(np.array(trainDF[self._X_cols])[:,0])
        X_test = list(np.array(testDF[self._X_cols])[:,0])

        X_all = X_train + X_test
        X_all = tfv.fit_transform(X_all)

        self._X_train = X_all[:len(X_train)]
        self._X_test = X_all[len(X_train):]


    def eval(self, cv_folds = 10):
        print("Evaluating {} fold CV score".format(cv_folds))
        return cross_validation.cross_val_score(self._lr_model, self._X_train, self._y,
                cv = cv_folds, scoring="roc_auc")

