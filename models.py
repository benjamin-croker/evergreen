import numpy as np
import pandas as pd

from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
from sklearn.svm import SVC


class TFIDF(object):
    _X_cols = ["boilerplate"]
    _y_col = "label"

    def __init__(self, trainDF, testDF):

        print("Transforming data for TFIDF...")

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



class NumerLog(object):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
        "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
        "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
        "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]

    _y_col = "label"

    def __init__(self, trainDF, testDF):

        print("Transforming data for Numeric Logistic Regression Model...")

        self._lr_model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                                   C=1, fit_intercept=True, intercept_scaling=1.0, 
                                   class_weight=None, random_state=None)
        
        self._y = trainDF[self._y_col]
        X_train = np.array(trainDF[self._X_cols])
        X_test = np.array(testDF[self._X_cols])

        # scale and take best 5 PCA components
        X_all = np.vstack([X_train, X_test])
        X_all = preprocessing.scale(X_all)

        pca = PCA(n_components=5)
        X_all = pca.fit_transform(X_all)

        self._X_train = X_all[:X_train.shape[0],:]
        self._X_test = X_all[X_train.shape[0]:,:]


    def eval(self, cv_folds = 10):
        print("Evaluating {} fold CV score".format(cv_folds))
        return cross_validation.cross_val_score(self._lr_model, self._X_train, self._y,
                cv = cv_folds, scoring="roc_auc")


class NumerSVC(object):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
        "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
        "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
        "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]

    _y_col = "label"

    def __init__(self, trainDF, testDF):

        print("Transforming data for Numeric SVC Model...")

        self._svc_model = SVC(probability=True, C=1, gamma=0.1, cache_size=1000)
        
        self._y = trainDF[self._y_col]
        X_train = np.array(trainDF[self._X_cols])
        X_test = np.array(testDF[self._X_cols])

        # scale and take best 5 PCA components
        X_all = np.vstack([X_train, X_test])
        X_all = preprocessing.scale(X_all)

        pca = PCA(n_components=5)
        X_all = pca.fit_transform(X_all)

        self._X_train = X_all[:X_train.shape[0],:]
        self._X_test = X_all[X_train.shape[0]:,:]


    def eval(self, cv_folds = 10):
        print("Evaluating {} fold CV score".format(cv_folds))
        return cross_validation.cross_val_score(self._svc_model, self._X_train, self._y,
                cv = cv_folds, scoring="roc_auc")


