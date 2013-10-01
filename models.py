import numpy as np

from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
from sklearn.svm import SVC


class TFIDF(object):
    _X_cols = ["boilerplate"]
    _y_col = "label"
    _AUCs = None

    def __init__(self, trainDF, testDF):
        print("Transforming data for TFIDF...")

        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)

        self._lr_model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                               C=1, fit_intercept=True, intercept_scaling=1.0,
                                               class_weight=None, random_state=None)

        self._y = trainDF[self._y_col]
        X_train = list(np.array(trainDF[self._X_cols])[:, 0])
        X_test = list(np.array(testDF[self._X_cols])[:, 0])

        X_all = X_train + X_test
        X_all = tfv.fit_transform(X_all)

        self._X_train = X_all[:len(X_train)]
        self._X_test = X_all[len(X_train):]

    def train_size(self):
        return self._X_train.shape[0]

    def eval(self, n_folds=10):
        print("Evaluating {} fold CV score".format(n_folds))
        AUCs = cross_validation.cross_val_score(self._lr_model, self._X_train, self._y,
                                                cv=n_folds, scoring="roc_auc")
        self._AUCs = np.array(AUCs)
        return self._AUCs

    def last_eval(self):
        return self._AUCs

    def init_one_fold(self, n_folds):
        print("Set up model for one fold evals")
        self._AUCs = np.zeros(n_folds)

    def one_fold_eval(self, train_indices, test_indices, fold_n):
        print("Evaluating fold {} of {}".format(fold_n, self._AUCs.size))
        X_fold_train = self._X_train[train_indices]
        y_fold_train = self._y[train_indices]

        X_fold_test = self._X_train[test_indices]
        y_fold_test = self._y[test_indices]

        self._lr_model.fit(X_fold_train, y_fold_train)

        preds = self._lr_model.predict_proba(X_fold_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_fold_test, preds)
        AUC = metrics.auc(fpr, tpr)

        self._AUCs[fold_n] = AUC
        return AUC


class NumerLog(object):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
               "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
               "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
               "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]
    _y_col = "label"
    _AUCs = None

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

        self._X_train = X_all[:X_train.shape[0], :]
        self._X_test = X_all[X_train.shape[0]:, :]

    def eval(self, n_folds=10):
        print("Evaluating {} fold CV score".format(n_folds))
        AUCs = cross_validation.cross_val_score(self._lr_model, self._X_train, self._y,
                                                cv=n_folds, scoring="roc_auc")
        self._AUCs = np.array(AUCs)
        return self._AUCs

    def last_eval(self):
        return self._AUCs


class NumerSVC(object):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
               "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
               "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
               "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]
    _y_col = "label"
    _AUCs = None

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

        self._X_train = X_all[:X_train.shape[0], :]
        self._X_test = X_all[X_train.shape[0]:, :]

    def eval(self, n_folds=10):
        print("Evaluating {} fold CV score".format(n_folds))
        AUCs = cross_validation.cross_val_score(self._svc_model, self._X_train, self._y,
                                                cv=n_folds, scoring="roc_auc")
        self._AUCs = np.array(AUCs)
        return self._AUCs

    def last_eval(self):
        return self._AUCs


