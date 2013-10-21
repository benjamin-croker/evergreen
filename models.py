import numpy as np
import pandas as pd

from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import transforms as tx

SEED = 42


class ClassifierModel(object):
    _X_cols = []
    _y_col = "label"
    _id_col = "urlid"
    _model = None

    X_train = None
    X_test = None
    y = None

    def __init__(self):
        self._AUCs = None
        print("{}: Transforming data".format(str(self)))

    def __str__(self):
        return "Classifier"

    def train_transform(self, X_train, X_test, y):
        return X_train, X_test

    def fit(self, X_train, y):
        """ fits the model. If fit_hash is provided, it will check fit_hash against the last
            given one, and only train the model if it's different
        """
        print("Fitting {} model".format(str(self)))
        self._model.fit(X_train, y)

    def predict(self, X_predict, data_note="test"):
        print("{}: Predicting for {} data".format(str(self), data_note))
        return self._model.predict_proba(X_predict)[:, 1]

    def train_size(self):
        return self.X_train.shape[0]

    def eval(self, n_folds=10):
        print("{}: Evaluating {} fold CV score".format(str(self), n_folds))
        self.init_one_fold(n_folds)

        # get an n fold CV split
        kf = cross_validation.KFold(n=self._model.train_size(), n_folds=n_folds,
                random_state=SEED, shuffle=True)

        fold_n = 0
        for train_indices, fold_eval_indices in kf:
            self.one_fold_eval(train_indices, fold_eval_indices, fold_n)
            fold_n += 1

        return self._AUCs

    def init_one_fold(self, n_folds):
        print("{}: Set up model for one fold evals".format(str(self)))
        self._AUCs = np.zeros(n_folds)

    def one_fold_eval(self, train_indices, fold_eval_indices, fold_n):
        print("{}: Evaluating fold {} of {}".format(str(self), fold_n + 1, self._AUCs.size))

        X_train, X_eval = self.train_transform(self.X_train[train_indices],
                                               self.X_train[fold_eval_indices],
                                               self.y[train_indices])
        self.fit(X_train, self.y[train_indices])
        preds = self.predict(X_eval, "train")

        fpr, tpr, thresholds = metrics.roc_curve(self.y[fold_eval_indices], preds)
        self._AUCs[fold_n] = metrics.auc(fpr, tpr)
        return preds

    def last_eval(self):
        print("{} overall AUC: mean={}, std={}".format(str(self), self._AUCs.mean(), self._AUCs.std()))
        return self._AUCs

    def get_data(self):
        """ returns X_train, X_test, y
        """
        return self.X_train, self.X_test, self.y

    def add_features(self, X_train, X_test):
        """ stacks X_train and X_test with the existing training and test data
            X_train and X_test must have the same number of columns as the
            existing training and test data
        """
        self.X_train = np.hstack((self.X_train, X_train))
        self.X_test = np.hstack((self.X_test, X_test))


class TFIDFLog(ClassifierModel):
    _X_cols = ["boilerplate"]

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

        self._model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                         C=3, fit_intercept=True, intercept_scaling=1.0,
                                         class_weight=None, random_state=None)

        self._ids_train = trainDF[self._id_col]
        self._ids_test = testDF[self._id_col]

        self.y = trainDF[self._y_col]
        self.X_train, self.X_test = tx.TFIDF_transform(trainDF[self._X_cols],
                                                       testDF[self._X_cols],
                                                       "snowball")

    def __str__(self):
        return "TFIDF Logistic Regression"


class TFIDFRandForest(ClassifierModel):
    _X_cols = ["boilerplate"]

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

        self._model = RandomForestClassifier(n_estimators=100, min_samples_split=16)

        self._ids_train = trainDF[self._id_col]
        self._ids_test = testDF[self._id_col]

        self.y = trainDF[self._y_col]
        self.X_train, self.X_test = tx.TFIDF_transform(trainDF[self._X_cols],
                                                       testDF[self._X_cols],
                                                       "snowball")

    def __str__(self):
        return "TFIDF Random Forest"

    def train_transform(self, X_train, X_test, y):
        return tx.select_important_TFIDF(X_train, X_test, y, 200)


class TFIDFNaiveBayes(ClassifierModel):
    _X_cols = ["boilerplate"]

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

        self._model = GaussianNB()

        self._ids_train = trainDF[self._id_col]
        self._ids_test = testDF[self._id_col]

        self.y = trainDF[self._y_col]
        self.X_train, self.X_test = tx.TFIDF_transform(trainDF[self._X_cols],
                                                       testDF[self._X_cols],
                                                       "snowball")

    def __str__(self):
        return "TFIDF Naive Bayes"

    def train_transform(self, X_train, X_test, y):
        return tx.select_important_TFIDF(X_train, X_test, y, 200)


class Mixer(object):
    def __init__(self, weights):
        self.set_weights(weights)

    def fit(self, X, y):
        """ just for compatability with API. Does nothing
        """
        pass

    def predict(self, X):
        return np.dot(X, self._weights)

    def set_weights(self, weights):
        # normalise weights
        self._weights = np.array(weights) / np.array(weights).sum()


class Stacker(object):
    _y_col = "label"
    _id_col = "urlid"
    y = None

    _model = None
    _models = None

    def __init__(self, trainDF, testDF,
                 model_classes=(TFIDFRandForest, TFIDFLog, TFIDFNaiveBayes),
                 weights=(0.125, 0.75, 0.125)):
        """ models is a list of models to stack using logistic regression
        """
        self._AUCs = None

        self._models = [Model(trainDF, testDF) for Model in model_classes]
        self._model = Mixer(weights)

        self.y = trainDF[self._y_col]
        self._ids_train = trainDF[self._id_col]
        self._ids_test = testDF[self._id_col]

    def __str__(self):
        return "Stacker"

    def fit_predict(self):
        """ fits the stacking model with all data
        """
        print("{}: Fitting model and predicting data".format(str(self)))

        submodel_preds = []
        for model in self._models:
            X_train, X_test = model.train_transform(model.X_train, model.X_test, model.y)
            model.fit(X_train, model.y)
            submodel_preds.append(model.predict(X_test, "test"))

        X = np.hstack([p[np.newaxis].T for p in submodel_preds])
        return self._model.predict(X)

    def eval(self, n_folds=10):
        print("{}: Evaluating {} fold CV score".format(str(self), n_folds))
        print("{}: Set up model for one fold evals".format(str(self)))
        self._AUCs = np.zeros(n_folds)

        # initialise the k-fold evaluation for all models
        for model in self._models:
            model.init_one_fold(n_folds)

        # get an n fold CV split
        kf = cross_validation.KFold(n=self._models[0].train_size(),
                                    n_folds=n_folds, random_state=SEED,
                                    shuffle=True)

        fold_n = 0
        for train_indices, fold_eval_indices in kf:
            # calculate the predictions for individual submodels
            submodel_preds = [model.one_fold_eval(train_indices, fold_eval_indices, fold_n)
                    for model in self._models]
            submodel_preds = np.hstack([p[np.newaxis].T for p in submodel_preds])
            # calculate the overall prediction
            preds = self._model.predict(submodel_preds)

            fpr, tpr, thresholds = metrics.roc_curve(self.y[fold_eval_indices], preds)
            self._AUCs[fold_n] = metrics.auc(fpr, tpr)

            fold_n += 1

        return self._AUCs

    def last_eval(self):
        print("{} overall AUC: mean={}, std={}".format(str(self), self._AUCs.mean(), self._AUCs.std()))
        for model in self._models:
            model.last_eval()
        return self._AUCs

    def set_weights(self, weights):
        self._model.set_weights(weights)

    def submission(self):
        print("{}: making a submission dataframe".format(str(self)))
        preds = self.fit_predict()

        submissionDF = pd.DataFrame(self._ids_test)
        submissionDF[self._y_col] = preds

        return submissionDF