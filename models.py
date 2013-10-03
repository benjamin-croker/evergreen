import numpy as np

from sklearn import metrics, preprocessing, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
from sklearn.svm import SVC

SEED = 42


class ClassifierModel(object):
    _X_cols = []
    _y_col = "label"

    _X_train = None
    _X_test = None
    _y = None

    _model = None

    def __init__(self):
        self._train_index_hash = None
        self._AUCs = None
        print("{}: Transforming data".format(str(self)))

    def __str__(self):
        return "Classifier"

    def fit(self, train_indices=None):
        """ fits the model, using a subset of the training data if given
        """
        X_train_subset = self._X_train
        y_subset = self._y

        if train_indices is not None:
            X_train_subset = X_train_subset[train_indices]
            y_subset = y_subset[train_indices]

        # check the hash of the training indices, and only retrain if needed
        if hash(tuple(train_indices)) == self._train_index_hash:
            print("{}: model already fitted".format(str(self)))

        else:
            if train_indices is not None:
                self._train_index_hash = hash(tuple(train_indices))
            else:
                self._train_index_hash = hash(train_indices)
            print("Fitting {} model".format(str(self)))
            self._model.fit(X_train_subset, y_subset)

    def predict(self, pred_data="test", pred_indices=None):
        print("{}: Predicting for {} data".format(str(self), pred_data))
        if pred_data == "test":
            X_pred_subset = self._X_test
        elif pred_data == "train":
            X_pred_subset = self._X_train
        else:
            return None

        if pred_indices is not None:
            X_pred_subset = X_pred_subset[pred_indices]

        return self._model.predict_proba(X_pred_subset)[:, 1]

    def train_size(self):
        return self._X_train.shape[0]

    def eval(self, n_folds=10):
        print("{}: Evaluating {} fold CV score".format(str(self), n_folds))
        AUCs = cross_validation.cross_val_score(self._model, self._X_train, self._y,
                                                cv=n_folds, scoring="roc_auc")
        self._AUCs = np.array(AUCs)
        return self._AUCs

    def init_one_fold(self, n_folds):
        print("{}: Set up model for one fold evals".format(str(self)))
        self._AUCs = np.zeros(n_folds)

    def one_fold_eval(self, train_indices, fold_eval_indices, fold_n):
        print("{}: Evaluating fold {} of {}".format(str(self), fold_n+1, self._AUCs.size))

        self.fit(train_indices=train_indices)
        preds = self.predict("train", fold_eval_indices)

        y_fold_test = self._y[fold_eval_indices]

        fpr, tpr, thresholds = metrics.roc_curve(y_fold_test, preds)
        self._AUCs[fold_n] = metrics.auc(fpr, tpr)

        return preds

    def last_eval(self):
        return self._AUCs


class TFIDF(ClassifierModel):
    _X_cols = ["boilerplate"]
    _y_col = "label"

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

        tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                              analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                              sublinear_tf=1)

        self._model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                            C=1, fit_intercept=True, intercept_scaling=1.0,
                                            class_weight=None, random_state=None)

        self._y = trainDF[self._y_col]
        X_train = list(np.array(trainDF[self._X_cols])[:, 0])
        X_test = list(np.array(testDF[self._X_cols])[:, 0])

        X_all = X_train + X_test
        X_all = tfv.fit_transform(X_all)

        self._X_train = X_all[:len(X_train)]
        self._X_test = X_all[len(X_train):]

    def __str__(self):
        return "TFIDF"


class NumerLog(ClassifierModel):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
               "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
               "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
               "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]
    _y_col = "label"

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

        self._model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
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

    def __str__(self):
        return "Numerical Logistic Regression"


class NumerSVC(ClassifierModel):
    _X_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
               "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
               "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
               "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]
    _y_col = "label"

    def __init__(self, trainDF, testDF):
        ClassifierModel.__init__(self)

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


class Stacker(object):
    _y_col = "label"
    _y = None

    _model = None
    _models = None

    def __init__(self, trainDF, testDF, model_classes=(TFIDF, NumerLog)):
        """ models is a list of models to stack using logistic regression
        """
        self._AUCs = None

        self._models = [Model(trainDF, testDF) for Model in model_classes]
        self._model = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                            C=1, fit_intercept=True, intercept_scaling=1.0,
                                            class_weight=None, random_state=None)
        self._y = trainDF[self._y_col]

    def __str__(self):
        return "Stacker"

    def fit(self, train_indices=None):
        """ fits the model, using a subset of the training data if given
        """
        print("{}: Fitting model".format(str(self)))

        y_subset = self._y
        if train_indices is not None:
            y_subset = y_subset[train_indices]

        # fit all the models
        for model in self._models:
            model.fit(train_indices)

        # get all the prediction scores
        X_train_subset = np.hstack([
            model.predict("train", train_indices)[np.newaxis].T for model in self._models
        ])

        # train the stacking model
        self._model.fit(X_train_subset, y_subset)

    def predict(self, pred_data="test", pred_indices=None):
        print("Predicting {} data for {} model".format(pred_data, self.__str__()))
        
        X_pred_subset = np.hstack([
            model.predict(pred_data, pred_indices)[np.newaxis].T for model in self._models
        ])
        
        return self._model.predict_proba(X_pred_subset)[:, 1]

    def eval(self, n_folds=10):
        print("{}: Evaluating {} fold CV score".format(str(self), n_folds))
        self.init_one_fold(n_folds)

        # get an n fold CV split
        kf = cross_validation.KFold(n=self._models[0].train_size(), n_folds=n_folds, random_state=SEED)

        fold_n = 0
        for train_indices, fold_eval_indices in kf:
            self.one_fold_eval(train_indices, fold_eval_indices, fold_n)
            # evaluate individual models to set the AUC scores
            for model in self._models:
                model.one_fold_eval(train_indices, fold_eval_indices, fold_n)
            fold_n += 1

        return self._AUCs

    def init_one_fold(self, n_folds):
        print("{}: Set up model for one fold evals".format(str(self)))
        self._AUCs = np.zeros(n_folds)

        # initialise the k-fold evaluation for all models
        for model in self._models:
            model.init_one_fold(n_folds)

    def one_fold_eval(self, train_indices, fold_eval_indices, fold_n):
        print("{}: Evaluating fold {} of {}".format(str(self), fold_n+1, self._AUCs.size))

        self.fit(train_indices=train_indices)
        preds = self.predict("train", fold_eval_indices)

        y_fold_test = self._y[fold_eval_indices]

        fpr, tpr, thresholds = metrics.roc_curve(y_fold_test, preds)
        self._AUCs[fold_n] = metrics.auc(fpr, tpr)

        return preds

    def last_eval(self):
        print("Overall AUC: mean={}, std={}".format(self._AUCs.mean(), self._AUCs.std()))
        for model in self._models:
            print("{} AUC: mean={}, std={}".format(str(model), model.last_eval().mean(),
                                                  model.last_eval().std()))
        return self._AUCs


