import numpy as np
import logging
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def TFIDF_transform(X_train, X_test, stemmer=None, stop_words=None):
    """ takes two arrays containing the training and test data and returns
        the TFIDF vectorised data.

        If stemmer equals "snowball" or "lancaster", then the text in
        X_train and X_test is stemmed first

        If n_important is set, only the most important n_words are included
        in the transformed data
    """


    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                          sublinear_tf=1, stop_words=stop_words)

    X_train = list(np.array(X_train)[:, 0])
    X_test = list(np.array(X_test)[:, 0])

    X_all = X_train + X_test

    # perform stemming if a stemmer is passed
    if stemmer is not None:
        logging.debug("Stemming tokens")
        if stemmer == "lancaster":
            stemmer = LancasterStemmer()
        elif stemmer == "snowball":
            stemmer = EnglishStemmer()

        # tokenizer will remove all punctuation
        tokenizer = RegexpTokenizer(r'\w+')

        stem_text = lambda text: " ".join([
            stemmer.stem(word) for word in tokenizer.tokenize(text)
        ])
        X_all = [stem_text(text) for text in X_all]

    X_all = tfv.fit_transform(X_all)

    return X_all[:len(X_train)], X_all[len(X_train):]


def select_important_TFIDF(X_train, X_test, y, n_tokens):
    """ takes a sparse TFIDF matrix, and selects the most important n_tokens
        returns X as a dense matrix with n columns
    """

    # work out the most important words by training a logistic regression model
    log_cl = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                C=1, fit_intercept=True, intercept_scaling=1.0,
                                class_weight=None, random_state=None)
    logging.debug("Extracting important words")
    log_cl.fit(X_train, y)

    # get the most important words
    coef = log_cl.coef_.ravel()
    important_words_ind = np.argsort(np.abs(coef))[-n_tokens:]

    return X_train[:, important_words_ind].todense(), X_test[:, important_words_ind].todense()


def feature_combine_transform(X_train, X_test, n_combine=2):
    """ for use with categorical data
        will create hashes of tuples of multiple features to create new features
    """
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    _, n_features = np.array(X_train).shape
    extra_train = []
    extra_test = []

    for inds in combinations(range(n_features), n_combine):
        extra_train.append([hash(tuple(x)) for x in X_train[:,inds]])
        extra_test.append([hash(tuple(x)) for x in X_test[:,inds]])

    return np.hstack((X_train, np.array(extra_train).T)), np.hstack((X_test, np.array(extra_test).T))


def onehot_transform(X_train, X_test):
    """ Performs a label encoder transform on each feature of the data
    """
    oneHotEncoder = preprocessing.OneHotEncoder()
    labelEncoder = preprocessing.LabelEncoder()

    X_all = np.vstack((X_train, X_test))

    # label encoder can only operate on one column at a time
    for i, column in enumerate(X_all.T):
        X_all.T[i] = labelEncoder.fit_transform(column)
    X_all = oneHotEncoder.fit_transform(X_all)

    return X_all[:len(X_train)], X_all[len(X_train):]
