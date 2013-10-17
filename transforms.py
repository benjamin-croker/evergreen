import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer


def TFIDF_transform(X_train, X_test, y_train, stemmer=None, n_important=None):
    """ takes two arrays containing the training and test data and returns
        the TFIDF vectorised data.

        If stemmer equals "snowball" or "lancaster", then the text in
        X_train and X_test is stemmed first

        If n_important is set, only the most important n_words are included
        in the transformed data
    """


    tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',
                          analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=1, smooth_idf=1,
                          sublinear_tf=1)

    X_train = list(np.array(X_train)[:, 0])
    X_test = list(np.array(X_test)[:, 0])

    X_all = X_train + X_test

    # perform stemming if a stemmer is passed
    if stemmer is not None:
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

    # work out the most important words
    if n_important is not None:
        log_cl = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                    C=1, fit_intercept=True, intercept_scaling=1.0,
                                    class_weight=None, random_state=None)
        print("Extracting important words")
        log_cl.fit(X_all[:len(X_train)], y_train)

        # get the most important words
        coef = log_cl.coef_.ravel()
        important_words_ind = np.argsort(np.abs(coef))[-n_important:]

        print("important words are:")
        print(np.array(tfv.get_feature_names())[important_words_ind])
        X_all = X_all[:, important_words_ind].todense()

    return X_all[:len(X_train)], X_all[len(X_train):]


def onehot_transform(X_train, X_test, y_train):
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
