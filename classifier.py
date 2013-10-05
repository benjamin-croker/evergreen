import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cPickle

from sklearn import metrics, preprocessing, cross_validation
from sklearn.decomposition import PCA

from models import TFIDFLog, TFIDFRandForest, NumerLog, NumerSVC, CaterLog, Stacker

# column categories and transforms
id_cols = ["url", "urlid"]
label_cols = ["label"]
text_cols = ["boilerplate"]

category_cols = ["alchemy_category", "is_news", "news_front_page", "lengthyLinkDomain"]
numeric_cols = ["avglinksize", "commonlinkratio_1", "commonlinkratio_2",
                "commonlinkratio_3", "commonlinkratio_4", "compression_ratio", "frameTagRatio", "html_ratio",
                "image_ratio", "linkwordscore", "non_markup_alphanum_characters", "numberOfLinks",
                "numwords_in_url", "parametrizedLinkRatio", "spelling_errors_ratio"]
# these have too many blank values
excluded_cols = ["framebased", "hasDomainLink", "embed_ratio", "alchemy_category_score"]

replacements = {
    "is_news": ("?", 0),
    # not sure if the below is a good idea. It should be ok, since it's a category
    "news_front_page": ("?", -1)
}


def load_data(fileName):
    print("loading {}".format(fileName))
    df = pd.read_table(fileName)
    for k in replacements:
        df[k] = df[k].replace(*replacements[k])

    return df


def investigate(trainDF, print_examples=False, print_counts=False):
    # order by COV of value counts
    cols = [(c, trainDF[c], trainDF[c].value_counts(),
             trainDF[c].value_counts().std() / trainDF[c].value_counts().mean())
            for c in trainDF.columns]
    cols = sorted(cols, key=lambda x: x[3])
    for col in cols:
        print("-----\nVariable: {}, COV of value counts: {}".format(
            col[0], col[3]))

        if np.isreal(col[1][0]):
            print("Mean: {}, STD: {}".format(col[1].mean(), col[1].std()))

        if print_examples:
            print("Examples:")
            print(col[1][:5])

        print("Value Counts (catagories: {})".format(col[2].size))
        if print_counts: print(col[2][:5])

    print("PCA analysis of numeric variables")
    # scale the data first
    X = np.array(trainDF[numeric_cols])
    X = preprocessing.scale(X)

    pca = PCA()
    pca.fit(X)
    print("Explained variance")
    print(pca.explained_variance_ratio_)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()


def cache_model(model, file_name):
    with open(os.path.join("cache", file_name), "wb") as f:
        print("Caching model {}".format(file_name))
        cPickle.dump(model, f)


def load_model(file_name):
    with open(os.path.join("cache", file_name)) as f:
        print("Loading model {}".format(file_name))
        return cPickle.load(f)

def gen_model(ModelClass):
    trainDF = load_data(os.path.join("data", "train.tsv"))
    testDF = load_data(os.path.join("data", "test.tsv"))
    return ModelClass(trainDF, testDF)

def model_evaluation(model):
    model.eval()
    model.last_eval()

def model_submission(model):
    submissionDF = model.submission()
    submissionDF.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    # model_evaluation(Stacker)

    # stacker = Stacker(load_data(os.path.join("data", "train.tsv")),
    #                   load_data(os.path.join("data", "test.tsv")))
    # cache_model(stacker, "stacker")

    stacker = load_model("stacker")
    model_submission(stacker)













