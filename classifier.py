import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def load_data(fileName):
    print("loading {}".format(fileName))
    return pd.read_table(fileName)

def investigate(trainDF):
    # print the type of data for each column, and the top 5 value counts
    for c in trainDF.columns:
        print("--Column: {}--".format(c))
        print("Type: {}".format(trainDF[c].dtype))

        valCounts = trainDF[c].value_counts()
        print("Value Counts: ({} catagories, COV={})".format(valCounts.size,
                valCounts.std()/valCounts.mean()))
        print(trainDF[c].value_counts()[:5])

if __name__ == "__main__":
    trainDF = load_data(os.path.join("data", "train.tsv"))

    investigate(trainDF)