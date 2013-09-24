import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def load_data(fileName):
    print("loading {}".format(fileName))
    return pd.read_table(fileName)

def investigate(trainDF):
    # order by COV of value counts
    cols = [(c, trainDF[c], trainDF[c].value_counts(),
            trainDF[c].value_counts().std()/trainDF[c].value_counts().mean())
            for c in trainDF.columns]
    cols = sorted(cols, key=lambda x: x[3])
    for col in cols:
        print("-----\nVariable: {}, COV of value counts: {}".format(
            col[0], col[3]))
        
        print("Examples:")
        print(col[1][:5])

        print("Value Counts (catagories: {})".format(col[2].size))
        print(col[2][:5])

if __name__ == "__main__":
    trainDF = load_data(os.path.join("data", "train.tsv"))

    investigate(trainDF)