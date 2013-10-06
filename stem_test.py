import classifier as cl
import numpy as np

from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer

df = cl.load_data("data/train.tsv")

stemmer = EnglishStemmer()
# tokenizer will remove all punctuation
tokenizer = RegexpTokenizer(r'\w+')

def stem_words(text):
    return " ".join([
        stemmer.stem(word) for word in tokenizer.tokenize(text)
        ])

X = list(np.array(df[["boilerplate"]])[:,0])
print X[-1]

X = [stem_words(text) for text in X]
print X[-1]