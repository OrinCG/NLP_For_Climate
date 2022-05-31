import sys

import fasttext
import nltk
import pandas

# from gensim.models.fasttext import FastText

# -*- coding: utf-8 -*-
""" Model class:

The Model class here is designed to set up the model (training, theorectically
cleaning of data, etc) and return it as needed. At the moment it is using fasttext
to train a supervised and unsupervised model on 
"""


class Model:

    def generateUnSupModel(self):
        climate_opin_csv = pandas.read_csv("archive/BBCNEWS.201701.csv")
        snippets = climate_opin_csv[["Snippet","Categories"]]
        print(snippets)
        half = int(len(snippets)/2)
        snippets_t = snippets.iloc[:half,]
        self.snippets_v = snippets.iloc[half:len(snippets) - 1,]
        snippets_t.to_csv("training.txt", sep="\n", index=False)
        #Add tokenising and stuff here
        return fasttext.FastText.train_unsupervised("training.txt")

    def __init__(self):
        self.snippets_v = None
        self.model = self.generateUnSupModel()

    def getModel(self):
        return self.model

    def main(self) -> None:
        print(self.generateUnSupModel())


if __name__ == '__main__':
    m1 = Model()
    m1.main()
    print(m1.model.words)
    print(m1.model.get_nearest_neighbors("climate"))
    print(m1.model.labels)
#    print(similarities.)
