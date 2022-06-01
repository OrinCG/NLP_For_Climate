import sys
from itertools import starmap

import fasttext
import nltk
import pandas

# from gensim.models.fasttext import FastText

# -*- coding: utf-8 -*-
from gensim.parsing import remove_stopwords

""" Model class:

The Model class here is designed to set up the model (training, theorectically
cleaning of data, etc) and return it as needed. At the moment it is using fasttext
to train a supervised and unsupervised model on 
"""


class Model:

    def combine(self,x,y):
        splitX = x.split(",")
        combinedS = ""
        for label in splitX:
            combinedS += "__label__" + label  + " "
        return combinedS + y

    def generateUnSupModel(self):
        climate_opin_csv = pandas.read_csv("archive/BBCNEWS.201701.csv")
        snippets = climate_opin_csv["Snippet"].apply(lambda x: remove_stopwords(x))
        labels = climate_opin_csv["Categories"].dropna()
        print(snippets)
        half = int(len(snippets)/2)
        snippets_t = snippets.iloc[:half,]
        label_snip = pandas.DataFrame(list(starmap(self.combine, zip(labels, snippets_t))))

        self.snippets_v = snippets.iloc[half:len(snippets) - 1,]
        label_snip.to_csv("training.txt", sep="\n", index=False)
        #Add tokenising and stuff here
        #fasttext.supervised("training.txt","model.txt")
        return fasttext.FastText.train_supervised("training.txt")

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
    print(m1.model.predict("Environmentalists warn new power plant could have knock on effects on global warming.",5))
    print(m1.model.labels)
    print("-----------------------------------------------------------------------------------------")
    for prediction in m1.snippets_v:
        print(prediction)
        print(m1.model.predict(prediction,3))
#    print(similarities.)
