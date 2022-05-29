import fasttext
# from gensim.models.fasttext import FastText


class Model:
    def generateModel(self):
        fasttext.train_unsupervised()

    def __init__(self):
        self.model = self.generateModel()

    def getModel(self):
        return self.model

