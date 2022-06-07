from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
import ast

app = Flask(__name__)
api = Api(app)
#https://towardsdatascience.com/the-right-way-to-build-an-api-with-python-cd08ab285f8f