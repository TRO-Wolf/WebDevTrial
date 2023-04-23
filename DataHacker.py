


import warnings
import requests
warnings.filterwarnings('ignore')


from pathlib import Path

import os, sys

parent = os.path.abspath('..')
sys.path.insert(1, parent)


import numpy as np
import pandas as pd


from numpy.random import choice, normal

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from collections import namedtuple, deque


import gc


from matplotlib import cm
from itertools import product

import time
from datetime import timedelta
import datetime

import talib





from sklearn.metrics import mean_squared_error, explained_variance_score, roc_curve, make_scorer, roc_auc_score, mean_absolute_error, recall_score, precision_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, _tree
from sklearn import tree
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier


import joblib

from joblib import load



import lightgbm as lgb


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor


from IPython.display import clear_output


class APIPull:

    def __init__(self, ticker):
        self.today = datetime.date.today()
        self.api = '3592ca58d809cc98b99c99ffd313e94e'
        self.ticker = ticker
        
        self.data = self.pull_basic()
        
    
    #pulls the ticker with OHLCV data
    def pull_basic(self, ticker=None):
        if ticker is None:
            ticker = self.ticker

        self.url = 'https://financialmodelingprep.com/api/v3/historical-price-full/{}?from=1999-03-12&to={}&apikey=3592ca58d809cc98b99c99ffd313e94e'.format(ticker, self.today)
        response = requests.get(self.url)
        json_data = response.json()

        data = pd.DataFrame(json_data['historical'])

        data = data.rename({'date':'Date'},axis=1)
        data['Date'] = data['Date'].astype('datetime64[ns]')
        data = data.sort_values(by='Date',ascending=True)
        data = data.set_index('Date')
        data = data[['open', 'high', 'low', 'close', 'volume']]
        data = data.astype('float32')
        
        return data
    

class Organizer:
    def __init__(self, ticker):
        self.ticker = ticker.upper()

        self.golden = APIPull(self.ticker)

        self.data = self.golden.data.copy()



        if self.ticker == 'AAPL':
            self.model = load('aapl.joblib')
        elif self.ticker == 'MSFT':
            self.model = load('msft.joblib')
        elif self.ticker == 'AMZN':
            self.model = load('amzn.joblib')
        elif self.ticker == 'GOOG':
            self.model = load('goog.joblib')
        elif self.ticker == 'GOOGL':
            self.model = load('googl.joblib')
        elif self.ticker =='TSLA':
            self.model = load('tsla.joblib')
        elif self.ticker == 'NVDA':
            self.model = load('nvda.joblib')
        elif self.ticker =='META':
            self.model = load('meta.joblib')
        elif self.ticker == 'PEP':
            self.model = load('pep.joblib')
        elif self.ticker == 'AVGO':
            self.model = load('avgo.joblib')


    
    def split(self, input_data):
        data = input_data.copy()
        oos = data['2023':]
        data = data[:'2023']    

        return data, oos

    def create_xy(self, input_data):
        data = input_data.copy()
        y = data.filter(like='target')
        X = data.drop(y.columns, axis=1)

        return X,y

        
    
    def pull(self, ticker):
        data = self.golden.pull_basic(ticker=ticker)

        return data


        
    def pre_process(self, ticker):
        data = self.pull(ticker=ticker)
        list1 = [1,3, 5, 8, 21]
        list2 = [1]

        for n in list1:
            data[f'RET {n}'] = data['close'].pct_change(n)

        data['VOL'] = data['volume'].pct_change(1)

        data['target'] = data['RET 1'].shift(-1)
        data['target'] = (data['target'] > 0).astype(int)


        data = data.drop(['open', 'high', 'low', 'close', 'volume'],axis=1)
        data = data.dropna()
    
    def live_data(self, ticker=None):
        if ticker == None:
            ticker = self.ticker
        data = self.pull(ticker=ticker)
        list1 = [1,3, 5, 8, 21]
        list2 = [1]

        for n in list1:
            data[f'RET {n}'] = data['close'].pct_change(n)

        data['VOL'] = data['volume'].pct_change(1)

        data['target'] = data['RET 1'].shift(-1)
        data['target'] = (data['target'] > 0).astype(int)


        data = data.drop(['open', 'high', 'low', 'close', 'volume'],axis=1)

        self.new_data = data

        X,y = self.create_xy(data)


        return X[-1:]

    
    def get_pred(self):
        X = self.live_data()
        prob = self.model.predict_proba(X)
        pred = self.model.predict(X)
        prob_mod = round(max(prob[0]) * 100, 2)
        list2 = [pred, prob_mod]

        results = np.array(list2)

        return results





    
    