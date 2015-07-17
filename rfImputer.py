# Take a dataset and impute values for each variable using a random
# forest with all other variables as predictors

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np


class rfImputer(object):

    def __init__(self, data, **kwargs):
        '''
        
        '''
        
        self.data = data
        self.miss_idx = self.find_missing()
        self.imputed_values = None
        self.col_types = self.detect_dtype(data)

        # Columns in which values should be imputed
        if 'incl_impute' in kwargs:
            self.incl_impute = kwargs['incl_impute']
        elif 'excl_impute' in kwargs:
            self.incl_impute = [c for c in data.columns if c not in kwargs['excl_impute']]
        else:
            self.incl_impute = data.columns

        # Columns that should be used as predictors for the imputation
        if 'incl_predict' in kwargs:
            self.incl_predict = kwargs['incl_predict']
        elif 'excl_predict' in kwargs:
            self.incl_predict = [c for c in data.columns if c not in kwargs['excl_predict']]
        else:
            self.incl_predict = data.columns

        ## Column types

        # Manually specified column types
        self.is_classification = kwargs['is_classification']
        self.is_regression = kwargs['is_regression']

        # Detect unspecified column types
        
        
    def detect_dtype():
        pass

    def mean_mode_impute():
        pass

    def rf_impute():
        pass

    def imputed_df():
        pass
