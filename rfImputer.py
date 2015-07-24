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
        self.col_types = self.assign_dtypes(data.columns, self.is_classificaion,
                                            self.is_regression, data)


    def detect_dtype(x):
        """
        Detect if variable requires classification or regression.

        Needs to be improved
        """

        if isinstance(x, float):
            out = 'regression'
        elif isinstance(x, str):
            out = 'classification'
        elif isinstance(x, int):
            if len(x.unique()) < 4:
                out = 'classification'
            else:
                out = 'regression'
        else:
            raise ValueError('Unrecognized data type')
        
        return out

    def assign_dtypes(variables, is_classification, is_regression, data):
        """
        Assign prespecified and detect non specified column types
        """
        dtypes = {}
        for var in variables:
            if var in is_classification:
                dtypes[var] = 'classification'
            elif var in is_regression:
                dtypes[var] = 'regression'
            else:
                dtypes[var] = detect_dtype(data[var])

        return dtypes

    def find_missing(data):
        pass
        
    
    def mean_mode_impute(df):

        """
        Impute mean for continuous and mode for categorical and binary data
        df: pandas data frame to impute
        missing_threshold: Maximum allowable proportion missing
        """
        df_out = df.copy()
        # Impute mean or mode depending on dtype
        for col in df_out.columns:
            idx = df[df[col].isnull()].index
            if detect_dtype(df[col]) == 'regression':
                df_out[col].iloc[idx] = df[col].mean()
            elif detect_dtype(df[col]) == 'classification':
                df_out[col].iloc[idx] = df[col].mode()[0]
            else:
                raise ValueError('Unknown data type')
        return(df_out)

    def imputed_df():
        pass

    def rf_impute(self, impute, predict, data, missing_idx):
        """
        Impute missing values in a data set using random forest
        impute: name or column number of variable to be imputed
        predict: list of names or column numbers of variables used as predictors
        data: pandas data frame containing impute and predict
        forest: a sklearn random Forest Classifier or Regressor

        Returns a list of values for the originally missing data
        """

        y = data[impute]
        X = data[predict]

        if detect_dtype(y) == 'classification':
            rf = RandomForestClassifier()
        else:
            rf = RandomForestRegressor()

        rf.fit(y = y, X = X)

    return rf.predict(X.iloc[missing_idx]) # Are these oob prediction? Clarify


    def impute_df(df, excl_impute = [], excl_predict = [], missing_threshold = 0.6,
                  factors = [], n_iter = 2):
        """
        Imputation of missing data using random forest

        df: data frame to be imputed
        excl_impute: list of strings, variables that should not be imputed
        excl_predict: list of strings, variables that should not be used as predictors
        factors: list of strings, variables that should be explicitly treated as
                 factors (classification)
        n_iter: Integer, number of iterations
        """

        # Exclude all vars that have too many missing values
        for col in df.columns:
            proportion_missing = prop_missing(df[col])
            if proportion_missing > missing_threshold:
                df.drop(col, axis = 1, inplace = True)

        # Initial guess imputation (mean/mode)
        df_imputed = mean_mode_impute(df)

        # Do the random forest imputation for each column
        i = 0
        while i < n_iter:
            print "=" * 50
            print "Iteration %d" %(i + 1)
            print "=" * 50
            for col in df.columns:

                if col in excl_impute:
                    continue

                print "Imputing %s" %col
                idx = df[df[col].isnull()].index
                predictors = [c for c in df.columns if c != col and c not in excl_predict]
                imp_values = rf_impute(col, predictors, df_imputed, idx)
                df[col].iloc[idx] = imp_values
                df_imputed[col].iloc[idx]  = imp_values
            i += 1

        print df.head()
        print df.shape
        return(df)


#### TODO:

# implement convergence criterion
# 
