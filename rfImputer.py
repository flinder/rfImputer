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
        """
        Returns a dictionary: 'var_name': [missing indices]
        """
        pass
        
    
    def mean_mode_impute(var, missing_idx):

        """
        Impute mean for continuous and mode for categorical and binary data
        """
        
        if self.col_types[var] == 'regression':
            statistic = self.data[var].mean()
        elif detect_dtype(df[col]) == 'classification':
            statistic = self.data[var].mode()[0]
        else:
            raise ValueError('Unknown data type')

        out = repeat(statistic, repeats = len(missing_idx))
        return out

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

        imputations = rf.predict(X.iloc[missing_idx]) # Are these oob prediction? Clarify
        imputations = np.array(imputations)
        
    return imputations 


    def get_divergence(self, imputed_old, imputed):

        # Calcualte continuous divergence
        div_cat = 0
        norm_cat = 0
        div_cont = 0
        norm_cont = 0
        for var in imputed:
            if self.col_types[var] == 'regression':
                div = imputed_old - imputed
                div_cont += diff.dot(diff)
                norm_cont += imputed.dot(imputed)
            elif self.col_types[var] == 'classification':
                div = [1 if old != new else 0 for old, new in zip(imputed_old, imputed)]
                div_cat += sum(div)
                norm_cat += len(div)
            else:
                raise ValueError("Unrecognized variable type")

        return div_cat/norm_cat, div_cont/norm_cont
    
    
    def impute_df(self):
        """
        Imputation of missing data using random forest
        """

        for var in varlist:
            imputations['var'] = mean_mode_impute(df)


        div_cat = 0
        div_cont = 0
        stop = False
        
        while not stop:

            # Store results from previous iteration
            imputations_old = imputations
            div_cat_old = div_cat
            div_cont_old = div_cont

            for var in varlist:
                imputations['var'] = rf_impute(var, predictors)

            div_cat, div_cont = self.get_divergence(imputations_old, imputations)

            
            if div_cat > div_cat_old and div_cont > div_cont_old:
                stop = True

        return imputations

#### TODO:

# implement convergence criterion
# get imputation quality stats 
