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
        self.missing = self.find_missing()
        self.prop_missing = self.prop_missing()
        self.imputed_values = {}

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
        try:
            self.is_classification = kwargs['is_classification']
        except KeyError:
            self.is_classification = []
        try:
            self.is_regression = kwargs['is_regression']
        except KeyError:
            self.is_regression = []

        # Detect unspecified column types
        self.col_types = self.assign_dtypes()


    def detect_dtype(self, x):
        """
        Detect if variable requires classification or regression.

        Needs to be improved
        """

        if x.dtype == 'float':
            out = 'regression'
        elif x.dtype == 'object':
            out = 'classification'
        elif x.dtype == 'int64':
            if len(x.unique()) < 4:
                out = 'classification'
            else:
                out = 'regression'
        else:
            msg = 'Unrecognized data type: %s' %x.dtype
            raise ValueError(msg)
        
        return out

    def assign_dtypes(self):
        """
        Assign prespecified and detect non specified column types
        """
        dtypes = {}
        for var in self.data.columns:
            if var in self.is_classification:
                dtypes[var] = 'classification'
            elif var in self.is_regression:
                dtypes[var] = 'regression'
            else:
                dtypes[var] = self.detect_dtype(self.data[var])

        return dtypes

    def find_missing(self):
        """
        Returns a dictionary: 'var_name': [missing indices]
        """
        missing = {}
        var_names = self.data.columns
        for var in var_names:
            col = self.data[var]
            missing[var] = col.index[col.isnull()]

        return missing

    def prop_missing(self):
        out = {}
        n = self.data.shape[0]
        for var in self.data.columns:
            out[var] = float(len(self.missing[var])) / float(n)

        return out
        
    
    def mean_mode_impute(self, var):

        """
        Impute mean for continuous and mode for categorical and binary data
        """
        
        if self.col_types[var] == 'regression':
            statistic = self.data[var].mean()
        elif self.col_types[var] == 'classification':
            statistic = self.data[var].mode()[0]
        else:
            raise ValueError('Unknown data type')

        out = np.repeat(statistic, repeats = len(self.missing[var]))
        return out

    def imputed_df(self):
        '''
        Fills the missing values in the input data frame with the values stored
        in imputed_values and returns a new data frame (copy)
        
        '''
        if len(self.imputed_values) == 0:
            raise ValueError('No imputed values available. Call impute() first')
        
        out_df = self.data.copy()
        for var in self.data.columns:
            for idx, imp in zip(self.missing[var], self.imputed_values[var]):
                out_df[var].iloc[idx] = imp

        return out_df
            

    def impute(self, imputation_type):

        if imputation_type == 'simple':
            for var in self.data.columns:
                self.imputed_values[var] = self.mean_mode_impute(var)

        else:
            msg = 'Unrecognized imputation type: %s' %imputation_type
            raise ValueError(msg)
        
            

    def rf_impute(impute, predict, pre_imputations, missing_idx):
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
