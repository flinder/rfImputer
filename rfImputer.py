# Take a dataset and impute values for each variable using a random
# forest with all other variables as predictors

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
import copy

class rfImputer(object):

    def __init__(self, data, **kwargs):
        '''
        
        '''
        
        self.data = data
        self.missing = self.find_missing()
        self.prop_missing = self.prop_missing()
        self.imputed_values = {}
        self.imputation_scores = {}

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
            if len(x.unique()) < 4:
                out = 'classification'
            else:
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

                    
    def rf_impute(self, impute_var, data_imputed, rf_params):
        """
        Impute missing values in a data set using random forest
        impute: name or column number of variable to be imputed
        predict: list of names or column numbers of variables used as predictors
        data: pandas data frame containing impute and predict
        forest: a sklearn random Forest Classifier or Regressor

        Returns a list of values for the originally missing data
        """

        y = data_imputed[impute_var]
        include = [x for x in self.incl_predict if x != impute_var]
        X = data_imputed[include]

        if self.col_types[impute_var] == 'classification':
            rf = RandomForestClassifier(oob_score = True, **rf_params)
            rf.fit(y = y, X = X)
            oob_predictions = np.argmax(rf.oob_decision_function_, axis = 1)
            oob_imputation = oob_predictions[self.missing[impute_var]]
            
        else:
            rf = RandomForestRegressor(oob_score = True, **rf_params)
            rf.fit(y = y, X = X)
            oob_imputation = rf.oob_prediction_[self.missing[impute_var]]
    
        self.imputation_scores[impute_var] = rf.oob_score_
        self.imputed_values[impute_var] = oob_imputation


    def get_divergence(self, imputed_old):

        # Calcualte continuous divergence
        div_cat = 0
        norm_cat = 0
        div_cont = 0
        norm_cont = 0
        for var in self.imputed_values:
            if self.col_types[var] == 'regression':
                div = imputed_old[var] - self.imputed_values[var]                
                div_cont += div.dot(div)
                norm_cont += self.imputed_values[var].dot(self.imputed_values[var])
            elif self.col_types[var] == 'classification':
                div = [1 if old != new
                       else 0
                       for old, new in zip(imputed_old[var], self.imputed_values[var])]
                div_cat += sum(div)
                norm_cat += len(div)
            else:
                raise ValueError("Unrecognized variable type")

        
        if norm_cat == 0:
            cat_out = 0
        else:
            cat_out = div_cat / norm_cat
        if norm_cont == 0:
            cont_out = 0
        else:
            cont_out = div_cont / norm_cont
        
        return cat_out, cont_out
    


    def impute(self, imputation_type, rf_params = None):

        if imputation_type == 'simple':
            for var in self.incl_impute:
                self.imputed_values[var] = self.mean_mode_impute(var)

        elif imputation_type == 'random_forest':

            print "-" * 50
            print "Starting Random Forest Imputation"
            print "-" * 50

            # Do a simple mean/mode imputation first
            
            for var in self.incl_impute:
                self.imputed_values[var] = self.mean_mode_impute(var)
            
            # Rf Imputation Loop
            div_cat = float('inf')
            div_cont = float('inf')
            stop = False
            i = 0
            while not stop:

                i += 1

                # Store results from previous iteration
                imputations_old = copy.copy(self.imputed_values)
                div_cat_old = div_cat
                div_cont_old = div_cont

                # Make an imputed df
                data_imputed = self.imputed_df()
                
                for var in self.incl_impute:
                    self.rf_impute(var, data_imputed, rf_params)

                div_cat, div_cont = self.get_divergence(imputations_old)

                print "Iteration %d:" %i
                print "Categorical divergence: %f" %div_cat
                print "Continuous divergence: %f" %div_cont

                # Check if stopping criterion is met
                if div_cat >= div_cat_old and div_cont >= div_cont_old:
                    stop = True


        else:
            msg = 'Unrecognized imputation type: %s' %imputation_type
            raise ValueError(msg)



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

#### TODO:

# implement convergence criterion
# get imputation quality stats 
