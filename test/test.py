import pandas as pd
import sys
sys.path.insert(1, '../')
from rfImputer import rfImputer
import numpy as np
from pprint import pprint

df = pd.read_csv('census_sample.csv', header = None)

# Fill in some missing data
n = df.shape[0]
for col in df.columns:
    n_missing = np.random.randint(low = 0, high = n/2, size = 1)[0]
    missing_idx = np.random.choice(n, n_missing, replace = False)
    df[col].iloc[missing_idx] = None


# Get some continuous variables
df['cont_1'] = np.random.randn(n)
df['cont_2'] = np.random.randn(n)


imp_df = rfImputer(df, is_classification = ['3', '10'])

imp_df.impute('simple')
print "IMPUTED"

out = imp_df.imputed_df()




