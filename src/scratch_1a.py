import pandas as pd
import numpy as np
import os

filename = 'data/good_pivot_full.csv'

see = pd.read_csv(filename, index_col=0)

BIG_NUMBER = 1e25

price = see['finalPrice'].to_numpy()

dd1= see.pivot_table(index=see.index, columns=['rangeLow'], values='liquidity', fill_value=0).cumsum()
ind1 = (0 == dd1).cumprod().astype('int')


dd2 = see.pivot_table(index=see.index, columns=['rangeHigh'], values='liquidity', fill_value=0).cumsum()
ind2 = (0 == dd2).cumprod().astype('int')

dd3=dd1.subtract(dd2, fill_value=0)
ind3 = (0 == dd3).cumprod().astype('int')
histogram_out = dd3.cumsum(axis=1)


B = np.where((price[:, np.newaxis])*(-BIG_NUMBER*ind3+1-ind3) > dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
price_lt = -((price[:, np.newaxis] - dd3.columns.to_numpy()[np.newaxis,:])*B + BIG_NUMBER*(1-B)).min(axis=1)+price

B1 = np.where((price[:, np.newaxis])*(BIG_NUMBER*ind3+1-ind3) <= dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
price_rt = ((dd3.columns.to_numpy()[np.newaxis,:]-price[:, np.newaxis])*B1+BIG_NUMBER*(1-B1)).min(axis=1)+price


#dd4.to_clipboard()


