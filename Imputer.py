# -*- coding: utf-8 -*-
"""
Impute Optimizer -->  Applying an optimal Impute Strategy to a data set
"""
# Core
import sys
import os
import pandas as pd
import numpy as np
from itertools import count as cnt

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.core.arrays.base import ExtensionOpsMixin
import seaborn as sns
from sktime.utils.plotting import plot_series

# Custom
from impute_strategies import find_best_imputer
from preprocessing import nan_data, index_to_nan, index_to_nan_fast
#from impute_optimize import impute_optimizer

# Suppres warning when encountering empty windows in Window Imputer
import warnings
warnings.filterwarnings("ignore")


# Importing sample data
# Using squeeze to return Series since one column
time_series = pd.read_csv("cpc_data.csv", usecols=["cpc"], index_col=False, squeeze=True)


def impute_optimizer(data, n_iter=10, to_nan=0.2, fast_impute=False):
    """
    Runs specified iternation on a training set calling impute() to find
    the best possible method to be used. Data is being iterated n times
    and values to be set randomly to_nan are specified as percentage of total
    data.

    Parameters
    ----------
    data : pd.Series
    n_iter : INT, optional. The default is 10.
    to_nan : FLOAT, optional. The default is 0.2

    Returns
    -------
    DataFrame
        Pivot table format with a summary of each interpolation method.
        Metrics: mean, max, min, and count how many times method was the best.
    """
    # Store results
    lst = []
    for i in range(n_iter):
        method, deviation = impute(data, to_nan=to_nan, fast_impute=fast_impute).split(" ")
        #print("Total nulls:\n", sum(data.isnull() == True))
        lst.append((method, float(deviation)))
    df = pd.DataFrame(lst, columns=["method", "deviation"])
    # Return summary table
    return (df.pivot_table(index='method', values='deviation',
                           aggfunc=["min", "max", "mean", "count"])
            .sort_values([('count', 'deviation')], axis=0, ascending=False))


def impute(data, to_nan=0.2, fast_impute=False):
      """
      Function to apply several imputing methods to randomly selected data.
      Randomly selected data gets set to NaN for testing.

      PARAMS:
      ----------
            data : Dataframe or Series
                  Will be used to randomly set to NaN.
            to_nan : Float
                  Percentage of Dataframe to be set to NaN.

      RETURNS:
      -------
            String of best method with total deviation to orginial data
      """
      # Number of values to be set to NaN in dataset along with existing NaNs (indicies)
      to_nan, existing_nans = nan_data(data, to_nan=0.2)
     
      # Setting random data to NaNs (except existing zeroes)
      # Preserving the original data (for deviation measurement)
      if fast_impute == True:
            data_imp, index_nan = index_to_nan_fast(data, existing_nans, to_nan)
      else:
            data_imp, index_nan = index_to_nan(data, existing_nans, to_nan)

      # Apply imputers and return results on their deviation
      return find_best_imputer(data, data_imp, index_nan)

# Check single run of all imputers
imp, score = impute(time_series).split(" ")
print("RESULT: \n", imp, score)

# Check multiple runs of all imputers
print("Mulit_RESULT:\n", impute_optimizer(time_series, n_iter=10))

