# -*- coding: utf-8 -*-
"""
Impute Optimizer -->  Applying an optimal Impute Strategy to a data set with Monte Carlo approach
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
from sktime.utils.plotting import plot_series

# Custom
from impute_optimize import impute, impute_apply, impute_optimizer

# Suppres warning when encountering empty windows in Window Imputer
import warnings
warnings.filterwarnings("ignore")


# Importing sample data
# Using squeeze to return Series since one column
time_series = pd.read_csv("cpc_data.csv", usecols=["cpc"], index_col=False, squeeze=True)


###################################
### RUN SINGLE OF ALL IMPUTERS ####
###################################

imp, score = impute(time_series).split(" ")
print("Runinng single impute iteration")
print(imp, score)


#####################################
### MULTIPLE RUN OF ALL IMPUTERS ####
#####################################

# Number of runs
iteration_runs = 150
print(f"Running {iteration_runs} impute iterations ...")
# Return overview
results = impute_optimizer(time_series, n_iter=iteration_runs, fast_impute=False)
print(results)
# Or with Pipe
# print(time_series.pipe(impute_optimizer, n_iter=iteration_runs))


###############################################################
### MULTIPLE RUN OF ALL IMPUTERS AND AUTO-APPLY TO DATASET ####
###############################################################

# Number of runs
iteration_runs = 150

# Running optimizer and applying the best imputer to data
print(f"Running {iteration_runs} impute iterations and applying best imputer...")
time_series, imputer, parameter = impute_apply(time_series, n_iter=iteration_runs, fast_impute=False)
# Check after applying
print(f"Applying \"{imputer}\" with \"{parameter}\" as parameter to dataset ...")
print("After imputing dataset has nulls: ", time_series.isnull().sum())