# -*- coding: utf-8 -*-
"""
Impute Optimizer -->  Applying an optimal Impute Strategy to a data set
"""
# Core
import os
import pandas as pd
import numpy as np
import sqlalchemy as sql
import getpass

# Plotting
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sktime.utils.plotting import plot_series
from pandas.plotting import autocorrelation_plot as autocorr_plot

# Importing sample data
time_series = pd.read_csv("cpc_data.csv")
