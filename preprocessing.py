import numpy as np
import pandas as pd
from itertools import count as cnt


def _nan_data(data, to_nan=0.2):
    """
    Define how many values will be set to NaN that are not and return indicies of existing NaNs.
    PARAMS:
    ------------
        data: pd.Series
        to_nan: float, Default 20%
    RETURNS:
    ------------
        to_nan: int
        existing_nans: pd.Series.Index
    """
    # Number of values to be NaNed as int
    to_nan = int(len(data) * to_nan)
    # Existing NaN's as indicies
    existing_nans = data[data.isnull() == True].index
    return to_nan, existing_nans

# Conditioning before randomizing
# Randomly picking index to set to nan except existing nan
def _index_to_nan(data, existing_nans, to_nan):
    """
    Randomly picking index to set to nan except existing nan. NaNs applied to copy of dataset.
    PARAMS:
    --------------------------------
        data: pd.Series
        existing_nans: list of int 
        to_nan: float
    RETURNS:
    --------------------------------
        data_imp: pd.Series
        index_nan: list of int
    """
    index_nan = np.random.choice([i for i in range(len(data)) if i not in existing_nans],
                    size=to_nan, replace=False)
    data_imp = data.copy()
    data_imp[index_nan] = np.nan
    return data_imp, index_nan

# Conditioning after randomizing
# Randomly picking index to set to nan except existing nan
def _index_to_nan_fast(data, existing_nans, to_nan):
      """
      Randomly picking index to set to nan except existing nan. NaNs applied to copy of dataset.
      Difference to above method, it uses an iterator and hence is efficient for larger datasets.
      Advisable for large datasets.
      PARAMS:
      ---------------------------------
            data: pd.Series
            existing_nans: list of int
            to_nan: float
      RETURNS:
      ---------------------------------
            data_imp: pd.Series
            index_nan: list of int    
      """
      index_nan = []
      randgen = (np.random.choice(len(data)) for _ in cnt(start=1))
      for i in range(to_nan):
            ix = next(filter(lambda x: x not in existing_nans and x not in index_nan, randgen))
            index_nan.append(ix)
      data_imp = data.copy()
      data_imp[index_nan] = np.nan
      return data_imp, index_nan
