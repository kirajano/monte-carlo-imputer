import pandas as pd
import numpy as np

# Imputing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from impyute.imputation.ts import locf
from impyute.imputation.ts import moving_window

# Custom
from impute_strategies import _find_best_imputer
from preprocessing import _nan_data, _index_to_nan, _index_to_nan_fast


def impute_apply(data, n_iter, to_nan=0.2, fast_impute=False):
      """
      Applies the impute strategy that resulted with most occurences 
      of providing the least error from impute optimizer. The returned
      data set is then imputed with no zero values.

      PARAMS:
      ----------
            data : pd.Series
            n_iter : INT, optional. The default is 10.
            to_nan : FLOAT, optional. The default is 0.2
      RETURNS:
      ----------
      data: pd.Series
      imputer: str
      param: str / int
      """
      output = impute_optimizer(data, n_iter=n_iter, to_nan=0.2, fast_impute=False)
      imputer, param = output.iloc[0,:].name.split("__")
      param = param.replace(":", "")

      if imputer == "SimpleImputer":
            ix = data.index.copy()
            data = (SimpleImputer(strategy=param)
                        .fit_transform(np.asarray(data).reshape(-1, 1)))
            data = pd.Series(data.flatten(), index=ix)
            del ix

      elif imputer == "KNNImputer":
            ix = data.index.copy()
            data = (KNNImputer(weights=param)
                        .fit_transform(np.asarray(data).reshape(-1, 1)))
            data = pd.Series(data.flatten(), index=ix)
            del ix

      elif imputer == "Interpolate":
            if param == "time":
                  data.index = pd.to_datetime(pd.to_timedelta(data.index, unit="days"))
                  data = data.interpolate(method=param, limit_direction="both")
            else:
                  data = data.interpolate(method=param, limit_direction="both")

      elif imputer == "Interpolate_with_order":
            # Order can be tweaked (default quadratic)
            data = data.interpolate(method=param, limit_direction="both", order=2)
      
      elif imputer == "TimeSeries_LOCF":
            ix = data.index.copy()
            data = locf(np.asarray(data).reshape(1, -1))
            data = pd.Series(data.flatten(), index=ix)
            del ix
      
      elif imputer == "Moving_Win_Imputer":
            ix = data.index.copy()
            param = int(param)
            remainder = -(len(data) % param)
            data = np.asarray(list(zip(*[iter(data)] * param)))
            data = np.asarray(moving_window(data, wsize=param))
            if remainder != 0:
                  data = pd.Series(data.flatten(),
                                    index=ix[:remainder])
            else:
                  data = pd.Series(data.flatten(), index=ix)
            del ix
      else:
            raise Exception
            print("Imputer passed through \"impute_optimize\" cannot be applied")
            print(f"Value passed: {impter}")
      
      return data, imputer, param


def impute_optimizer(data, n_iter=10, to_nan=0.2, fast_impute=False):
    """
    Runs specified iternation on a training set calling impute() to find
    the best possible method to be used. Impute() is being iterated n times.
    Results are displayed as pivot table format with a summary of each interpolation method.
    Metrics: mean, max, min, and count how many times method was the best.

    PARAMS:
    ----------
      data : pd.Series
      n_iter : INT, optional. The default is 10.
      to_nan : FLOAT, optional. The default is 0.2

    RETURNS:
    -------
      Pivot table: pd.DataFrame
    """
    # Store results
    lst = []
    for i in range(n_iter):
        method, deviation = impute(data, to_nan=to_nan, fast_impute=fast_impute).split(" ")
        lst.append((method, float(deviation)))
    df = pd.DataFrame(lst, columns=["method", "deviation"])
    # Return summary table
    df = (df.pivot_table(index='method', values='deviation',
                           aggfunc=["min", "max", "mean", "count"])
            .sort_values([('count', 'deviation')], axis=0, ascending=False))
    df.columns = pd.Index(["min_error", "max_error", "mean_error", "count_best_method"])
    df.index.name = "Imputer__Parameter"
    return df


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
      to_nan, existing_nans = _nan_data(data, to_nan=0.2)
     
      # Setting random data to NaNs (except existing zeroes)
      # Preserving the original data (for deviation measurement)
      if fast_impute == True:
            data_imp, index_nan = _index_to_nan_fast(data, existing_nans, to_nan)
      else:
            data_imp, index_nan = _index_to_nan(data, existing_nans, to_nan)

      # Apply imputers and return results on their deviation
      return _find_best_imputer(data, data_imp, index_nan)
