# Core
import pandas as pd
import numpy as np

# Imputing
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from impyute.imputation.ts import locf
from impyute.imputation.ts import moving_window

# Listing Imputer and their Methods
# The grid could be extended with further methods and their params
imputers = {"SimpleImputer": 
                        ["mean", "median", "most_frequent"],
            "KNNImputer": 
                        ["uniform", "distance"],
            "Interpolate":
                        ["time", "linear", "index", "nearest", "zero",
                        "slinear", "quadratic", "cubic",
                        "piecewise_polynomial", "pchip", "akima"],
            "Interpolate_with_order": 
                        ["polynomial", "spline"],
            "TimeSeries_LOCF": None,
            "Moving_Win_Imputer":
                        # Moving Window uses uneven window lengths
                        # default is 3, 5, 7 ... 15 (could be extended if needed)
                        [x for x in range(3,16) if x % 2 != 0]
            }


def _find_best_imputer(data, data_imp, index_nan, imputers=imputers):
    # Dict for storing results
    results = {}
    
    # Looping over all methods and collecting deviations from original
    for imputer, params in imputers.items():
        
        # Simple Imputer handles most basic strategies like mean or median
        if imputer == "SimpleImputer":
            si_results = _simple_imputer(data, data_imp, params, index_nan)
            try:
                results.update(si_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        # KNN Imputer is based on a mean of nearest neighboors to NaN
        elif imputer == "KNNImputer":
            knn_results = _knn_imputer(data, data_imp, params, index_nan)
            try:
                results.update(knn_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        # Interpolate Methods from pandas
        # Some demand "order" param, therefore different handling
        elif imputer == "Interpolate":
            interpolate_results = _interpolate(data, data_imp, params, index_nan)
            try:
                results.update(interpolate_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        elif imputer == "Interpolate_with_order":
            interpolate_with_order_results = _interpolate_with_order(data, data_imp, params, index_nan)
            try:
                results.update(interpolate_with_order_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        # Time Series specific imputers
        elif imputer == "TimeSeries_LOCF":
            locf_results = _locf(data, data_imp, index_nan)
            try:
                results.update(locf_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        # Window Imputer applies imputing inside window splits 
        # that are applied to target data
        elif imputer == "Moving_Win_Imputer":
            mvi_results = _moving_win_imputer(data, data_imp, params, index_nan)
            try:
                results.update(mvi_results)
            except TypeError:
                print(f"\"{imputer}\" encountered a TypeError and will not return results")

        # Raise Error if unexpected imputing method
        else:
            raise NotImplementedError(f"The {imputer} is not implemented or \
                                      or not imported in module")
    # Return the best method from all applied
    best_method = _return_best_method(results)
    return best_method

# Individual imputer methods
# tied to the grid abouve

# Simple Strategy Imputer
def _simple_imputer(data, data_imp, params, index_nan):
    results = {}
    try:
        for strategy in params:
            imputed = (SimpleImputer(strategy=strategy)
                        .fit_transform(np.asarray(data_imp).reshape(-1, 1)))
            imputed = pd.Series(imputed.flatten(), index=data_imp.index)
            results["SimpleImputer" + "__" + strategy] = _compare(data, imputed, index_nan)
        return results
    except ValueError:
        print(f"Parameter {strategy} passed to \"Simple Imputer\" is not supported")

# KNN Imputer
def _knn_imputer(data, data_imp, params, index_nan):
    results = {}
    try:
        for weights in params:
            imputed = (KNNImputer(weights=weights)
                        .fit_transform(np.asarray(data_imp).reshape(-1, 1)))
            imputed = pd.Series(imputed.flatten(), index=data.index)
            results["KNNImputer" + "__" + weights] = _compare(data, imputed, index_nan)
        return results
    except ValueError:
        print(f"Parameter {weights} passed to \"KNN Imputer\" is not supported")

# Interpolation
def _interpolate(data, data_imp, params, index_nan):
    results = {}
    try:
        # Interpolation happens in both direction (forward and backward)
        for method in params:
            if method == "time":
                # Creating copies since below index formating happens inplace
                # and breaks the index formating for other methods
                data_imp_time = data_imp.copy()
                data_time = data.copy()
                # Applying datetime index formating for both data sets
                # needed when both enter _compare and are can be sorted
                data_imp_time.index = pd.to_datetime(pd.to_timedelta(data_imp_time.index, unit="days"))
                data_time.index = pd.to_datetime(pd.to_timedelta(data_time.index, unit="days"))
                imputed = data_imp_time.interpolate(method=method, limit_direction="both")
                # Reverting index format back to original
                imputed.reset_index(drop=True, inplace=True)
                # deleting copies
                del data_time, data_imp_time
            else:
                imputed = data_imp.interpolate(method=method, limit_direction="both")
            results["Interpolate" + "__" + method] = _compare(data, imputed, index_nan)
        return results
    except ValueError:
        print(f"Parameter \"{method}\" passed to \"Interpolate\" is not supported \
            or encounted and error.")

# Interpolation with order of magnitude
def _interpolate_with_order(data, data_imp, params, index_nan, order=2):
    # Interpolation happens in both direction (forward and backward)
    # Order can be tweaked (default quadratic)
    results = {}
    try:
        for method in params:
            imputed = data_imp.interpolate(method=method, limit_direction="both", order=order)
            results["Interpolate_with_order" + "__" + method] = _compare(data, imputed, index_nan)
        return results
    except ValueError:
        print(f"Parameter {method} passed to \"Interpolate with order\" is not supported")

# Last-Obeseration-Carried-Forward Imputer
def _locf(data, data_imp, index_nan):
    # Last Obeserveratino Carried Forward imputes based on past values
    # or forward values if consecutive NaNs
    results = {}
    try:
        imputed = locf(np.asarray(data_imp).reshape(1, -1))
        imputed = pd.Series(imputed.flatten(), index=data.index)
        results["TimeSeries_LOCF"] = _compare(data, imputed, index_nan)
        return results
    except Exception:
        print("\"LOCF Imputer\" encountered an error")

# Moving Window Imputer
def _moving_win_imputer(data, data_imp, params, index_nan):
    results = {}
    try:
        for wsize in params:
            # Remainder for data when win split applied
            remainder = -(len(data) % wsize)
            # Spliting in winsize 2d arrays und imputing with moving win
            # By default func=np.mean
            win_split = np.asarray(list(zip(*[iter(data_imp)] * wsize)))
            imputed = np.asarray(moving_window(win_split, wsize=wsize))
            # Excluding remainder if found from data for index alignment
            if remainder != 0:
                imputed = pd.Series(imputed.flatten(),
                                    index=data.index[:remainder])
                # Intersecting NaNs without the remainder to run compare
                index_nan_adj = np.intersect1d(index_nan, data.reset_index(drop=True)
                                            .index[:remainder])
                results["Moving_Win_Imputer" + "__" + str(wsize)] = _compare(data, imputed, 
                                                                    index_nan=index_nan_adj)
            # If no remainder - take all train data
            else:
                imputed = pd.Series(imputed.flatten(), index=data.index)
                results["Moving_Win_Imputer" + "__" + str(wsize)] = _compare(data, imputed,
                                                                    index_nan=index_nan_adj)
        return results
    except AssertionError:
        print(f"Parameter {wsize} passed to \"Moving Window Imputer\" is not uneven")


def _compare(data_org, data_imp, index_nan):
    """
    Utility method to compare the deviation of imputed from actuals
    """
    compare = (pd.concat([data_org[index_nan], data_imp[index_nan]], axis=1)).sort_index()
    compare.columns = ["original", "imputed"]
    compare["Delta"] = compare["original"] - compare["imputed"]
    return abs(sum(compare["Delta"]))


def _return_best_method(results):
    """
    Method used in conjunction with run imputers to iterate through all methods
    and return the least deviation of imputed output from the actuals.
    results: dict
    """
    best_method = ""
    for k, v in results.items():
        if v == min(results.values()):
            best_method = (f"{k}: {v:.3f}")
    return best_method