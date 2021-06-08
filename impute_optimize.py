from Imputer import impute
import pandas as pd



def impute_optimizer(data, n_iter=10, to_nan=0.2):
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
    for x in range(n_iter):
        method, deviation = impute(data, to_nan=to_nan).split(" ")
        lst.append((method, float(deviation)))
    df = pd.DataFrame(lst, columns=["method", "deviation"])
    # Return summary table
    return (df.pivot_table(index='method', values='deviation',
                           aggfunc=["min", "max", "mean", "count"])
            .sort_values([('count', 'deviation')], axis=0, ascending=False))