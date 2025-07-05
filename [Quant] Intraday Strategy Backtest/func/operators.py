#%%
import numpy as np
import numba as nb
import pandas as pd
import os
from scipy import stats
from copy import deepcopy
import warnings


def _wrap_DataFrame(array, index=None, columns=None):
    return pd.DataFrame(array, index=index, columns=columns)

# Cross-Sectional Operators
def normalize(x: np.array, useStd:bool=False, limit:float=0.0):
    """
    Calculates the mean value of all valid alpha values for a certain date, then subtracts that mean from each element
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values
    with warnings.catch_warnings(): # ignore empty rows
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if useStd:
            return (x - np.nanmean(x, axis=-1).reshape(-1, 1)) / np.nanstd(x, axis=-1).reshape(-1, 1)
        else:
            return x - np.nanmean(x, axis=-1).reshape(-1, 1)

def rank(x: np.array) -> np.array:
    """
    Rank 2D array values row by row, and then scale values to be in [0, 1]
    ex) [1, 2, 3, 4] -> [0, 1/3, 2/3, 1]
    """ 
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.rank(axis=1, pct=True)
    elif isinstance(x, np.ndarray):
        div = (np.count_nonzero(~np.isnan(x), axis=-1) -1)
        div = np.where(div <= 1, 1, 1/div)
        return np.where(np.isnan(x), np.nan, x.argsort().argsort()) * div.reshape((div.size, 1))

def scale(weight):
    """
    Scales the weights by dividing each element by the sum of the absolute values of its row.
    """
    sum_abs_values = np.nansum(np.abs(weight), axis=1)
    return weight / sum_abs_values[:, np.newaxis]

def zscore(x: np.array) -> np.array:
    """
    Calculates the z-score of each element in a 2D array row-wise, which standardizes the values by subtracting the mean and dividing by the standard deviation.
    """
    return stats.zscore(x, axis=1, nan_policy='omit')

def quantile(x:np.array, q:int):
    """
    Ranks 2D array values row by row and then scales them into q quantiles.
    """
    temp = np.floor(rank(x) * q)/q
    return np.where(temp == 1, 1-1/q, temp)

# %%

# Time-Series Operators
def ts_returns(x, d):
    """
    Calculates the percentage change in values over a specified number of periods d.
    """
    return x.pct_change(d)

def ts_delta(x, d):
    """
    Calculates the difference between the current value and the value from d periods ago.
    """
    return x.diff(d)

def ts_delay(x, d):
    """
    Shifts the values by d periods.
    """
    return x.shift(d)

def ts_returns(x, d):
    return x.pct_change(d)

def ts_max(x, d):
    """
    Computes the rolling maximum over a window of d periods.
    """
    return x.rolling(window=d, min_periods=1).max()

def ts_min(x, d):
    """
    Computes the rolling minimum over a window of d periods.
    """
    return x.rolling(window=d, min_periods=1).min()

def ts_argmax(x, d):
    """
    Finds the index of the maximum value over a rolling window of d periods and adds 1 to it.
    """
    return x.rolling(window=d, min_periods=1).apply(np.argmax, engine='numba', raw=True).add(1)

def ts_argmin(x, d):
    """
    Finds the index of the minimum value over a rolling window of d periods and adds 1 to it.
    """
    return x.rolling(window=d, min_periods=1).apply(np.argmin, engine='numba', raw=True).add(1)

def ts_mean(x, d):
    """
    Calculates the rolling mean over a window of d periods.
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.rolling(window=d, min_periods=1).mean()
    elif isinstance(x, np.ndarray):
        return pd.DataFrame(x).rolling(window=d, min_periods=1).mean().values

def ts_decay_linear(df, d):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    _temp = np.arange(1, d+1)
    return df.rolling(d).apply(lambda x: np.dot(x, _temp)/ (d * (d+1)/2), raw=True)

def ts_median(x, d):
    return x.rolling(window=d, min_periods=1).median()

def ts_std(x, d):
    return x.rolling(window=d, min_periods=1).std()

def ts_rank(x, d):
    # return x.rolling(window=d, min_periods=1).apply(lambda x: x.rank().iloc[-1])
    if type(x) == pd.core.frame.DataFrame:
        index, columns = x.index, x.columns
    sw = np.lib.stride_tricks.sliding_window_view(x, d, axis=0).T
    scores_np = (sw <= sw[-1:, ...]).sum(axis=0).T / sw.shape[0]    
    return _wrap_DataFrame(np.concatenate((np.zeros((d-1, x.shape[1])), scores_np), axis=0), index, columns).replace(0, np.nan)

def ts_sum(x, d):
    # return x.rolling(window=d).apply(np.sum)
    return x.rolling(window=d, min_periods=1).apply(np.nansum, engine='numba', raw=True)

def ts_prod(x, d):
    # return x.rolling(window=d, min_periods=1).apply(np.prod)
    return x.rolling(window=d, min_periods=1).apply(np.nanprod, engine='numba', raw=True)

def ts_corr(x, y, d):
    return x.rolling(window=d, min_periods=1).corr(y)

def ts_cov(x, y, d):
    return x.rolling(window=d, min_periods=1).cov(y)

def ts_ir(x,  d):
    return (x - ts_mean(x, d))/ ts_std(x, d)

def ts_stairs(array, d):
    for i in range(0, array.shape[0], d):
        # Check if we have enough rows remaining to do the assignment
        if i + d - 1 < array.shape[0]:
            # Assign the value of the 10*i-th row to the next 9 rows
            array[i + 1:i + d] = array[i]
    return array

# Group Operators
def _groupby_aggregate(df1, df2, agg_func=zscore):
    if type(df2) == np.ndarray:
        categories = list(set(df2[0]))
        temp = np.stack([(np.where(df2==category, df1, np.nan)) for category in categories])
        return np.nansum([rank(temp[i]) for i in range(len(temp))], axis=0)

    else:
        raise TypeError

def group_rank(df1, df2):
    return _groupby_aggregate(df1, df2, agg_func=rank)

def group_zscore(df1, df2):
    return _groupby_aggregate(df1, df2, agg_func=zscore)

