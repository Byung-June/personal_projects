from os import path
import sys

sys.path.append(path.abspath('..'))

import numpy as np
import numba as nb
from sklearn.model_selection import TimeSeriesSplit

from func.read_files import get_intra_data
from func.read_files import get_daily_data
from func.operators import *
from func.evaluation import get_stats_intraday


def get_weight_share_from_ratio(weight, account, price=None):
    if price is None:
        price = get_intra_data('midp_twa', start=None, end=None)
    weight_share_target = weight * account // price.shift(1)
    
    if isinstance(weight_share_target, pd.DataFrame):
        return weight_share_target.fillna(0)
    elif isinstance(weight_share_target, np.ndarray):
        return np.nan_to_num(weight_share_target, nan=0.) 
    else:
        raise TypeError

@nb.jit(nopython=True)
def _get_weight_adj_iter(weight_share_target, volume):
    weight_share_target_adj = np.zeros_like(weight_share_target)
    weight_share_target_adj = np.concatenate((np.zeros((1, volume.shape[1])), weight_share_target_adj), axis=0)

    for i in nb.prange(1, len(weight_share_target_adj)):
        weight_share_target_adj[i] = weight_share_target_adj[i-1] + np.minimum(volume[i-1], weight_share_target[i-1] - weight_share_target_adj[i-1])
    return weight_share_target_adj[1:]

def get_weight_share_adj(weight_share_target, volume=None, feasible_volume_ratio=1/100):
    if volume is None:
        volume = get_intra_data('volume', start=None, end=None)
    volume = volume.fillna(0).values * feasible_volume_ratio
    if isinstance(weight_share_target, pd.DataFrame):
        weight_share_target = weight_share_target.fillna(0).values
    elif isinstance(weight_share_target, np.ndarray):
        weight_share_target = np.nan_to_num(weight_share_target, nan=0)

    return _get_weight_adj_iter(weight_share_target, volume)


def get_cost(weight_share_target, cost):
    return (np.abs(weight_share_target.diff()) * cost).sum(axis=1)

def get_pnl(weight_share_target, price, cost=None):
    if isinstance(weight_share_target, pd.DataFrame):
        weight_share_target = weight_share_target.values
    if isinstance(weight_share_target, np.ndarray):
        weight_share_target = np.roll(weight_share_target, 1, axis=0)
        weight_share_target[0] = 0
    
    pnl = (price.diff() * weight_share_target).sum(axis=1)
    if cost is not None:
        pnl = pnl - get_cost(weight_share_target, cost)
    pnl.fillna(0, inplace=True)
    return pnl

def get_neutral_weight(weight, roll=1):
    if roll:
        return np.roll(scale(normalize(weight)), 1, axis=0) 
    return scale(normalize(weight))

def get_daily_sum(array):
    if type(array) == np.ndarray:
        return array.reshape(-1, 330).sum(axis=1)
    elif type(array) == pd.Series:
        return array.values.reshape(-1, 330).sum(axis=1)
    
def _no_overnight(weight):
    weight[::330] = 0
    weight[-1::330] = 0
    return weight

def get_backtest(weight, account, price, volume=None, feasible_volume_ratio=1/100, borrow_rate=0.03, overnight=False, plot=True):
    if not overnight:
        weight = _no_overnight(weight)
    weight_share_target = get_weight_share_from_ratio(weight, account, price)
    weight_share_adj = get_weight_share_adj(weight_share_target, volume.shift(1), feasible_volume_ratio)
    
    pnl = get_pnl(weight_share_target, price)
    pnl_adj = get_pnl(weight_share_adj, price)

    if plot:
        temp = pd.concat([pnl, pnl_adj], axis=1).cumsum()
        temp.columns = ['before volume filter', 'after volume filter']
        temp.plot()

    # borrow_rate = 0.03 / 360
    # est_fee = np.where(weight<0, -weight * borrow_rate, 0)

    return weight_share_target, weight_share_adj, pnl, pnl_adj

def _get_sharpe(pnl):
    return np.nanmean(pnl)/np.nanstd(pnl)

def cross_validate_strategy(strategy, params, account, price, volume, test_size=10, k=3):
    n_splits = volume.shape[0]//330//test_size - 1
    test_size *= 330

    # Generate weight and pnl of given 'strategy' for 'params'
    weight_list = []
    pnl_list = []
    for param in params:
        weight = strategy(param)
        weight_list.append(weight)

        # result = weight_share_target, weight_share_adj, pnl, pnl_adj
        result = get_backtest(weight, account, price, volume, plot=False)
        pnl_list.append(result[-1])
    
    # Walk-Forward Cross-Validation
    fitted_weight = []
    fitted_param = []
    test_index_all = []
    tscv = TimeSeriesSplit(test_size=test_size, n_splits=n_splits)
    for train_index, test_index in tscv.split(range(price.shape[0])):
        train_index = train_index[-test_size*k:]   # len(train_index) == len(test_index) * k
        test_index_all.extend(test_index)

        max_idx = 0
        max_sharpe = 0
        for i in range(len(list(params))):
            if max_sharpe < _get_sharpe(pnl_list[i].iloc[train_index]):
                max_sharpe = _get_sharpe(pnl_list[i].iloc[train_index])
                max_idx = i
        
        fitted_weight.append(weight_list[max_idx][test_index])
        fitted_param.append(max_idx)

    fitted_weight = np.vstack(fitted_weight)
    return fitted_weight, fitted_param, test_index_all


def strategy_pipeline(strategy, params, account, vwap, volume, test_size, overnight=False, plot=True, log=True, logpath=None) :
    fitted_weight, fitted_param, test_index = cross_validate_strategy(strategy, params, account, vwap, volume, test_size)
    
    # result = weight_share_target, weight_share_adj, pnl, pnl_adj
    result = get_backtest(fitted_weight, account, price=vwap.iloc[test_index], volume=volume.iloc[test_index], overnight=overnight, plot=plot)
    statistics = get_stats_intraday(result[1], result[3], vwap, account, plot=True)

    if log:
        # Save the result
        pd.DataFrame(result[1], index=result[3].index, columns=volume.columns).to_csv(f'{logpath}\\{strategy.__name__}_weight.csv')
        result[3].to_csv(f'{logpath}\\{strategy.__name__}_pnl.csv')
        statistics.to_csv(f'{logpath}\\{strategy.__name__}_statistics.csv')

    return result, statistics