import os
import pandas as pd 
from itertools import chain


def get_intra_data_path(data, path=None, start=None, end=None):
    if path is None:
        # path = os.path.dirname(os.getcwd())+ '/sample_data/'
        path = os.getcwd()+ '/sample_data/'
    all_files = os.listdir(path)
    all_files = [file for file in all_files if file.split('.')[0][:-5] == data]
    if start is not None:
        all_files = [file for file in all_files if int(start) <= int(file.split('.')[0][-4:])]
    if end is not None:
        all_files = [file for file in all_files if int(file.split('.')[0][-4:]) <= int(end)]

    dates = [file.split('.')[0].split('_')[-1] for file in all_files]
    return all_files, dates

def _fillna_intra(df, data:str):
    fill_zero = ['volume', 'traded_value', 'traded_value', 'num_trades', 'hit_volume', 'hit_value', 'num_hits',  'lift_volume', 'lift_value', 'num_lifts']
    if data in fill_zero:
        df.fillna(0, inplace=True)
    
    return df

def get_intra_data(data, path=None, start=None, end=None):
    path = os.getcwd()+ '/sample_data/'
    all_files, dates = get_intra_data_path(data, path, start, end)
    df = pd.concat([pd.read_csv(path + file) for file in all_files])
    df['Dates'] = list(chain.from_iterable([[date] * 330 for date in dates]))
    df['Dates'] = pd.to_datetime(df['Dates'] + ' ' + df['Minutes'], format='%m%d %H:%M:%S')
    df.drop(columns=['Minutes'], inplace=True)
    df.set_index('Dates', inplace=True)

    return _fillna_intra(df, data)

def get_daily_data(data='daily_close_price'):
    path = os.path.dirname(os.getcwd())+ '/sample_data/'
    if data == 'daily_close_price':
        df = pd.read_csv(path + 'daily_close_price.csv')
    if data == 'daily_close_volume':
        df = pd.read_csv(path + 'daily_close_volume.csv')
    if data == 'daily_open_price':
        df = pd.read_csv(path + 'daily_close_price.csv')
    if data == 'daily_open_volume':
        df = pd.read_csv(path + 'daily_close_volume.csv')
    df.set_index('Dates', inplace=True)
    return df