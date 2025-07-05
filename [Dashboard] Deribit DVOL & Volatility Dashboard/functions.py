from api_functions import get_volatility_index_data, get_book_summary_by_currency
from datetime import datetime
import pandas as pd
import numpy as np
import pytz
import math
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
from scipy.special import ndtr

# --- Chart helper functions ---
def _make_line_chart(df: pd.DataFrame, x_col: str, y_cols: list[str], height: int = 400, template: str = 'plotly_dark') -> go.Figure:
    fig = px.line(df, x=x_col, y=y_cols, height=height, template=template)
    fig.update_layout(hovermode='x unified')
    for trace in fig.data:
        trace.hovertemplate = '%{y:.4f}<extra></extra>'
    return fig


def _apply_visibility(fig: go.Figure, time_frames: list[int], show_frames: list[int]) -> None:
    for i, frame in enumerate(time_frames):
        if frame not in show_frames and i < len(fig.data):
            fig.data[i].visible = 'legendonly'


def _make_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, xaxis_title: str, yaxis_title: str, height: int = 400, template: str = 'plotly_dark') -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x_col], y=df[y_col], mode='lines'))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=template,
        height=height
    )
    return fig

# --- Historical Volatility Charts ---
def hv_charts(currency_data: pd.DataFrame, time_frames: list[int]):
    df_hv = currency_data.reset_index().rename(columns={'index': 'Date'})
    # Close-to-close
    y_close = [f'{d}_day_close_vol' for d in time_frames]
    fig_close = _make_line_chart(df_hv, 'Date', y_close)
    # Parkinson
    y_park = [f'{d}_day_park_vol' for d in time_frames]
    fig_park = _make_line_chart(df_hv, 'Date', y_park)
    # Ratio
    y_ratio = [f'{d}_day_park_close_ratio' for d in time_frames]
    fig_close_park_ratio = _make_line_chart(df_hv, 'Date', y_ratio)
    # Visibility
    default_show = [7, 30, 365]
    _apply_visibility(fig_close, time_frames, default_show)
    _apply_visibility(fig_park, time_frames, default_show)
    _apply_visibility(fig_close_park_ratio, time_frames, default_show)
    # Volatility cones
    df_clean = currency_data.dropna()
    percentiles = [10, 50, 90]
    pct_vals = {w:{p:np.percentile(df_clean[f'{w}_day_park_vol'],p) for p in percentiles} for w in time_frames}
    fig_vol_cones = go.Figure()
    for p in percentiles:
        fig_vol_cones.add_trace(go.Scatter(
            x=time_frames, y=[pct_vals[w][p] for w in time_frames],
            mode='lines+markers', name=f'{p}th percentile',
            line=dict(color={10:'MediumPurple',50:'MediumSeaGreen',90:'MediumPurple'}[p])
        ))
    min_max = {w:{'min':df_clean[f'{w}_day_park_vol'].min(),'max':df_clean[f'{w}_day_park_vol'].max()} for w in time_frames}
    fig_vol_cones.add_trace(go.Scatter(x=time_frames, y=[min_max[w]['min'] for w in time_frames], mode='lines+markers', name='Min', line=dict(color='crimson')))
    fig_vol_cones.add_trace(go.Scatter(x=time_frames, y=[min_max[w]['max'] for w in time_frames], mode='lines+markers', name='Max', line=dict(color='crimson')))
    fig_vol_cones.update_layout(xaxis_title='Window Length (days)', yaxis_title='Volatility', template='plotly_dark', hovermode='x unified')
    return fig_close, fig_park, fig_close_park_ratio, fig_vol_cones

# --- DVOL data and charts ---
def dvol_charts(currency: str, start_timestamp: int, end_timestamp: int, dvol_resolution: int):
    raw = get_volatility_index_data(currency, start_timestamp, end_timestamp, dvol_resolution)
    columns = ['timestamp','open','high','low','close']
    df = pd.DataFrame(raw['data'], columns=columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    current_vol = df.iloc[-1]['close']
    year_min = df['low'].min()
    year_max = df['high'].max()
    iv_rank = (current_vol - year_min)/(year_max - year_min)*100
    iv_percentile = len(df[df['close'] <= current_vol]) / len(df) * 100
    candles = go.Figure(data=[
        go.Candlestick(
            x=df['date'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        )
    ])
    candles.update_layout(
        height=400,
        template='plotly_dark',
        title=f'{currency} DVOL    High: {year_max}, Low: {year_min}, Current: {current_vol}',
        shapes=[
            dict(
                type='line', yref='y', y0=current_vol, y1=current_vol,
                xref='x', x0=df['date'].min(), x1=df['date'].max(),
                line=dict(color='magenta', width=1, dash='dot')
            )
        ]
    )
    return df, current_vol, iv_rank, iv_percentile, candles

# --- Combined DVOL fetch ---
def get_dvol_data():
    now = datetime.now()
    end_ts = round(now.timestamp() * 1000)
    year_ms = 1000 * 60 * 60 * 24 * 365
    start_ts = end_ts - year_ms
    dvol_resolution = 3600 * 24

    df_btc, btc_current_vol, btc_iv_rank, btc_iv_percentile, btc_candles = dvol_charts(
        'BTC', start_ts, end_ts, dvol_resolution
    )
    df_eth, eth_current_vol, eth_iv_rank, eth_iv_percentile, eth_candles = dvol_charts(
        'ETH', start_ts, end_ts, dvol_resolution
    )

    # BTC/ETH DVOL ratio
    df_eth['ratio'] = df_btc['close'] / df_eth['close']
    ratio = _make_scatter_chart(
        df_eth, 'date', 'ratio',
        title='BTC/ETH DVOL Ratio',
        xaxis_title='Date', yaxis_title='Ratio'
    )

    return (
        btc_candles,
        btc_iv_rank,
        btc_iv_percentile,
        eth_candles,
        eth_iv_rank,
        eth_iv_percentile,
        ratio
    )

def calculate_time_difference(date_string):
    now = datetime.now(pytz.utc)
    date = datetime.strptime(date_string, "%d%b%y")
    date = date.replace(tzinfo=pytz.utc)
    target_time = date.replace(hour=8, minute=0, second=0)
    time_difference = (target_time - now).total_seconds()
    seconds_in_a_year = 365 * 24 * 60 * 60
    time_difference_years = time_difference / seconds_in_a_year
    return time_difference_years

def bs_price(S, K, T, R, sigma, option_type):
    d1 = (np.log(S / K) + (R + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        price = S * ndtr(d1) - K * np.exp(-R*T) * ndtr(d2)
    elif option_type == "P":
        price = K * np.exp(-R*T) * ndtr(-d2) - S * ndtr(-d1)
    return price

def bs_delta(S, K, T, R, sigma, option_type):
    d1 = (np.log(S / K) + (R + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    if option_type == "C":
        delta = ndtr(d1)
    elif option_type == "P":
        delta = ndtr(d1) - 1
    return delta

def calculate_implied_volatility(option_price, S, K, T, R, option_type):
    MAX_ITERATIONS = 100
    PRECISION = 0.0001
    sigma_low = 0.01
    sigma_high = 5
    implied_volatility = None

    for i in range(MAX_ITERATIONS):
        sigma = (sigma_low + sigma_high) / 2.0
        price = bs_price(S, K, T, R, sigma, option_type)
        diff = option_price - price

        if abs(diff) < PRECISION:
            implied_volatility = sigma
            break

        if diff > 0:
            sigma_low = sigma
        else:
            sigma_high = sigma

    return implied_volatility

def find_closest_strike(row):
    # finds the strike closest to the underlying price (ATM)
    idx = np.abs(row['strike'] - row['underlying_price']).idxmin()
    return row.loc[idx, ['strike', 'expiry_date', 'implied_volatility']]

def vol_term_structure(currency):
    data = get_book_summary_by_currency(currency, 'option')
    df = pd.DataFrame(data)
    df = df[['underlying_price', 'mark_price', 'instrument_name']]
    df[['currency', 'expiry', 'strike', 'type']] = df['instrument_name'].str.split('-', expand=True)
    df = df.drop(df[df['type'] == 'P'].index)
    df['usd_price'] = df['underlying_price'] * df['mark_price']
    df['strike'] = df['strike'].astype(float)
    # Drop rows where underlying_price is more than 15% away from strike
    df = df.drop(df[abs(df['underlying_price'] - df['strike']) / df['strike'] > 0.15].index)
    df['time_to_expiry'] = df['expiry'].apply(lambda x: calculate_time_difference(x))
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    # Apply the calculate_implied_volatility function to create the 'implied_volatility' column
    df['implied_volatility'] = df.apply(lambda row: calculate_implied_volatility(
        row['usd_price'],
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['type']
    ), axis=1)
    # Group the DataFrame by 'expiry'
    grouped = df.groupby('expiry_date')
    # Apply the 'find_closest_strike' function to each group and collect the results
    df_term_structure = grouped.apply(find_closest_strike)
    # Reset the index and drop the original index column
    df_term_structure = df_term_structure.reset_index(drop=True)

    # Create a line chart using Plotly
    fig = go.Figure(data=go.Scatter(
        x=df_term_structure['expiry_date'],
        y=df_term_structure['implied_volatility'],
        mode='lines',
        # line_shape='spline'  # enable this for a smoothed line
    ))
    # Generate vertical lines for each expiry_date
    shapes = []
    for expiry_date in df_term_structure['expiry_date']:
        shapes.append(
            dict(
                type='line',
                xref='x', x0=expiry_date, x1=expiry_date,
                yref='y', y0=df_term_structure['implied_volatility'].min(),
                y1=df_term_structure['implied_volatility'].max(),
                line=dict(
                    color='rgba(255, 0, 255, 0.5)',
                    width=1,
                    dash='dash',
                )
            )
        )
    # Customize the layout
    fig.update_layout(
        title=f'{currency} Implied Volatility Term Structure',
        xaxis=dict(title='Expiry Date'),
        yaxis=dict(title='Implied Volatility'),
        template='plotly_dark',
        shapes=shapes,
        height=400,
    )

    return fig

def vol_surface(currency):
    data = get_book_summary_by_currency(currency, 'option')
    df = pd.DataFrame(data)
    df = df[['underlying_price', 'mark_price', 'instrument_name']]
    df[['currency', 'expiry', 'strike', 'type']] = df['instrument_name'].str.split('-', expand=True)
    df = df.drop(df[df['type'] == 'P'].index)
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    df['usd_price'] = df['underlying_price'] * df['mark_price']
    df['strike'] = df['strike'].astype(float)
    df['time_to_expiry'] = df['expiry'].apply(lambda x: calculate_time_difference(x))
    df['expiry_date'] = pd.to_datetime(df['expiry'], format='%d%b%y')
    # Apply the calculate_implied_volatility function to create the 'implied_volatility' column
    df['implied_volatility'] = df.apply(lambda row: calculate_implied_volatility(
        row['usd_price'],
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['type']
    ), axis=1)
    df['delta'] = df.apply(lambda row: bs_delta(
        row['underlying_price'],
        row['strike'],
        row['time_to_expiry'],
        0,
        row['implied_volatility'],
        row['type']
    ), axis=1)

    # drop extremes of delta
    df = df[df['delta'] >= 0.01]
    df = df[df['delta'] <= 0.99]

    fig = go.Figure(data=go.Scatter3d(
        x=df['expiry_date'],
        y=df['delta'],
        z=df['implied_volatility'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['implied_volatility'],  # Color code based on 'implied_volatility' values
            colorscale='Sunset_r',  # Choose a colorscale
            opacity=0.8
        ),
        hovertemplate=
        '<b>Expiry Date:</b>: %{x}' +
        '<br><b>Delta:</b>: %{y}' +
        '<br><b>Implied Volatility:</b>: %{z}<br>' +
        '<extra></extra>', # removes the secondary box
    ))
    # round the maximum vol for the z axis to the nearest 0.2
    max_vol = df['implied_volatility'].max()
    rounded_max_vol = math.ceil(max_vol / 0.2) * 0.2
    fig.update_layout(
        title=f'{currency} 3D Volatility Surface',
        scene=dict(
            xaxis_title='Expiry Date',
            yaxis_title='Delta',
            zaxis_title='Implied Volatility',
            zaxis=dict(range=[0, rounded_max_vol], dtick=0.2)
        ),
        template='plotly_dark',
        height=800,
    )

    return fig

def draw_indicator(color, minimum, maximum, title, value, width, height):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [minimum, maximum], 'dtick': 20},
               'bar': {'color': color}}
    ))
    fig.update_layout(
        template='plotly_dark',
        autosize=False,
        margin=dict(
            l=30,  # left margin
            r=40,  # right margin
            b=0,  # bottom margin
            t=20,  # top margin
            pad=0  # padding
        ),
        paper_bgcolor="rgba(0,0,0,0)",  # makes the background transparent
        height=height,
        width=width,
    )
    return fig

def chart_card(title, chart, info_text):
    card = dbc.Card(
        children=[
            dbc.CardHeader(title, style={'padding-left': '50px'}),
            dbc.CardBody(chart, style={'padding': '0px'}),
            dbc.Badge(
                html.B("i"),
                color="primary",
                id=f'{chart.id}_info',
                pill=True,
                style={"position": "absolute", "top": "10px", "left": "20px", "zIndex": 2}
            ),
            dbc.Tooltip(
                info_text,
                target=f'{chart.id}_info',
            ),
        ],
        style={
            'width': '49%',
            'display': 'inline-block',
            'min-width': '600px',
            'margin': '2px',
        }
    )
    return card
