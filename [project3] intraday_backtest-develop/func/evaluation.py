import numpy as np
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt


def get_plot(pnl, drawdown, account):
    # Create a figure with 2 subplots, one above the other
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot cumulative profit with primary y-axis on the first subplot
    ax2 = ax1.twinx()  # Create a twin y-axis for the first subplot
    ax1.plot(pnl.index, pnl.cumsum(), label='PNL', color='b')
    ax1.set_ylabel('Cumulative Profit (₩)')
    ax1.tick_params(axis='y')

    # Plot cumulative return percentage on the twin y-axis of the first subplot
    ax2.plot(pnl.index, pnl.cumsum()/account)
    ax2.set_ylabel('PNL (%)')
    ax2.tick_params(axis='y')

    # Add title for the first subplot
    ax1.set_title('PNL (₩ and %) Over Time')

    # Combine legends for the first subplot
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot return percentage on the second subplot
    ax3.bar(pnl.index, pnl/account, label='Return (%)', color='black')
    ax3.set_ylabel('Return (%)')
    ax3.set_xlabel('Date')
    ax3.tick_params(axis='y')

    # Create twin y-axis for the second subplot
    ax4 = ax3.twinx()
    ax4.plot(pnl.index, drawdown, label='Drawdown (%)', color='r')
    ax4.set_ylabel('Drawdown (%)')
    ax4.tick_params(axis='y')

    # Add title for the second subplot
    ax3.set_title('Return (%) Over Time')

    # Combine legends for the second subplot
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    # Rotate x-axis labels by 90 degrees for better readability
    plt.xticks(rotation=90)

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0.1)  # Increase the value to create more space

    # Display the combined plot
    plt.show()

def get_stats_intraday(weight, pnl, price, account, plot=True):
    weight = pd.DataFrame(weight, index=pnl.index, columns=range(1, 101))
    turnover = (weight.diff().fillna(0).abs() * price.loc[weight.index].values).sum(axis=1)/account
    daily_turnover = turnover.groupby(turnover.index.date).sum()
    daily_turnover.index = pd.to_datetime(daily_turnover.index)
    
    # Drawdown
    pnl_cum = pnl.cumsum()
    drawdown = (pnl_cum - pnl_cum.cummax())/account

    # Monthly Stats
    month_returns = pnl.groupby(pnl.index.month).sum()/account * 100
    month_sharpe = pnl.groupby(pnl.index.month).apply(lambda x: np.mean(x)/np.std(x) if np.std(x)!=0 else 0) * np.sqrt(252 * 330)
    npnl = pnl[pnl<0]
    month_sortino = pnl.groupby(pnl.index.month).mean() / npnl.groupby(npnl.index.month).std() * np.sqrt(252 * 330)
    month_drawdown = drawdown.groupby(pnl.index.month).min() * 100
    month_turover = daily_turnover.groupby(daily_turnover.index.month).mean()
    month_total_turnover = daily_turnover.groupby(daily_turnover.index.month).sum()
    month_fitness = month_sharpe * (abs(month_returns) /100 / month_turover.apply(lambda x: max(x, 0.125)))**0.5
    month_margin = month_returns / month_total_turnover * 100
    month_long_count = np.sum(weight.fillna(0)>0, axis=1).groupby(pnl.index.month).mean() 
    month_short_count = np.sum(weight.fillna(0)<0, axis=1).groupby(pnl.index.month).mean()

    # Monthly Stats Aggregation
    df_stats_month = pd.concat([month_returns, month_sharpe, month_sortino, month_drawdown, month_turover, month_fitness, month_margin, month_long_count, month_short_count], axis=1)
    df_stats_month.columns = ['returns(%)', 'sharpe', 'sortino', 'drawdown(%)', 'turnover', 'fitness', 'margin(‱)', 'long_count', 'short_count']
    df_stats_month.apply(lambda x: round(x, 3))

    # Total Stats
    total_return = pnl_cum.iloc[-1]/account * 100
    total_sharpe = np.mean(pnl)/np.std(pnl) * np.sqrt(252 * 330)
    total_sortino = pnl.mean()/npnl.std() * np.sqrt(252 * 330)
    total_drawdown = drawdown.min() * 100
    total_turnover = daily_turnover.mean()
    total_fitness = total_sharpe * (abs(total_return) /100 / max(total_turnover, 0.125))**0.5
    total_margin = total_return / daily_turnover.sum() * 100
    total_long_count = np.sum(weight.fillna(0)>0, axis=1).mean() 
    total_short_count = np.sum(weight.fillna(0)<0, axis=1).mean()
    df_stats_total = pd.DataFrame([[total_return, total_sharpe, total_sortino, total_drawdown, total_turnover, total_fitness, total_margin, total_long_count, total_short_count]], 
                                    columns=df_stats_month.columns, index=['Total'])
    if plot:
        daily_pnl = pnl.groupby(pnl.index.date).sum()
        daily_drawdown = drawdown.groupby(drawdown.index.date).max()
        get_plot(daily_pnl, daily_drawdown, account)    

    return pd.concat([df_stats_total, df_stats_month], axis=0)
