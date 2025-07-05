# Intraday Backtest for Simple Hypothesis
Programming project for testing simple trading hypothesis

## Codes & Development Environment Description

**Environment**: requirements.txt
* python 3.10.
* libraries: numpy, pandas, numba, scikit-learn, scipy, matplotlib

<br/> <br/> 

![flowchart](https://github.com/user-attachments/assets/b349e476-2db2-4fe8-8066-99577e827b76)

**Step 1) Merge & Nan-handling**

**read_files.py**
* *get_intra_data*: read intraday data and merge it to single dataframe
* *get_daily_data*: read daily data
* nan-handling: 
    * fill 0 for 'volume', 'traded_value', 'num_trades', 'hit_volume', 'hit_value', 'num_hits',  'lift_volume', 'lift_value', 'num_lifts'
* brief EDA
    * intraday data starts from 9:30 and ends 16:00
    * intraday has no data from 12:00 to 13:00
    * intraday has 330 time slots for each date
    * data is 182 days from 0401 to 1210
    * there are 60060 munites in the data
    * intraday close at 16:00 can be different with daily close
        * avoid overnight position for precise calculation


<br/> <br/> 

**Step 2) Alpha Ideation**
* Simple alpha ideas from the data list
* Define alpha function which generate raw weight from idea
<br/> <br/> 

**operators.py**
* functions in the 'operators.py' file can be utilized to express alpha in the functional form

* **Cross-Sectional Operators**
    ```python 
    normalize(x: np.array, useStd:bool=False, limit:float=0.0)
    ```
    * Calculates the mean value of all valid alpha values for a certain date, then subtracts that mean from each element. Optionally, it can also divide by the standard deviation.

    ```python 
    rank(x: np.array) -> np.array:
    ```
    
    * Ranks 2D array values row by row, and then scales values to be in [0, 1]. For example, [1, 2, 3, 4] becomes [0, 1/3, 2/3, 1].

    ```python 
    scale(weight):
    ```    
    * Scales the weights by dividing each element by the sum of the absolute values of its row.

    ```python 
    zscore(x: np.array) -> np.array
    ```
    * Calculates the z-score of each element in a 2D array row-wise, which standardizes the values by subtracting the mean and dividing by the standard deviation.

    ```python 
    quantile(x:np.array, q:int)
    ```
    * Ranks 2D array values row by row and then scales them into q quantiles.
<br/> <br/> 

* **Time-Series Operators**
    ```python 
    ts_returns(x, d)
    ```
    * Calculates the percentage change in values over a specified number of periods d.

    ```python 
    ts_delta(x, d)
    ```    
    * Calculates the difference between the current value and the value from d periods ago.

    ```python 
    ts_delay(x, d)
    ```    
    * Shifts the values by d periods.

    ```python  
    ts_max(x, d)
    ```    
    * Computes the rolling maximum over a window of d periods.

    ```python  
    ts_min(x, d)
    ```    
    * Computes the rolling minimum over a window of d periods.

    ```python  
    ts_argmax(x, d)
    ```
    * Finds the index of the maximum value over a rolling window of d periods and adds 1 to it.

    ```python 
    ts_argmin(x, d)
    ```    
    * Finds the index of the minimum value over a rolling window of d periods and adds 1 to it.

    ```python 
    ts_mean(x, d)
    ```  
    * Calculates the rolling mean over a window of d periods.

    ```python 
    ts_decay_linear(df, d)
    ```    
    * Computes a linear weighted moving average over d periods.

    ```python 
    ts_median(x, d)
    ```    
    * Calculates the rolling median over a window of d periods.

    ```python 
    ts_std(x, d)
    ```
    * Computes the rolling standard deviation over a window of d periods.

    ```python  
    ts_rank(x, d)
    ```    
    * Ranks values within a rolling window of d periods.

    ```python  
    ts_sum(x, d)
    ```    
    * Calculates the rolling sum over a window of d periods.

    ```python 
    ts_prod(x, d)
    ```    
    * Computes the rolling product over a window of d periods.

    ```python 
    ts_corr(x, y, d)
    ```    
    * Calculates the rolling correlation between two arrays x and y over a window of d periods.

    ```python 
    ts_cov(x, y, d)
    ```  
    * Calculates the rolling covariance between two arrays x and y over a window of d periods.

    ```python  
    ts_ir(x, d)
    ```
    * Computes the information ratio, which is the difference between the values and the rolling mean divided by the rolling standard deviation over d periods.

    ```python 
    ts_stairs(array, d)
    ```    
    * Applies a stair-step transformation to the array, where each segment of d rows is assigned the value of the first row in that segment.

* **Group Operators**

    ```python  
    group_rank(df1, df2)
    ```    
    * Ranks elements in df1 within the groups specified by df2.

    ```python 
    group_zscore(df1, df2)
    ```    
    * Calculates the z-score of elements in df1 within the groups specified by df2.

</br></br>

**Step 3) Alpha Transformation**

From steps 3 to 5, the pipeline of alpha generation will handle all processes efficiently. This process is done with 'backtest.py'.
<br/> <br/> 

**backtest.py**
* *get_neutral_weight*: normalize and scale the weight
    * **Long-Short** ) after **normalization**, sum of weights for each time is 0
    * **Scale** ) after **scaling**, sum of absolute value of weight ratios for each time is 1
    * By normalization, we neutralize the market risk
    * If we take normalization for each quantile group of 'traded_value', we can neutralize the liquidity risk, too

* *get_weight_share_from_ratio*: calculate share amount from weight ratio
    * positions are measured in the number of shares to be closer to the actual trading constraints

* *get_weight_share_adj*: filter weight share with volume data
    * if change of 'target weight share' > volume * feasible_volume_ratio, then we can only change position with 'volume * feasible_volume_ratio'
    * default feasible_volume_ratio is 0.01, which is reasonably small

<br/> <br/> 


**Step 4) Backtest wth Walk-Forward Cross-Validation**

![walk-forward_cross_validation](https://github.com/user-attachments/assets/6f6738fb-f875-40e2-8f88-37618ab4ecea)


* Figure from 'Traffic Analysis and Prediction in Urban Areas' DOI:10.13140/RG.2.2.17702.14400

**backtest.py**

* *cross_validate_strategy*: Walk-Forward Cross-Validation
    * find optimal parameters in each train sets and get target shares with that parameters in test sets
    * evauation criteria for cross-validation is the ***sharpe ratio***
* *get_backtest*:
    * given weight ratio, calculate target ratio (step 3)
* *strategy_pipeline*: 
    * input: alpha idea with function form
    * output: weight_share_target, weight_share_adj, pnl, pnl_adj 
        * 'adj' refers 'filtered by volume'
    * process *cross_validate_strategy* & *get_backtest* to get target shares in total test periods
    * log target shares and pnl in the *log folder*

<br/> <br/> 

**Step 5) Generate Target Position Files**

In step 4, '*strategy_pipeline*' function from *backtest.py* generates the target position in the *log folder* for each strategy.

<br/> <br/> 



## Model Description

### General Idea & Assumptions

* Target Share
    * At the end of time t, data is recorded by the information from time interval [t-1, t]
    * I calculate the target position for time t+1 with data at t
    * The target position is aimed to be achieved at time t+1, and position adjustment is performed between time intervals [t, t+1]
    
* Transaction
    * All strategies are simple interest, not compound
        * To avoid liquidity problem
        * Initial value of account is assumed to be $100,000 for each strategy
    * The change of target position at each time interval is restricted by **volume filtering**
        * if change of 'target weight share' > volume * feasible_volume_ratio, then we can only change position with 'volume * feasible_volume_ratio'
        * default feasible_volume_ratio is 0.01, which is reasonably small
    * Assume the average transaction price at each time interval is the same as vwap at each time interval
        * Default volume filtering ratio is 1% which is a reasonably small portion to avoid slippage

* Neutralization
    * To neutralize the market risk, all strategies are long-short strategies
    * The sum of long value and short value is aimed to be zero
        * Since we calculate the target share, not the target ratio, it can be nonzero
    
<br/> <br/> 

## Strategies

**strategies.ipynb**

* Strategy 1 to Strategy 13 is described in *strategies.ipynb*
* Results of strategies are logged in 'log' folder
    * strategy{i}_pnl: pnl of strategy i
    * strategy{i}_statistics: statistics table of strategy i
    * strategy{i}_weight: weight of strategy i
<br/> <br/> 

**Example) Strategy 1**

```python 
# Volatility Risk Premium over Liquidity
# Higher volatility over liquidity -> Long
def strategy1(param):
    x = (ts_mean((high-low)/(spread_twa), param))
    weight = bt.get_neutral_weight(quantile(x, 5), roll=1)
    weight[0] = 0
    return weight

params = [10, 20, 30]
result, statistics = bt.strategy_pipeline(strategy1, params, account, vwap, volume, test_size, overnight=False, logpath=logpath)
statistics
```

* The comment section at the beginning explains the underlying ideas
* The function definition part below is the alpha implementation part and converts the underlying idea into weight
* The function *bt.strategy_pipeline* performs walk-forward cross-validation and searches for optimal parameters in *params*
* The results are contained in results and statistics
    * results: target_share, target_share_adj, pnl, pnl_adj
    * statistics: total statistics and monthly statistics
        * returns(%): total return
        * sharpe: annualized sharpe	
        * sortino: annualized sortino		
        * drawdown(%): drawdown	
        * turnover:	daily turnover
        * fitness: sharpe * return / sqrt(max(turnover, 0.125))
        * margin(‱): return / turnover
        * long_count/short_count: average number of stocks in long/short position
<br/> <br/> 

## Further Works

* Current strategies have high turnover values
    1) Conditional trade with longer frequency data
    * Take the intraday long/short position only when the gap between the open price and the previous day's close price is negative/positive
    * Take the intraday long/short position only when daily price momentum is upward/downward

    2) Combine with longer frequency singals
    * Monday return reversal: Monday afternoon returns shows reversal over remaining week
    * Resample one minute data to longer period data (ex. 5 min or 1 hour)
    
    3) Noise filtering
    * Volume filtering enhances the pnl, and it means noisy transactions of illiquid stocks deteriorate the pnl
    * Liquidity neutralizaiton
        * Normalization -> Group Normalization (with liquidity grouping)
        * Give more weights on liquid group of stocks

* Alpha Merge
    * Alpha signals with low or negative correlation can enhance pnl of the total portfolio
    * The cancelation of target weights (or target share) of different strategies can lower turnover, and lower position opened (leverage effect without cost)
