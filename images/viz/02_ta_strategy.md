
1. [Dual Moving Average Strategy](#Dual-Moving-Average-Strategy)

1. [Pyfolio Backtesting](#Pyfolio-Backtesting)

## Import libraries


```python
import pandas as pd
import pyfolio as pf
from joblib import load
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
```


```python
import zipline
%load_ext zipline
```

# Dual Moving Average Strategy
<a id = "Dual Moving Average Strategy"></a>

## Baseline Strategy

There are two common strategies in finance: <i>momentum strategy</i>, usually referred to as trend trading and, the opposite, <i>reversion strategy</i>, frequently known as convergence or cycle trading. In this project, we will focuse on momentum strategies, whereby we will exploit upward/downward trends of our stock.<br> 

More concretely, we will perform a **dual moving average crossover**, which occurs when a short-term average crosses a long-term average. This signal is used to identify that momentum is shifting in the direction of the short-term average. A buy signal is generated when the short-term average crosses the long-term average and rises above it, while a sell signal is triggered by a short-term average crossing long-term average and falling below it.<br>

We customise one of [zipline examples](#https://github.com/quantopian/zipline/tree/master/zipline/examples) next, with the following specifications:
<ul>
    <li>the series'unit has been left as prices, more information on <code>03_ml_strategy.ipynb</code></li>
    <li>initial capital base of 10.000 USD</li>
    <li>trading period of 635 days (corresponding to the last ML model's test cross-validation sample <code>03_ml_strategy.ipynb</code>, from 2015-5-13 to 2018-3-15</li>
    <li>moving average windows also correspond to ML model's lagged and predicted values <code>03_ml_strategy.ipynb</code>, so as to facilitate comparison, if any</li>
    <li>in line with the aforementioned dual moving average crossover, if the fast (short-term) moving average crosses up the slow (long-term) moving average, we trade 100 shares and viceversa</li>
    <li>no commissions or slippage have been defined</li>
</ul>


```python
%%zipline --start 2015-5-13 --end 2018-3-15 --capital-base 10000.0 -o ../strategies/models/dma_strategy.joblib --no-benchmark -b quandl

from zipline.api import order, record, symbol

# parameters 
stock = 'AAPL'
slow_ma_periods = 32
fast_ma_periods = 8

def initialize(context):
    context.time = 0
    context.asset = symbol(stock)
    context.has_position = False
    
def handle_data(context, data):
    context.time += 1
    if context.time < slow_ma_periods:
        return

    fast_ma = data.history(context.asset, 'price', bar_count=fast_ma_periods, frequency="1d").mean()
    slow_ma = data.history(context.asset, 'price', bar_count=slow_ma_periods, frequency="1d").mean()

    # Trading logic
    if (fast_ma > slow_ma) & (not context.has_position):
        order(context.asset, 100)
        context.has_position = True
    elif (fast_ma < slow_ma) & (context.has_position):
        order(context.asset, -100)
        context.has_position = False

    record(price=data.current(context.asset, 'price'),
           fast_ma=fast_ma,
           slow_ma=slow_ma)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>algo_volatility</th>
      <th>algorithm_period_return</th>
      <th>alpha</th>
      <th>benchmark_period_return</th>
      <th>benchmark_volatility</th>
      <th>beta</th>
      <th>capital_used</th>
      <th>ending_cash</th>
      <th>ending_exposure</th>
      <th>ending_value</th>
      <th>...</th>
      <th>short_value</th>
      <th>shorts_count</th>
      <th>slow_ma</th>
      <th>sortino</th>
      <th>starting_cash</th>
      <th>starting_exposure</th>
      <th>starting_value</th>
      <th>trading_days</th>
      <th>transactions</th>
      <th>treasury_period_return</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-05-13 20:00:00+00:00</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-14 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-15 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-18 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-19 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-20 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-21 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-22 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-26 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-27 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-28 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-05-29 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-01 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-02 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>14</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-03 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>15</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-04 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-05 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-08 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-09 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>19</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-10 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-11 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>21</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-12 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>22</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-15 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-16 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>24</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-17 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-18 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-19 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-22 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>28</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-23 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2015-06-24 20:00:00+00:00</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10000.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-02-01 21:00:00+00:00</th>
      <td>0.171110</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>173.517813</td>
      <td>1.109489</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>687</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-02 21:00:00+00:00</th>
      <td>0.170985</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>173.095938</td>
      <td>1.108682</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>688</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-05 21:00:00+00:00</th>
      <td>0.170861</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>172.504375</td>
      <td>1.107877</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-06 21:00:00+00:00</th>
      <td>0.170738</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>172.144688</td>
      <td>1.107074</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>690</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-07 21:00:00+00:00</th>
      <td>0.170614</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>171.681875</td>
      <td>1.106273</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>691</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-08 21:00:00+00:00</th>
      <td>0.170491</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>171.066563</td>
      <td>1.105473</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>692</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-09 21:00:00+00:00</th>
      <td>0.170368</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.471563</td>
      <td>1.104675</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>693</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-12 21:00:00+00:00</th>
      <td>0.170245</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.225937</td>
      <td>1.103879</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>694</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-13 21:00:00+00:00</th>
      <td>0.170123</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.030312</td>
      <td>1.103085</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>695</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-14 21:00:00+00:00</th>
      <td>0.170000</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.914375</td>
      <td>1.102292</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>696</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-15 21:00:00+00:00</th>
      <td>0.169879</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.031875</td>
      <td>1.101501</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>697</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-16 21:00:00+00:00</th>
      <td>0.169757</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.037187</td>
      <td>1.100712</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>698</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-20 21:00:00+00:00</th>
      <td>0.169636</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.025313</td>
      <td>1.099924</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>699</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-21 21:00:00+00:00</th>
      <td>0.169514</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.964063</td>
      <td>1.099138</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>700</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-22 21:00:00+00:00</th>
      <td>0.169393</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.889062</td>
      <td>1.098354</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>701</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-23 21:00:00+00:00</th>
      <td>0.169273</td>
      <td>0.345139</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.926719</td>
      <td>1.097571</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>702</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-26 21:00:00+00:00</th>
      <td>0.169154</td>
      <td>0.344234</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>-17906.0485</td>
      <td>-4454.6595</td>
      <td>17897.0</td>
      <td>17897.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.071719</td>
      <td>1.094589</td>
      <td>13451.3890</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>703</td>
      <td>[{'order_id': '29cc2adb921b44bf9853ffe0ce4a4bb...</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-27 21:00:00+00:00</th>
      <td>0.169058</td>
      <td>0.338434</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17839.0</td>
      <td>17839.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.199844</td>
      <td>1.079448</td>
      <td>-4454.6595</td>
      <td>17897.0</td>
      <td>17897.0</td>
      <td>704</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-02-28 21:00:00+00:00</th>
      <td>0.168944</td>
      <td>0.335734</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17812.0</td>
      <td>17812.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.288594</td>
      <td>1.072048</td>
      <td>-4454.6595</td>
      <td>17839.0</td>
      <td>17839.0</td>
      <td>705</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-01 21:00:00+00:00</th>
      <td>0.169423</td>
      <td>0.304534</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17500.0</td>
      <td>17500.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.223281</td>
      <td>0.987325</td>
      <td>-4454.6595</td>
      <td>17812.0</td>
      <td>17812.0</td>
      <td>706</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-02 21:00:00+00:00</th>
      <td>0.169385</td>
      <td>0.316634</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17621.0</td>
      <td>17621.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.223906</td>
      <td>1.016546</td>
      <td>-4454.6595</td>
      <td>17500.0</td>
      <td>17500.0</td>
      <td>707</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-05 21:00:00+00:00</th>
      <td>0.169284</td>
      <td>0.322734</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17682.0</td>
      <td>17682.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.152656</td>
      <td>1.030761</td>
      <td>-4454.6595</td>
      <td>17621.0</td>
      <td>17621.0</td>
      <td>708</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-06 21:00:00+00:00</th>
      <td>0.169167</td>
      <td>0.321234</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17667.0</td>
      <td>17667.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.071719</td>
      <td>1.026362</td>
      <td>-4454.6595</td>
      <td>17682.0</td>
      <td>17682.0</td>
      <td>709</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-07 21:00:00+00:00</th>
      <td>0.169221</td>
      <td>0.304834</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17503.0</td>
      <td>17503.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.964531</td>
      <td>0.983476</td>
      <td>-4454.6595</td>
      <td>17667.0</td>
      <td>17667.0</td>
      <td>710</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-08 21:00:00+00:00</th>
      <td>0.169314</td>
      <td>0.323934</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17694.0</td>
      <td>17694.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>169.962656</td>
      <td>1.029762</td>
      <td>-4454.6595</td>
      <td>17503.0</td>
      <td>17503.0</td>
      <td>711</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-09 21:00:00+00:00</th>
      <td>0.169724</td>
      <td>0.354334</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17998.0</td>
      <td>17998.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.054531</td>
      <td>1.102679</td>
      <td>-4454.6595</td>
      <td>17694.0</td>
      <td>17694.0</td>
      <td>712</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-12 20:00:00+00:00</th>
      <td>0.169764</td>
      <td>0.371734</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>18172.0</td>
      <td>18172.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.288906</td>
      <td>1.143080</td>
      <td>-4454.6595</td>
      <td>17998.0</td>
      <td>17998.0</td>
      <td>713</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-13 20:00:00+00:00</th>
      <td>0.169827</td>
      <td>0.354234</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17997.0</td>
      <td>17997.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.565781</td>
      <td>1.098827</td>
      <td>-4454.6595</td>
      <td>18172.0</td>
      <td>18172.0</td>
      <td>714</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-14 20:00:00+00:00</th>
      <td>0.169852</td>
      <td>0.338934</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17844.0</td>
      <td>17844.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>170.782344</td>
      <td>1.060031</td>
      <td>-4454.6595</td>
      <td>17997.0</td>
      <td>17997.0</td>
      <td>715</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2018-03-15 20:00:00+00:00</th>
      <td>0.169735</td>
      <td>0.341034</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>0.0000</td>
      <td>-4454.6595</td>
      <td>17865.0</td>
      <td>17865.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0</td>
      <td>171.116406</td>
      <td>1.064285</td>
      <td>-4454.6595</td>
      <td>17844.0</td>
      <td>17844.0</td>
      <td>716</td>
      <td>[]</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>716 rows × 40 columns</p>
</div>



We then load our strategy for further analysis and we plot
<ol>
    <li>stock price and two moving averages</li>
    <li>portfolio's value</li>
</ol>
For an interactive plot with altair, please refer to <code>04_trading_viz.ipynb</code>


```python
dma = pd.read_csv('../reports/dma_strategy.csv', parse_dates=True)
```


```python
# Initialize the plot figure
fig = plt.figure(figsize=(16, 8))      

# Subplot for axes 2
ax1 = fig.add_subplot(111, ylabel='Price in $')

# Plot the price and two moving averages
dma['price'].plot(ax=ax1, color='r', lw=2.)
dma[['fast_ma', 'slow_ma']].plot(ax=ax1, lw=2.)

# Plot the buy and sell signals
ax1.plot(dma.loc[dma.positions == 1.0].index, 
         dma['fast_ma'][dma['positions'] == 1.0],
         '^', markersize=10, color='m')
ax1.plot(dma.loc[dma.positions == -1.0].index, 
         dma['fast_ma'][dma['positions'] == -1.0],
         'v', markersize=10, color='k')

plt.show()
```


![png](02_ta_strategy_files/02_ta_strategy_10_0.png)



```python
# Initialize the plot figure
fig = plt.figure(figsize=(16, 8))      

# Subplot for axes 1
ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')

# Plot the equity curve in dollars
dma['portfolio_value'].plot(ax=ax1, lw=2.)

ax1.plot(dma.loc[dma.positions == 1.0].index, 
         dma['portfolio_value'][dma.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(dma.loc[dma.positions == -1.0].index, 
         dma['portfolio_value'][dma.positions == -1.0],
         'v', markersize=10, color='k')

# Show the plot
plt.show()
```


![png](02_ta_strategy_files/02_ta_strategy_11_0.png)


And finally, we retrieve our portfolio value on our last trading day and the sum of capital used during the trading period.


```python
dma['portfolio_value'].iloc[-1]
```




    13410.340499999995




```python
dma.capital_used.sum()
```




    -14454.659499999987



# Pyfolio Backtesting
<div style = "float:right"><a style="text-decoration:none" href = "#inicio">Inicio</a></div>

Pyfolio is a common tool for trading strategy backtesting that is integrated in Zipline local research environment.<br>
However, we encountered some incompatibilities with the empyrical library (the financial statistical library, also integrated in Zipline environment) if returns's tear sheet were directly run in the code, that were only partially solved:
<oll>
    <li>Use <code>idxmin()</code> instead of <code>argmin()</code>
        [link](#https://github.com/quantopian/pyfolio/issues/601) </li>
    <li>fix bug <code>np.log1p()</code>
        [link](#https://stackoverflow.com/questions/57339209/numpy-runtimewarning-invalid-value-encountered-in-log1p) </li>
</ol>
The first issue was satisfactorily fixed but not the latter, since we are not able to upgrade numpy package under the Zipline environment. <br>

Nonetheless, we were finally able to conduct these analysis from the .joblib format file.


```python
dma = load('../strategies/models/dma_strategy.joblib')
```


```python
dma.dtypes
```




    algo_volatility                        float64
    algorithm_period_return                float64
    alpha                                   object
    benchmark_period_return                float64
    benchmark_volatility                   float64
    beta                                    object
    capital_used                           float64
    ending_cash                            float64
    ending_exposure                        float64
    ending_value                           float64
    excess_return                          float64
    fast_ma                                float64
    gross_leverage                         float64
    long_exposure                          float64
    long_value                             float64
    longs_count                              int64
    max_drawdown                           float64
    max_leverage                           float64
    net_leverage                           float64
    orders                                  object
    period_close               datetime64[ns, UTC]
    period_label                            object
    period_open                datetime64[ns, UTC]
    pnl                                    float64
    portfolio_value                        float64
    positions                               object
    price                                  float64
    returns                                float64
    sharpe                                 float64
    short_exposure                         float64
    short_value                            float64
    shorts_count                             int64
    slow_ma                                float64
    sortino                                float64
    starting_cash                          float64
    starting_exposure                      float64
    starting_value                         float64
    trading_days                             int64
    transactions                            object
    treasury_period_return                 float64
    dtype: object



## Pyfolio's full tear sheet:


```python
returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(dma)
```


```python
pf.create_full_tear_sheet(returns)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;"><th>Start date</th><td colspan=2>2015-05-13</td></tr>
    <tr style="text-align: right;"><th>End date</th><td colspan=2>2018-03-15</td></tr>
    <tr style="text-align: right;"><th>Total months</th><td colspan=2>34</td></tr>
    <tr style="text-align: right;">
      <th></th>
      <th>Backtest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Annual return</th>
      <td>10.9%</td>
    </tr>
    <tr>
      <th>Cumulative returns</th>
      <td>34.1%</td>
    </tr>
    <tr>
      <th>Annual volatility</th>
      <td>17.0%</td>
    </tr>
    <tr>
      <th>Sharpe ratio</th>
      <td>0.69</td>
    </tr>
    <tr>
      <th>Calmar ratio</th>
      <td>0.61</td>
    </tr>
    <tr>
      <th>Stability</th>
      <td>0.75</td>
    </tr>
    <tr>
      <th>Max drawdown</th>
      <td>-17.8%</td>
    </tr>
    <tr>
      <th>Omega ratio</th>
      <td>1.18</td>
    </tr>
    <tr>
      <th>Sortino ratio</th>
      <td>1.06</td>
    </tr>
    <tr>
      <th>Skew</th>
      <td>0.67</td>
    </tr>
    <tr>
      <th>Kurtosis</th>
      <td>10.16</td>
    </tr>
    <tr>
      <th>Tail ratio</th>
      <td>1.10</td>
    </tr>
    <tr>
      <th>Daily value at risk</th>
      <td>-2.1%</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Worst drawdown periods</th>
      <th>Net drawdown in %</th>
      <th>Peak date</th>
      <th>Valley date</th>
      <th>Recovery date</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.81</td>
      <td>2016-04-14</td>
      <td>2016-07-26</td>
      <td>2017-02-06</td>
      <td>213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.97</td>
      <td>2017-11-09</td>
      <td>2018-03-01</td>
      <td>NaT</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.28</td>
      <td>2017-05-12</td>
      <td>2017-07-31</td>
      <td>2017-08-30</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.08</td>
      <td>2015-11-03</td>
      <td>2015-11-13</td>
      <td>2016-03-03</td>
      <td>88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.60</td>
      <td>2017-09-01</td>
      <td>2017-10-19</td>
      <td>2017-10-31</td>
      <td>43</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Stress Events</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fall2015</th>
      <td>-0.13%</td>
      <td>-3.54%</td>
      <td>1.82%</td>
    </tr>
    <tr>
      <th>New Normal</th>
      <td>0.05%</td>
      <td>-6.30%</td>
      <td>7.13%</td>
    </tr>
  </tbody>
</table>



![png](02_ta_strategy_files/02_ta_strategy_21_3.png)



![png](02_ta_strategy_files/02_ta_strategy_21_4.png)


## Individual Pyfolio's plots

We will customise the plots for more in-depth analysis (see [Quantopian's lecture](#https://www.quantopian.com/lectures/portfolio-analysis) for more information). As a reminder, it is adviced in Zipline's version 1.4 to set benchmark to false, and subsequently, we have no benchmark data.


```python
[f for f in dir(pf.plotting) if 'plot_' in f]
```




    ['plot_annual_returns',
     'plot_capacity_sweep',
     'plot_cones',
     'plot_daily_turnover_hist',
     'plot_daily_volume',
     'plot_drawdown_periods',
     'plot_drawdown_underwater',
     'plot_exposures',
     'plot_gross_leverage',
     'plot_holdings',
     'plot_long_short_holdings',
     'plot_max_median_position_concentration',
     'plot_monthly_returns_dist',
     'plot_monthly_returns_heatmap',
     'plot_monthly_returns_timeseries',
     'plot_perf_stats',
     'plot_prob_profit_trade',
     'plot_return_quantiles',
     'plot_returns',
     'plot_rolling_beta',
     'plot_rolling_returns',
     'plot_rolling_sharpe',
     'plot_rolling_volatility',
     'plot_round_trip_lifetimes',
     'plot_sector_allocations',
     'plot_slippage_sensitivity',
     'plot_slippage_sweep',
     'plot_turnover',
     'plot_txn_time_hist',
     'show_and_plot_top_positions']



### Cumulative returns

Cumulative returns of 34,1% over the whole period.


```python
# Cumulative Returns
sns.set_style("white")
plt.subplot(2,1,1)
pf.plotting.plot_rolling_returns(returns)

# Daily, Non-Cumulative Returns
plt.subplot(2,1,2)
pf.plotting.plot_returns(returns)
plt.tight_layout()
```


![png](02_ta_strategy_files/02_ta_strategy_27_0.png)


### Distribution of returns

The graphs on distribution of returns gauge how the algorithm performs on a yearly and monthly basis. It particularly performs well in 2017, the first quarter of the year and in June, August and October (above 5%). The distribution of the monthly returns of this strategy are similar to the returns of the stock (<code>03_ml_strategy.ipynb</code>).


```python
fig = plt.figure(1)
plt.subplot(1,3,1)
pf.plot_annual_returns(returns)
plt.subplot(1,3,2)
pf.plot_monthly_returns_dist(returns)
plt.subplot(1,3,3)
pf.plot_monthly_returns_heatmap(returns)
plt.tight_layout()
fig.set_size_inches(15,5)
```


![png](02_ta_strategy_files/02_ta_strategy_30_0.png)


Box and whiskers are illustrative of the median, quarters returns (25th and 75th percentile) and outliers. Monthly, we can see that the whiskers (returns falling out of the 25th - 75th percentiles) rank +/-10%, which reflect that returns are spread out. This is not ideal when these outliers are negative. The trading strategy should conveniently handle volatility.


```python
pf.plot_return_quantiles(returns);
```


![png](02_ta_strategy_files/02_ta_strategy_32_0.png)


### Rolling plots

Rolling plots which show how an estimate changes throughout backtest period. A Sharpe ratio is the average return earned in excess of risk-free asset over its volatility, it is a mesure of portfolio performance. A volatile Sharpe ratio may indicate that the strategy may be riskier or non-performing at certain time points.<br>
In this particular case, the Sharpe ratio has been around 1% except in 2017 peaking at 5% (this also coincides with great returns). It should be explored if any market-events influencing this behaviour occurred during that period.


```python
pf.plot_rolling_sharpe(returns);
```


![png](02_ta_strategy_files/02_ta_strategy_35_0.png)


### Drawdown graphs

The first graph portrays the top 5 drawdown periods measured in terms of cumulative returns and the second graph depicts percentage drawdown. Both plots allow for a quick check into the time periods during which the algorithm fare with difficulties. Broadly speaking, the less volatile an algorithm is, the more minimal the drawdowns.<br>
In 2016, the strategy endured several drawdowns with a major one reaching -17.8%.


```python
pf.plot_drawdown_periods(returns);

```


![png](02_ta_strategy_files/02_ta_strategy_38_0.png)



```python
pf.plot_drawdown_underwater(returns);
```


![png](02_ta_strategy_files/02_ta_strategy_39_0.png)


### Gross leverage ratio

This is an important ratio as it affects how you trade on margin. <br>
Good strategies generally start with an initial leverage of 1, which can be adapted upon strategy's viability. A lower Sharpe ratio indicates that the strategy has a higher volatility per unit return, making it more risky to lever up. On the other hand, a higher Sharpe ratio indicates lower volatility per unit return, allowing you to increase the leverage and correspondingly, returns.<br>
For this strategy, 2016 has definitely been a risky year.


```python
pf.plot_gross_leverage(returns, positions);
```


![png](02_ta_strategy_files/02_ta_strategy_42_0.png)


### Daily turnover

This plot reflects how many shares are traded as a fraction of total shares, which can be indicative of the transaction costs associated to the algorithm but also provide a better out of sample estimation.


```python
pf.plot_turnover(returns, transactions, positions);
```


![png](02_ta_strategy_files/02_ta_strategy_45_0.png)


**Final note**<br>
Convert Notebook to Markdown and graphs to .png format for conclusions' presentation.


```python
!jupyter nbconvert --to markdown 02_ta_strategy.ipynb
```
