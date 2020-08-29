# Machine Learning for Algorithmic Trading

Capstone project for **Master Data Science - KSchool - 19ed**.<br>

Based on daily close price data from APPLE stock from 1990 until 2018, **this project aims at testing whether ML based <i>buy-and-hold</i> trading strategies have the power to outperform more traditional ones based on technical analysis**. <br>

Hence, the project takes a four-step approach:
<ul>
    <li>implement and backtest a <i>dual moving average crossover</i> trading strategy</li>
    <li>develop a ML Model for stock prediction</li>
    <li>plug-in the ML Model into a Trading Strategy and backtest it and,</li>
    <li>compare results</li>
</ul>

For such purpose, we use a locally installed Zipline research environment for trading sponsored by Quantopian,inc and employ different ML techniques such as random forests, gradient boosting, support vector machines and kernel density estimations.<br>


# Installation

A conda environment has been created in order to (i) run Zipline in a Python version 3.5 environment, (ii) isolate Zipline's dependencies and (iii) control for possible interactions with base environment. For further details, please visit [Zipline Install](https://www.zipline.io/install.html).<br>

A file `environment.yml` has also been included in the dossier for replication purposes.<br>


# Folder's structure

`project's report`
This file summarises the project set-up, implementation and key findings, namely (i) problem statement, (ii) trading and backtesting: concepts and libraries, (iii) methodological approach, (iv) technical and technological specifications and pitfalls, (v) project development, (vi) main findings and conclusions and (vii) the way forward.<br>

`notebooks`
The notebooks contain a step-by-step project's narrative and implementation:
<ol>
    <li>`01_eda_quandl.ipynb`: you would find data bundle ingestion and loading, exploratory data analysis and time series analysis. Albeit the provision of US equities' financial series by Quandl (the bundle used for this project), Zipline allows for custom data bundles ingestion, so a testimonial exercise with REPSOL (IBEX35) equity has also been added.</li>
    <li>`02_ta_strategy.ipynb`: contains definition and backtesting of the dual moving average crossover trading strategy</li>
    <li>`03_ml_strategy.ipynb` enshrines feature's engineering, ML model design, training and selection. After the trading order based on price prediction has been executed (python scripts), model backtesting has been performed in the notebook</li>
    <li>`04_trading_viz.ipynb`: displays graphs and creates a web page for visualisation purposes.
</ol>

`images`
Pics of trading order's execution results and backtest graphs to be used for visualisation purposes.<br>

`reports`
Reports issued from order's execution are saved in this file.<br>

`strategies`
Python scripts for the <i>buy-and-hold</i> strategy are located here as well as a folder for ML models saved.<br>


# Main findings and conclusions
## Findings
<ul>
    <li>AAPL time series show a clear rising trend from summer 2016 onwards, thus triggering mostly buying signals along the way,</li>
    <li>trading signals vary in a 1:3 proportion according to trading strategy: (i) 26 signals stemming from arithmetic moving averages crossover in front of (ii) 80 signals (out 635 days for a 8-day trading window) from ML stock's prediction,</li>
    <li>besides that, gradient boosting and random forest predict at a high goodness of fit which allow the ML based strategy to turn accurate forecasts into trading signals.</li>
    <li>As a consequence, more capital has been fueled-in and capitalised in ML based strategy compared to TA strategy. This leads to extraordinary benefits but as analysed in conclusions also extraordinary drawdowns (on grounds of volatility).</li>
</ul>

<img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/TA_strategy.png'>
<img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/ML_strategy.png'>

As such, the following conclusions have been raised.<br>


## Conclusions
Comparing ml strategy with ta strategy:
<ol>
    <li>important cumulative returns accruing to ML strategy (6049%) compared to TA strategy (34%) over the period
        <img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/cumulative_returns.png'></li>
    <li>distribution of returns (yoy and mom) show that negative and positive returns in ta strategy are incommensurate in ml strategy.
	<img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/returns_distribution.png'></li>
    <li>volatility has been very high from mid-2016 until mid-2017 skyrocketting at impossible levels of 10.080%. This also reflects on the variability of the rolling sharpe ratio measure (excess return over volatility), which depending on the dimension of both elements, it can be either positive or negative
	<img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/rolling_volatility.png'></li>
    <li>drawdown periods of ml and ta strategies do not overlap. The former coincide in time with price volatility whereas the latter occur at the beginning of the period (until the beginning of 2017), when portfolio value compensate for losses. The maximum drawdown is set to -418% for ml strategy.
	<img><src='/home/isabel/Repos/ml_for_algo_trading/images/viz/drawdowns.png'></li>
</ol>

# Future Steps

At this stage, it is rather obvious that the ML strategy is better than the TA strategy in reaping the momentum benefits of a positive trend but on the contrary, it leads to a setback during negative trends. Hence, the trading strategy should address the following, among others:
<ul>
    <li><b>contain volatility</b> with VaR (Value-at-Risk) and CVaR (Conditional-Value-at-Risk) metrics, that can also be predicted along with (i) parametric estimations, such as Monte Carlo estimations and/or (ii) non-parametric estimations with ML techniques, such as SVR and KDE</li>
    <li>explore different trading strategies other than <i>buy-and-hold</i> ones and define market <b>exit conditions</b> with stop-loss values</li>
    <li>restrict short/long <b>exposure to risk</b> or value</li>
</ul>

Furthermore, we can also expand our feature's engineering with readily available built-in factors at Quantopian's and include more assets for portfolio diversification and optimisation.<br>

All in all, this is a <i>work in progress</i> and more research in trading and ML/DL for stock prediction shall be undertaken in the future.
