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


## Installation

A conda environment has been created in order to (i) run Zipline in a Python version 3.5 environment, (ii) isolate Zipline's dependencies and (iii) control for possible interactions with base environment. For further details, please visit [Zipline Install](https://www.zipline.io/install.html).<br>

A file `environment.yml` has also been included in the dossier for replication purposes.<br>


## Folder's structure

### project's report
This file summarises the problem statement, the project set-up, implementation and key results and conclusion plus the visualisation user's guide.<br>

### notebooks
The notebooks contain a step-by-step project's narrative and implementation:
<ul>
    <li>01_eda_quandl: you would find data bundle ingestion and loading, exploratory data analysis and time series analysis. Albeit the provision of US equities' financial series by Quandl (the bundle used for this project), Zipline allows for custom data bundles ingestion, so a testimonial exercise with REPSOL (IBEX35) equity has also been added.</li>
    <li>02_ta_strategy contains definition and backtesting of the dual moving average crossover trading strategy</li>
    <li>03_ml_strategy enshrines feature's engineering, ML model design, training and selection. After the trading order based on price prediction has been executed (python scripts), model backtesting has been performed in the notebook</li>
    <li>04_trading_viz displays graphs and creates a web page for visualisation purposes</li>.
</ul>

### images
Pics of trading order's execution results and backtest graphs to be used for visualisation purposes.<br>

### reports
Reports issued from order's execution are saved in this file.<br>

### strategies
Python scripts for the *buy-and-hold* strategy are located here as well as a folder for ML models saved.<br>


## Main results
### Results
<ul>
    <li><b>Both trading strategies' performance are different</b> Actually, TA strategy records cumulative returns of 34,1%, consistent with small investors’ conservative strategies, whereas ML strategy reports 6050%. Capital used at the end of the period amounts to 14.454 USD and 116.175 USD respectively.</li>
![strategies](/images/viz/strategies.png)
![cumulative_returns](/images/viz/cumulative_returns.png)
    <li><b>Number of trading signals is not comparable</b>. Trading signals vary in a 1:3 proportion according to trading strategy: (i) 26 signals stem from arithmetic moving averages crossover whereas (ii) 80 signals were issued from 8-day trading window (out 635 days) in ML stock's strategy. Considering that most of them are buying signals (see paragraph 3.1), more trading (buying) opportunities have arouse and more capital has fueled-in and has been capitalised in ML based strategy compared to TA strategy leading to extraordinary cumulative returns.</li>
    <li><b>Increased volatility in ML strategy</b>. When the stock price trend becomes steeper from mid-2016 until mid-2017, volatility skyrockets at impossible levels, which is also reflected on the variability of the rolling sharpe (excess return over volatility) and risk exposure. The absence of shorts positions (paragraph 3.1), exit conditions, such as stop-losses, contribute to it. As a direct consequence, not only profits but also losses tend to be very large.</li>
![rolling_volatility](/images/viz/rolling_volatility.png)
![ml_exposure](/images/viz/ml_exposure.png)
![ml_shorts_longs](/images/viz/ml_shorts_longs.png)
![ml_PnL](/images/viz/ml_PnL.png)
    <li><b>Huge drawdowns in ML strategy</b>.Drawdowns are huge and occur at the beginning of the trading period (until beginning 2017), when portfolio value does not compensate for losses. Whereas ML strategy’s drawdowns are more dependent on volume, TA strategy’s drawdowns are more constraint to stock’s price variability.</li>
![drawdowns](/images/viz/drawdowns.png)
</ul>

Generally speaking, it can be stated that ML strategy is better than TA strategy in reaping positive momentum benefits as well as losses from negative momentum.<br> 
Nonetheless, in light of results, this seem a weak statement. In purity, both strategies are not comparable in terms of trading opportunities and capital used, all the more so, considering that basic trading features, such as the definition of short trades, stop losses, capital constraint and risk management have not been defined.<br>

## Future Steps
This project has been conceived as a research in progress and as such, it has the vocation to further delve into stock prediction and algorithmic trading. Possible further steps can be:<br>
<ul>
    <li>explore different trading strategies other than buy-and-hold with aforementioned features</li>
    <li>contain volatility with VaR (Value-at-Risk) and CVaR (Conditional-Value-at-Risk) metrics, that can also be predicted with (i) parametric estimations, such as Monte Carlo estimations and/or (ii) non-parametric estimations with ML, such as SVR and KDE</li>
    <li>explore other machine learning and deep learning models for stock prediction</li>
    <li>expand features’ engineering  with readily available built-in factors at Quantopian's and,</li>
    <li>include more assets for portfolio diversification and optimisation.</li>
</ul>
