from zipline.api import order, symbol, record, set_benchmark
from matplotlib import pyplot as plt
from joblib import load
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

import pandas as pd


class BuyAndHold:
    stocks = ['AAPL']
    lag = 33
    forecast = 8

    def initialize(self, context):
        context.has_ordered = False
        context.stocks = self.stocks
        context.asset = symbol('AAPL')
        set_benchmark(False)
        context.regressor = load('./strategies/models/rfr_regressor.joblib')

    def handle_data(self, context, data):
        for stock in context.stocks:
            timeseries = data.history(
                symbol(stock),
                'price',
                bar_count=self.lag,
                frequency='1d'
            )
            np_timeseries = np.array(timeseries.values).reshape(1, -1)
            preds = context.regressor.predict(np_timeseries)
            max_price = np.max(preds)
            historical_mean = np.mean(np_timeseries)

        if max_price > historical_mean:
            order(symbol(stock), 100)

        if max_price < historical_mean:
            order(symbol(stock), -100)

        record(AAPL=data.current(context.asset, 'price'))

    def _test_args(self):
        return {
            'start': pd.Timestamp('2000-1-1', tz='utc'),
            'end': pd.Timestamp('2018-3-27', tz='utc'),
            'capital_base': 1e5
        }

    def analyze(self, context, perf):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('Portfolio value in EUR')

        ax2 = fig.add_subplot(212)
        perf['AAPL'].plot(ax=ax2)

        ax2.set_ylabel('price in EUR')
        plt.legend(loc=0)
        plt.show()

        # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perf)
        # pf.create_returns_tear_sheet(returns, benchmark_ret=None)

