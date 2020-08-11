import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from strategies.run_zipline import run_strategy

def main():
    print("*** KSchool Master of Data Science ed.19 Capstone Project: Machine Learning for IBEX35 stocks Algorithmic Trading ***")
    perf = run_strategy('buy_and_hold')
    perf.to_csv("reports/buy_and_hold.csv")


if __name__ == '__main__':
    main()
