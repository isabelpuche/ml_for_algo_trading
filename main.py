import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from strategies.run_zipline import run_strategy

def main():
    print("*** KSchool Master of Data Science ed.19 Capstone Project: Machine Learning for NYSE stocks Algorithmic Trading ***")
    print("*** Plug-in Random Forest Implementation into Trading Bot***")
    perf = run_strategy('buy_and_hold')
    perf.to_csv("reports/buy_and_hold_RF.csv")


if __name__ == '__main__':
    main()
