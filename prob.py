import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mba = pd.read_csv('/media/sushil/suman/DataSet1/mba.csv')
mba

import scipy.stats as stats
stats.norm.ppf(0.975,0,1)   #Same as qnorm in r
stats.norm.cdf(740,711,29)  #Same as pnorm in r
stats.norm.cdf(740,711,29)-stats.norm.cdf(690,711,29) 
