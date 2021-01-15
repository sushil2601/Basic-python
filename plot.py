import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mba = pd.read_csv('/home/sushil/mba.csv')
mba

# plt.boxplot(mba['gmat'])
# plt.boxplot(mba['datasrno'])
# plt.boxplot(mba['workex'])

# plt.hist(mba['gmat'])
# plt.hist(mba['datasrno'])
# plt.hist(mba['workex'])

mba['gmat'].skew()
mba['gmat'].kurt()

mba['datasrno'].skew()
mba['datasrno'].kurt()

mba['workex'].skew()
mba['workex'].kurt()