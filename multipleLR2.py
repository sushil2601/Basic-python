import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

price = pd.read_csv('/home/sushil/Documents/Assingment/MLR-Ass5/ToyotaCorolla.csv',usecols = ['Price', 'Age_08_04', 'KM', 'HP', 'cc', 'Doors', 'Gears', 'Quarterly_Tax', 'Weight'])

price

price.columns

price.head()

#plt.hist(price.Age_08_04)
#plt.boxplot(price.Age_08_04)

#plt.hist(price.KM)
#plt.boxplot(price.KM)

#plt.hist(price.HP)
#plt.boxplot(price.HP)

#plt.hist(price.cc)
#plt.boxplot(price.cc)
#plt.hist(price.Doors)
#plt.boxplot(price.Doors)
#plt.hist(price.Gears)
#plt.boxplot(price.Gears)
#plt.hist(price.Quarterly_Tax)
#plt.boxplot(price.Quarterly_Tax)
#plt.hist(price.Weight)
#plt.boxplot(price.Weight)

#plt.hist(price.Price)
#plt.boxplot(price.Price)

#price.describe()

price.corr()
price.Price.value_counts
price['Price'].nunique()

# import seaborn as sns
# sns.pairplot(price)


import statsmodels.formula.api as smf

ml1 = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=price).fit()

ml1.params
ml1.summary()

m_cc = smf.ols('Price~cc',data=price).fit()
m_cc.params
m_cc.summary()

m_Doors = smf.ols('Price~Doors',data=price).fit()
m_Doors.params
m_Doors.summary()

ml2 = smf.ols('Price~cc+Doors',data=price).fit()
ml2.params
ml2.summary()

import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

price_new = price.drop(price.index[[80,960,221]], axis = 0)
price_new

ml1_new = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=price_new).fit()
ml1_new.params
ml1_new.summary()

print(ml1_new.conf_int(0.01))
price_new.head()

 Price_pred = ml1_new.predict(price_new[['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']])
Price_pred

plt.scatter(price_new.Price, Price_pred)

plt.scatter(Price_pred, ml1_new.resid_pearson);plt.axhline(y=0, color = 'green')

from sklearn.model_selection import train_test_split

price_train, price_test = train_test_split(price_new, test_size = 0.2)

price_train

price_test

model_train = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = price_train).fit()

model_train.summary()

train_pred = model_train.predict(price_train)

train_pred

train_resid = train_pred - price_train.Price

train_resid

train_rmse = np.sqrt(np.mean(train_resid * train_resid))

train_rmse

test_pred = model_train.predict(price_test)

test_pred

test_resid = test_pred - price_test.Price

test_resid

test_rmse = np.sqrt(np.mean(test_resid * test_resid))

test_rmse

model_test = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = price_test).fit()

model_test.params

model_test.summary()

