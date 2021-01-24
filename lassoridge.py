import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

startup = pd.read_csv('/home/sushil/Documents/Assingment/MLR-Ass5/50_Startups.csv')

startup.head()

# startup.columns = ['rd','adm','mar','state','pf']
# startup.columns

startup.head()

# plt.hist(startup.rd)
# plt.boxplot(startup.rd)

# plt.hist(startup.adm)
# plt.boxplot(startup.adm)

# plt.hist(startup.mar)
# plt.boxplot(startup.mar)

# plt.hist(startup.pf)
# plt.boxplot(startup.pf)


# startup.head()
# startup.corr()
# startup.state.value_counts
# startup['state'].nunique()

# startup.head()

# startup = startup.rename(columns = {'state_California' : 'ST_CAL', 'state_Florida' : 'ST_FLO', 'state_New York' : 'ST_NY'})
# startup.head()

startup.columns

# import seaborn as sns
# sns.pairplot(startup)

# import statsmodels.formula.api as smf

# ml1 = smf.ols('pf~rd+adm+mar+ST_CAL+ST_FLO+ST_NY',data=startup).fit()

# ml1.params
# ml1.summary()


# mladm = smf.ols('pf~adm',data=startup).fit()
# mladm.params
# mladm.summary()

# mlmar = smf.ols('pf~mar',data=startup).fit()
# mlmar.params
# mlmar.summary()

# ml2 = smf.ols('pf~adm+mar',data=startup).fit()
# ml2.params
# ml2.summary()


# import statsmodels.api as sm
# sm.graphics.influence_plot(ml1)


# startup_new = startup.drop(startup.index[[49,48,46]],axis=0)
# startup_new

# ml1_new = smf.ols('pf~rd+adm+mar+ST_CAL+ST_FLO+ST_NY',data=startup_new).fit()
# ml1_new.params
# ml1_new.summary()

# print(ml1_new.conf_int(0.01))
# startup_new.head()


# pf_pred = ml1_new.predict(pd.DataFrame(startup_new[['rd','adm','mar','ST_CAL','ST_FLO','ST_NY']]))
# pf_pred

# rsd_rd = smf.ols('rd~adm+mar+ST_CAL+ST_FLO+ST_NY',data=startup_new).fit().rsquared
# vif_rd = 1/(1-rsd_rd)
# vif_rd

# rsd_adm = smf.ols('adm~rd+mar+ST_CAL+ST_FLO+ST_NY',data=startup_new).fit().rsquared
# vif_adm = 1/(1-rsd_adm)
# vif_adm

# rsd_mar = smf.ols('mar~rd+adm+ST_CAL+ST_FLO+ST_NY',data=startup_new).fit().rsquared
# vif_mar = 1/(1-rsd_mar)
# vif_mar

# d1 = {'variable':['rd','adm','mar'],'VIF':[vif_rd,vif_adm,vif_mar]}
# vif_frame = pd.DataFrame(d1)
# vif_frame

# sm.graphics.plot_partregress_grid(ml1_new)


# final_model = smf.ols('pf~rd+mar+ST_CAL+ST_FLO+ST_NY', data = startup_new).fit()
# final_model.params
# final_model.summary()


startup_new.columns


# from sklearn.model_selection import train_test_split
# startup_train,startup_test = train_test_split(startup_new,test_size = 0.4)

# model_train = smf.ols('pf~rd+mar+ST_CAL+ST_FLO+ST_NY', data = startup_train).fit()
# model_train.summary()


# train_pred = model_train.predict(startup_train)
# train_pred

# train_resid = train_pred - startup_train.pf
# train_resid

# train_rmse = np.sqrt(np.mean(train_resid*train_resid))
# train_rmse

# model_test = smf.ols('pf~rd+mar+ST_CAL+ST_FLO+ST_NY',data=startup_test).fit()
# model_test.params
# model_test.summary()


# test_pred = model_train.predict(startup_test)
# test_pred

# test_resid = test_pred - startup_test.pf
# test_resid

# test_rmse = np.sqrt(np.mean(test_resid*test_resid))
# test_rmse

startup = pd.get_dummies(startup, columns=['State'])

startup = startup.rename(columns = {'state_California' : 'ST_CAL', 'state_Florida' : 'ST_FLO', 'state_New York' : 'ST_NY'})
startup.head()
startup.isnull().sum()
startup.describe()
startup.dtypes
np.isfinite(startup.all())

from sklearn.model_selection import train_test_split
train,test = train_test_split(startup,test_size = 0.35)


train.head()

test.head()

from sklearn.linear_model import Ridge

rm1=Ridge(alpha=0.01,normalize=True)

rm1.fit(train.iloc[:,:7],train.Profit)

rm1.coef_

rm1.intercept_

rm1.alpha

pred_rm1 = rm1.predict(train.iloc[:,:7])

pred_rm1

rm1.score(train.iloc[:,:7],train.Profit)

np.sqrt(np.mean((pred_rm1-train.Profit)**2))

train_rmse = []
test_rmse = []
r_sqrd = []
alphas = np.arange(0,50,0.5)

for i in alphas:
     RM = Ridge(alpha = i,normalize = True)
     RM.fit(train.iloc[:,:7],train.Profit)
     r_sqrd.append(RM.score(train.iloc[:,:7],train.Profit))
#    train_rmse.append(np.sqrt(np.mean(RM.predict(train.iloc[:,:7]-train.Profit)**2)))
#     test_rmse.append(np.sqrt(np.mean(RM.predict(test.iloc[:,:7]-test.Profit)**2)))
    

plt.scatter(x=alphas,y=r_sqrd);plt.xlabel("alphas");plt.ylabel("R_squared")

#plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alphas");plt.ylabel("train_rmse")

#plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alphas");plt.ylabel("test_rmse")

#plt.legend(("alphas vs R_squared","alphas vs train_rmse","alphas vs test_rmse"))

from sklearn.linear_model import Lasso

ls1=Lasso(alpha=0.01,normalize=True)

ls1.fit(train.iloc[:,:7],train.Profit)

ls1.coef_

ls1.intercept_

ls1.alpha

pred_ls1 = ls1.predict(train.iloc[:,:7])

pred_ls1

ls1.score(train.iloc[:,:7],train.Profit)

np.sqrt(np.mean((pred_ls1-train.Profit)**2))

train_rmse = []
test_rmse = []
r_sqrd = []
alphas = np.arange(0,30,0.5)

for i in alphas:
    LS = Lasso(alpha = i,normalize = True,max_iter=500)
    RM.fit(train.iloc[:,:7],train.Profit)
    r_sqrd.append(RM.score(train.iloc[:,:7],train.Profit))
#    train_rmse.append(np.sqrt(np.mean(RM.predict(train.iloc[:,:7]-train.Profit)**2)))
#   test_rmse.append(np.sqrt(np.mean(RM.predict(test.iloc[:,:7]-test.Profit)**2)))
    

plt.scatter(x=alphas,y=r_sqrd);plt.xlabel("alphas");plt.ylabel("R_squared")

#plt.scatter(x=alphas,y=train_rmse);plt.xlabel("alphas");plt.ylabel("train_rmse")

#plt.scatter(x=alphas,y=test_rmse);plt.xlabel("alphas");plt.ylabel("test_rmse")

plt.legend(("alphas vs R_squared","alphas vs train_rmse","alphas vs test_rmse"))

