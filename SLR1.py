
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salaryhike = pd.read_csv('/home/sushil/Documents/Assingment/SLR-ASS4/Salary_Data.csv')
salaryhike
salaryhike.columns
salaryhike
x1 = plt.hist(salaryhike.YearsExperience)
x = plt.boxplot(salaryhike.YearsExperience)
y1 = plt.hist(salaryhike.Salary)
y = plt.boxplot(salaryhike.Salary)

salaryhike.isnull()
salaryhike.isnull().sum()

plt.plot(salaryhike.YearsExperience,salary_hike.Salary,"bo");plt.xlabel("YearsExperience");plt.ylabel("Salary")

salaryhike.Salary.corr(salaryhike.YearsExperience)

# For preparing linear regression model

import statsmodels.formula.api as smf
model = smf.ols("Salary~YearsExperience",data = salaryhike).fit()

model.params 
    
model.summary()

print(model.conf_int(0.05))

pred = model.predict(salaryhike.iloc[:,0])
pred

plt.scatter(x = salaryhike['YearsExperience'],y = salaryhike['Salary'],color ='red');plt.plot(salaryhike['YearsExperience'],pred,color = 'black');plt.xlabel('Experience');plt.ylabel('Salary')
pred.corr(salaryhike.Salary)

# Transforming varaible for accuracy

model2 = smf.ols('Salary~np.log(YearsExperience)',data = salaryhike).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01))

pred2 = model2.predict(pd.DataFrame(salaryhike.iloc[:,0]))
pred2
pred2.corr(salaryhike.Salary)

plt.scatter(x = salaryhike['YearsExperience'],y = salaryhike['Salary'],color ='green');plt.plot(salaryhike['YearsExperience'],pred2,color = 'black');plt.xlabel('Experience');plt.ylabel('Salary')

Exponential transformation

model3 = smf.ols('np.log(Salary)~YearsExperience',data = salaryhike).fit()
model3.params
model3.summary()

print(model3.conf_int(0.01))

pred_log = model3.predict(pd.DataFrame(salaryhike.iloc[:,0]))
pred_log
pred3 = np.exp(pred_log)
pred3


pred3.corr(salaryhike.Salary)
plt.scatter(x = salaryhike['YearsExperience'],y = salaryhike['Salary'],color ='pink');plt.plot(salaryhike['YearsExperience'],pred3,color = 'black');plt.xlabel('Experience');plt.ylabel('Salary')

resid = model3.resid_pearson
resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel('Observation number');plt.ylabel('Standarized Residual')

Predicted vs Actual value

plt.scatter(x=pred3,y=salaryhike.Salary);plt.xlabel('Predicted');plt.ylabel('Actual')

Quadratic model

salaryhike['YearsExperience_sq'] = salaryhike.YearsExperience*salaryhike.YearsExperience
model_quad = smf.ols('np.log(Salary)~YearsExperience+YearsExperience_sq',data = salaryhike).fit()
model_quad.params
model_quad.summary()

pred_quad = model_quad.predict(salaryhike)
pred_quad
model_quad.conf_int(0.05)
plt.scatter(salaryhike.YearsExperience,salaryhike.Salary,c='b');plt.plot(salaryhike.YearsExperience,pred_quad,"r")
plt.scatter(np.arange(30),model_quad.resid_pearson);plt.axhline(y=0,color = 'red')
plt.hist(model_quad.resid_pearson)

