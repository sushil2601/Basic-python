import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

delivery =pd.read_csv('/home/sushil/Documents/Assingment/SLR-ASS4/delivery_time.csv')
delivery
delivery.columns
delivery
delivery.columns = ['Del_time','Sor_time']
delivery
x1 = plt.hist(delivery.Sor_time)
x = plt.boxplot(delivery.Sor_time)
y1 = plt.hist(delivery.Del_time)
y = plt.boxplot(delivery.Sor_time)

delivery.isnull()
delivery.isnull().sum()

plt.plot(delivery.Sor_time,delivery.Del_time,"bo");plt.xlabel("Sor_time");plt.ylabel("Del_time")

delivery.Del_time.corr(delivery.Sor_time)

# For preparing linear regression model

import statsmodels.formula.api as smf
model = smf.ols("Del_time~Sor_time",data = delivery).fit()

model.params 
    
model.summary()

print(model.conf_int(0.05))

pred = model.predict(delivery.iloc[:,0])
pred

plt.scatter(x = delivery['Sor_time'],y = delivery['Del_time'],color ='red');plt.plot(delivery['Sor_time'],pred,color = 'black');plt.xlabel('Sor_time');plt.ylabel('Del_time')
pred.corr(delivery.Del_time)

# Transforming varaible for accuracy

model2 = smf.ols('Del_time~np.log(Sor_time)',data = delivery).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01))

pred2 = model2.predict(pd.DataFrame(delivery.iloc[:,0]))
pred2
pred2.corr(salaryhike.Salary)

plt.scatter(x = delivery['Sor_time'],y = delivery['Del_time'],color ='green');plt.plot(delivery['Sor_time'],pred2,color = 'black');plt.xlabel('Sortingtime');plt.ylabel('Deliveringtime')

# Exponential transformation

model3 = smf.ols('np.log(Del_time)~Sor_time',data = delivery).fit()
model3.params
model3.summary()

print(model3.conf_int(0.01))

pred_log = model3.predict(pd.DataFrame(delivery.iloc[:,0]))
pred_log
pred3 = np.exp(pred_log)
pred3

plt.scatter(x = delivery['Sor_time'],y = delivery['Del_time'],color ='pink');plt.plot(delivery['Sor_time'],pred3,color = 'black');plt.xlabel('Sortingtime');plt.ylabel('Deliverytime')

resid = model3.resid_pearson
resid
plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel('Observation number');plt.ylabel('Standarized Residual')

# Predicted vs Actual value

plt.scatter(x=pred3,y=delivery.Del_time);plt.xlabel('Predicted');plt.ylabel('Actual')

# Quadratic model

delivery['Sor_time_sq'] = delivery.Sor_time*delivery.Sor_time
model_quad = smf.ols('np.log(Del_time)~Sor_time+Sor_time_sq',data = delivery).fit()
model_quad.params
model_quad.summary()

pred_quad = model_quad.predict(delivery)
pred_quad
model_quad.conf_int(0.05)
plt.scatter(delivery.Sor_time,delivery.Del_time,c='b');plt.plot(delivery.Sor_time,pred_quad,"r")
plt.scatter(np.arange(30),model_quad.resid_pearson);plt.axhline(y=0,color = 'red')
plt.hist(model_quad.resid_pearson)
