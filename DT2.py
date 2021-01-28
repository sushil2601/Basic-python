import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fraud = pd.read_csv('/home/sushil/Downloads/Fraud_check.csv')

fraud.head()

fraud.describe()

fraud.corr()

fraud = fraud.rename(columns = {'Marital.Status' : 'marital_status' , 'Taxable.Income' : 'taxable_income' , 'City.Population' : 'city_population' , 'Work.Experience' : 'work_experience'})

fraud.head()

plt.hist(fraud.taxable_income)

sns.boxplot(fraud.taxable_income)

plt.hist(fraud.city_population)

sns.boxplot(fraud.city_population)

plt.hist(fraud.work_experience)

sns.boxplot(fraud.work_experience)

fraud.isnull().sum()

sns.heatmap(fraud.isnull(), yticklabels = False, cmap = 'viridis' )

def cal(i):
    
    if i <= 30000:
        return 'Risky'
    
    else:
        return 'Good'

fraud['taxable_income'] = fraud['taxable_income'].apply(cal)

fraud['taxable_income']

sns.countplot('taxable_income',hue = 'Undergrad', data = fraud)

sns.stripplot(x = 'taxable_income', y = 'work_experience', data = fraud)

sns.stripplot(x = 'taxable_income', y = 'city_population', data = fraud)

#sns.stripplot(x = 'taxable_income', y = 'Urban', data = fraud)

fraud_new = pd.get_dummies(fraud[['Undergrad', 'marital_status', 'Urban']])

fraud_new

fraud_new = pd.concat([fraud[['work_experience','city_population','taxable_income']],fraud_new],axis = 1)

fraud_new

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier

x = fraud_new.drop('taxable_income',axis = 1)

x.head()

y = fraud_new['taxable_income']

y.head()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)

model = DecisionTreeClassifier(criterion = 'entropy')

model

model.fit(x_train,y_train)

predictions = model.predict(x_test)

predictions

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))

