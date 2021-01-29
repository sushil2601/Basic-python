import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

fraud = pd.read_csv('/home/sushil/Desktop/Fraud_check.csv')

fraud.head()

fraud.shape

fraud.describe()

fraud.corr()

fraud = fraud.rename(columns = {'Marital.Status' : 'marital_status' , 'Taxable.Income' : 'taxable_income' , 'City.Population' : 'city_population' , 'Work.Experience' : 'work_experience'})

fraud.head()

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

fraud_new = pd.get_dummies(fraud[['Undergrad', 'marital_status', 'Urban']])

fraud_new.head()

fraud_new = pd.concat([fraud[['work_experience','city_population','taxable_income']],fraud_new],axis = 1)

fraud_new.head()

x = fraud_new.drop('taxable_income',axis = 1)

x.head()

y = fraud_new['taxable_income']

y

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4)

from imblearn.over_sampling import SMOTE

sm = SMOTE()

x_train,y_train = sm.fit_resample(x_train,y_train)

sns.countplot(y_train)

RF = RandomForestClassifier(n_jobs = 3, criterion = 'entropy', oob_score = True)

RF.fit(x_train,y_train)

RF.score(x_test,y_test)

RF.score(x_train,y_train)

x1_train,x1_test,y1_train,y1_test = train_test_split(x,y,test_size = 0.2)

x1_train,y1_train = sm.fit_resample(x1_train,y1_train)

RF = RandomForestClassifier(n_jobs = 5, criterion = 'entropy', oob_score = True)

RF.fit(x1_train,y1_train)

RF.score(x1_test,y1_test)

RF.score(x1_train,y1_train)

x2_train,x2_test,y2_train,y2_test = train_test_split(x,y,test_size = 0.35)

x2_train,y2_train = sm.fit_resample(x2_train,y2_train)

RF = RandomForestClassifier(n_jobs = 10, criterion = 'entropy', oob_score = True)

RF.fit(x2_train,y2_train)

RF.score(x2_test,y2_test)

RF.score(x2_train,y2_train)

