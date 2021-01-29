import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

company = pd.read_csv('/home/sushil/Desktop/Company_Data.csv')

company.head()

company.describe()

company.shape

company['Sales'].value_counts()

company.corr()

np.median(company['Sales'])

data = company

data.head()

def cal(i):
    
    if i >= 7.49:
        return 'High'
    else:
        return 'Low'

data['Sales']=data['Sales'].apply(cal)

data['Sales'].head()

sns.countplot(data['Sales'])

data_new = pd.get_dummies(data[['ShelveLoc','Urban' , 'US']],drop_first = True)

data_new

data_new = pd.concat([data.iloc[:,[0,1,2,3,4,5,6,7,8]],data_new],axis = 1)

data_new

data_new1 = data_new.drop('ShelveLoc',axis = 1)

data_new1.head()

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,classification_report

from imblearn.over_sampling import SMOTE

x = data_new1.drop('Sales', axis = 1)

y = data_new1['Sales']

x.head()

y.head()

y.value_counts()

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)

sm = SMOTE()

x_train, y_train = sm.fit_resample(x_train, y_train)

sns.countplot(y_train)

RF = RandomForestClassifier(n_jobs = 3, criterion = 'entropy', oob_score = True)

RF.fit(x_train,y_train)

RF.estimators_

RF.n_classes_

RF.oob_score

predictions = RF.predict(x_test)

predictions

from sklearn.metrics import classification_report

print(classification_report(predictions, y_test))

RF.score(x_test,y_test)

RF.score(x_train,y_train)

x1_train, x1_test, y1_train, y1_test = train_test_split(x,y, test_size = 0.4)

sns.countplot(y1_train)

RF = RandomForestClassifier(n_jobs = 4, criterion = 'entropy', oob_score = True)

RF.fit(x1_train,y1_train)

RF.score(x1_test,y1_test)

RF.score(x1_train,y1_train)

