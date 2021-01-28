import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

company = pd.read_csv('/home/sushil/Downloads/Company.csv')

company.head()

company.shape

company.describe()

company['US'].nunique()

company['US'].unique()

company['US'].value_counts()

plt.hist(company.Sales)

sns.boxplot(company.Sales)

plt.hist(company.CompPrice)

sns.boxplot(company.CompPrice)

plt.hist(company.Income)

sns.boxplot(company.Income)

plt.hist(company.Advertising)

sns.boxplot(company.Advertising)

plt.hist(company.Population)

sns.boxplot(company.Population)

plt.hist(company.Price)

sns.boxplot(company.Price)

plt.hist(company.Age)

sns.boxplot(company.Age)

plt.hist(company.Education)

sns.boxplot(company.Education)

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

sns.stripplot(x = 'Sales', y = 'Income', data = data)

sns.countplot('Sales', hue = 'Urban', data = data)

sns.stripplot(x = 'Sales', y = 'Advertising', data = data)

sns.stripplot(x = "Sales",y = "CompPrice",data = data)

sns.stripplot(x= "Sales" , y = "Population" , data = data)

sns.stripplot(x = "Sales" , y = "Price" , data = data)

#from sklearn.preprocessing import LabelBinarizer 

#label_binarizer = LabelBinarizer()

#label_binarizer_output = label_binarizer.fit_transform( data['ShelveLoc'])

#result_data = pd.DataFrame(label_binarizer_output, 
                         columns = label_binarizer.classes_) 

#result_data

data_new = pd.get_dummies(data[['ShelveLoc','Urban' , 'US']],drop_first = True)

data_new

#data_new.drop('ShelveLoc',axis = 1)

data_new = pd.concat([data.iloc[:,[0,1,2,3,4,5,6,7,8]],data_new],axis = 1)

data_new.head()

data_new1 = data_new.drop('ShelveLoc',axis = 1)

data_new1

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix,classification_report

x = data_new1.drop('Sales',axis = 1)

y = data_new1['Sales']

x.head()

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = DecisionTreeClassifier(criterion = 'entropy')

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

