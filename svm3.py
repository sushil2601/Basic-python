import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/sushil/Documents/Assingment/SVM-17/forestfires.csv')

data.head()

data.describe()

data.columns

data.shape

data.corr().head()

sns.boxplot(x = 'size_category',y = 'FFMC',data = data,palette = 'hls')

sns.boxplot(y = 'size_category' , x = 'DMC',data = data , palette = 'hls')

sns.boxplot(y = 'size_category' ,x = 'DC' , data = data,palette = 'hls')

sns.boxplot(y = 'size_category', x = 'ISI' , data = data , palette = 'hls')

plt.hist(data['size_category'])

data['size_category'].value_counts()

plt.hist(data['DMC'])

plt.hist(data['ISI'])

sns.heatmap(data.isnull(), cmap = 'viridis', yticklabels = False)

data.isnull().sum().head()

plt.figure(figsize = (14,7))
sns.countplot('month', hue = 'size_category', data = data)

plt.figure(figsize = (14,7))
sns.countplot('day', hue = 'size_category', data = data)

data_new = data.drop(['day','month'],axis = 1)

data_new.head()

x = data_new.drop('size_category',axis = 1)

x.head()

y = data_new['size_category']

y.head()

#from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.3)

from sklearn.svm import SVC

###### Using Linear kernal ########

model = SVC(kernel = 'linear')

model.fit(x_train , y_train)

test_pred = model.predict(x_test)

test_pred

np.mean(test_pred == y_test)

model.score(x_train,y_train)

####### Linear model accuracy = 98.71% ##########

from sklearn.metrics import classification_report , confusion_matrix

print(classification_report(test_pred , y_test))

print(confusion_matrix(test_pred , y_test))

########## using poly kernel ##########

model1 = SVC(kernel = 'poly')

model1.fit(x_train,y_train)

model1.score(x_train , y_train)

test1_pred = model1.predict(x_test)

test1_pred

np.mean(test1_pred == y_test)

print(classification_report(test1_pred,y_test))

print(confusion_matrix(test1_pred,y_test))

########  Using poly kernel model gets underfitted #########

#######   Using rbf kernel ########

model2 = SVC(kernel = 'rbf')

model2.fit(x_train,y_train)

test2_pred = model2.predict(x_test)

test2_pred

np.mean(test2_pred == y_test)

model2.score(x_train,y_train)

######## Again model2 build using poly kernel is gets underfitted ####### 

########## So , the Ist model performance is best ###########