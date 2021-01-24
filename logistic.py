import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report

credit=pd.read_csv('/home/sushil/Downloads/creditcard.csv')

credit.head()

credit.drop('Unnamed: 0',inplace=True,axis=1)

credit.head()

credit.shape

credit.corr()

plt.hist(credit.card)

plt.boxplot(credit.reports)

#plt.hist(credit.age)
plt.boxplot(credit.age)

#plt.hist(credit.income)
plt.boxplot(credit.income)

# plt.hist(credit.share)
# plt.boxplot(credit.share)
# plt.hist(credit.expenditure)
# plt.boxplot(credit.expenditure)
# plt.hist(credit.dependents)
# plt.boxplot(credit.dependents)
# plt.hist(credit.months)
# plt.boxplot(credit.months)
# plt.hist(credit.majorcards)
# plt.boxplot(credit.majorcards)
# plt.hist(credit.active)
# plt.boxplot(credit.active)

credit.isnull().sum()

sns.heatmap(credit.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

sns.countplot(x='card',data=credit,palette='hls')

pd.crosstab(credit.card,credit.owner).plot(kind='bar')

pd.crosstab(credit.card,credit.selfemp).plot(kind='bar')

sns.countplot(credit.owner)

pd.crosstab(credit.owner,credit.selfemp).plot(kind='bar')

sns.countplot('selfemp', hue ='dependents', data = credit)

sns.countplot(credit.dependents)

sns.countplot('owner', hue = 'dependents', data = credit)

np.mean(credit.income)

sns.pairplot(credit)

 credit['card'].replace(('yes','no'),(1,0),inplace=True)

credit.head()

credit = pd.get_dummies(credit,columns=['owner','selfemp'],drop_first = True)

credit.head()

sns.boxplot(x='card',y='reports',data=credit,palette='hls')

x=credit.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]

x

y=credit.iloc[:,0]

y

classifier = LogisticRegression()

classifier.fit(x,y)

classifier.coef_

classifier.predict_proba(x)

y_pred=classifier.predict(x)

y_pred

credit['y_pred']=y_pred

credit.head()

y_prob=pd.DataFrame(classifier.predict(x.iloc[:,:]))

y_prob

pred= pd.concat([credit,y_prob],axis=1)

from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y,y_pred)

confusion_matrix

type(y_pred)

accuracy = sum(y==y_pred)/credit.shape[0]

accuracy

pd.crosstab(y_pred,y)

train,test = train_test_split(credit,test_size=0.3)

train

train.shape

test

test.shape

model = LogisticRegression().fit(train.iloc[:,1:],train.iloc[:,0])

train_pred=model.predict(train.iloc[:,1:])

train_pred

test_pred = model.predict(test.iloc[:,1:])

test_pred

from sklearn.metrics import accuracy_score

accuracy_score(train.iloc[:,0],train_pred)

accuracy_score(test.iloc[:,0],test_pred)

