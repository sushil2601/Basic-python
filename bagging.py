import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

wbcd = pd.read_csv('/home/sushil/Desktop/wbcd.csv')

wbcd.head()

wbcd.shape

wbcd.describe()

wbcd = wbcd.drop('id',axis = 1)

wbcd.head()

wbcd['diagnosis'].value_counts()


sns.heatmap(wbcd.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

sns.countplot(wbcd['diagnosis'])

from imblearn.over_sampling import SMOTE

x = wbcd.iloc[:,1:30]

x.head()

y = wbcd.iloc[:,0]

y.head()

x.columns

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

oversampler=SMOTE(random_state=0)
os_x,os_y=oversampler.fit_sample(x_train,y_train)


len(os_y[os_y==1])

oversampler

os_x.shape

os_y.shape

dt = DecisionTreeClassifier()

dt.fit(os_x,os_y)

dt.score(x_test,y_test)

dt.score(os_x,os_y)

rf = RandomForestClassifier(n_estimators = 10)

rf.fit(os_x,os_y)

rf.score(x_test,y_test)

rf.score(os_x,os_y)

bg = BaggingClassifier(DecisionTreeClassifier(),max_samples = 0.5,max_features = 1,n_estimators = 20)

bg.fit(os_x,os_y)

bg.score(x_test,y_test)

bg.score(os_x,os_y)

ada = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators = 20,learning_rate = 0.001)

ada.fit(os_x,os_y)

ada.score(x_test,y_test)

ada.score(os_x,os_y)

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(kernel = 'poly',degree = 2)

evc = VotingClassifier(estimators = [('lr',lr),('dt',dt),('svm',svm)],voting = 'hard')

evc.fit(os_x,os_y)

evc.score(x_test,y_test)

evc.score(os_x,os_y)

