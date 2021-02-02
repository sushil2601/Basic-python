import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

salary_train = pd.read_csv('/home/sushil/Documents/Assingment/SVM-17/SalaryData_Train(1).csv')

salary_train.head()

salary_test = pd.read_csv('/home/sushil/Documents/Assingment/SVM-17/SalaryData_Test(1).csv')

salary_test.head()

salary_train['Salary'].value_counts()

salary_test['Salary'].value_counts()

salary_train.shape

salary_test.shape

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'workclass', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'workclass', data = salary_test)

sns.boxplot(y = 'Salary' , x = 'age',data = salary_train , palette = 'hls')

sns.boxplot(y = 'Salary' , x = 'educationno' , data = salary_train , palette = 'hls')

sns.boxplot(y = 'Salary' , x = 'hoursperweek' , data = salary_train , palette = 'hls')

plt.hist(salary_train['Salary'])

sns.stripplot(x = 'Salary', y = 'age', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'workclass', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'education', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'maritalstatus', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'occupation', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'relationship', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'race', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'sex', data = salary_train)

plt.figure(figsize = (10,6))
sns.countplot('Salary', hue = 'native', data = salary_train)

sns.heatmap(salary_train.isnull(), cmap = 'viridis', yticklabels = False)

salary_train.isnull().sum()

sns.heatmap(salary_test.isnull(), cmap = 'viridis', yticklabels = False)

salary_test.isnull().sum()

salary_train_new = pd.get_dummies(salary_train.iloc[:, [1,2,4,5,6,7,8,12]], drop_first = True)

salary_train_new.head()

salary_train_new.count()

salary_train_new = pd.concat([salary_train_new, salary_train.iloc[:, [0,3,9,10,11,13]]], axis = 1)

salary_train_new.head()

salary_test_new = pd.get_dummies(salary_test.iloc[:, [1,2,4,5,6,7,8,12]], drop_first = True)

salary_test_new = pd.concat([salary_test_new, salary_test.iloc[:, [0,3,9,10,11,13]]], axis = 1)

salary_test_new.head()

x = salary_train_new.drop('Salary',axis = 1)

x.head()

x.columns

x.shape

y = salary_train_new['Salary']

y.head()

y.shape

from sklearn.feature_selection import SelectKBest, chi2

best_features = SelectKBest(score_func = chi2, k = 20)

best_features

fit = best_features.fit(x,y)

fit

scores = pd.DataFrame(fit.scores_)

scores.head()

column_names = pd.DataFrame(x.columns)

column_names.head()

feature_scores = pd.concat([column_names, scores], axis = 1)

feature_scores.head()

feature_scores.columns = ['Features', 'Scores']

feature_scores.columns

print(feature_scores.nlargest(10, 'Scores'))

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(x,y)

feat_importances = pd.Series(model.feature_importances_, index = x.columns)

feat_importances

feat_importances.nlargest(10).plot(kind = 'barh')

from imblearn.over_sampling import SMOTE

sm = SMOTE()

x_train = salary_train_new[['educationno', 'maritalstatus_ Never-married', 'capitalgain', 'maritalstatus_ Married-civ-spouse', 'hoursperweek', 'age']]

x_train.head()

y_train = salary_train_new['Salary']

x_test = salary_test_new[['educationno', 'maritalstatus_ Never-married', 'capitalgain', 'maritalstatus_ Married-civ-spouse', 'hoursperweek', 'age']]

x_test.head()

y_test = salary_test_new['Salary']

x_train, y_train = sm.fit_resample(x_train, y_train)

y_train.value_counts()

sns.countplot(y_train)

from sklearn.svm import SVC

model_poly = SVC(kernel = 'poly')

model_poly.fit(x_train,y_train)

model_poly.score(x_train,y_train)

pred_test_poly = model_poly.predict(x_test)

pred_test_poly

np.mean(pred_test_poly == y_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(pred_test_poly , y_test))

print(confusion_matrix(pred_test_poly , y_test))

######## model_poly is underfitted model ###########

model_rbf = SVC(kernel = 'rbf')

model_rbf.fit(x_train,y_train)

pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf == y_test)

model_rbf.score(x_train,y_train)

print(classification_report(pred_test_rbf , y_test))

print(confusion_matrix(pred_test_rbf , y_test))

######## Model_rbf is also underfitted #########

model_linear = SVC(kernel = 'linear')

model_linear.fit(x_train,y_train)

pred_test_line = model_linear.predict(x_test)

np.mean(pred_test_line == y_test)

model_linear.score(x_train , y_train)

print(classification_report(pred_test_line , y_test))

print(confusion_report(pred_test_line , y_test))

