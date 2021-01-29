import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

glass = pd.read_csv('glass.csv')

glass.head()

sns.heatmap(glass.isnull(), cmap = 'viridis', cbar = False, yticklabels = False)

glass.shape

sns.countplot(glass['Type'])

glass['Type'].value_counts()

plt.figure(figsize = (20,6))
sns.stripplot(x = glass['Ba'], y = glass['Fe'])

sns.boxplot(glass['Al'])

sns.boxplot(glass['Si'])

sns.boxplot(glass['Mg'])

sns.boxplot(glass['Na'])

sns.boxplot(glass['RI'])

sns.boxplot(glass['K'])

sns.boxplot(glass['Ba'])

sns.boxplot(glass['Fe'])

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

x = glass.iloc[:, 0:9]

x.head()

y = glass.iloc[:, 9]

y.head()

bestfeatures = SelectKBest(score_func = chi2, k = 6)

fit = bestfeatures.fit(x,y)

scores = pd.DataFrame(fit.scores_)

column_names = pd.DataFrame(x.columns)

featureScores = pd.concat([column_names, scores], axis = 1)

featureScores.columns = ['Features', 'Scores']

print(featureScores.nlargest(10, 'Scores'))

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(x,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index = x.columns)

feat_importances.nlargest(9).plot(kind = 'barh')

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

glass1 = glass

x2 = glass1.drop(['Type', 'Fe', 'Ba', 'Si', 'Na'], axis = 1)

y2 = glass1['Type']

x2.head()

y2.head()

x_train, x_test, y_train, y_test = train_test_split(x2,y2, test_size = 0.25, random_state = 27)

sm = SMOTE(k_neighbors = 4, random_state = 27)

x_train.head()

y_train.head()

x_train, y_train = sm.fit_resample(x_train, y_train)

from sklearn.neighbors import KNeighborsClassifier as KNC

acc = []

for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(x_train, y_train)
    train_acc = np.mean(neigh.predict(x_train) == y_train)
    test_acc = np.mean(neigh.predict(x_test) == y_test)
    acc.append([train_acc, test_acc])

plt.figure(figsize = (10,7))
plt.plot(np.arange(3,50,2), [i[0] for i in acc], 'ro-')
plt.plot(np.arange(3,50,2), [i[1] for i in acc], 'bo-')
plt.legend(["train", "test"])

neigh = KNC(n_neighbors = 5)

neigh.fit(x_train, y_train)

train_acc = np.mean(neigh.predict(x_train) == y_train)
test_acc = np.mean(neigh.predict(x_test) == y_test)

train_acc

test_acc

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_train, neigh.predict(x_train)))


print(classification_report(y_test, neigh.predict(x_test)))


glass2 = glass 

x3 = glass2.drop(['Na', 'Ca', 'Fe', 'Si', 'RI'], axis = 1)

x3.drop('Type', axis = 1, inplace = True)

x3.tail()

y3 = glass2['Type']

y3.head()

x2_train, x2_test, y2_train, y2_test = train_test_split(x3,y3, test_size = 0.25, random_state = 27)

sm2 = SMOTE(k_neighbors = 4, random_state = 27)

x2_train, y2_train = sm.fit_resample(x2_train, y2_train)

acc2 = []

for i in range(3,50,2):
    neigh = KNC(n_neighbors = i)
    neigh.fit(x2_train, y2_train)
    train_acc2 = np.mean(neigh.predict(x2_train) == y2_train)
    test_acc2 = np.mean(neigh.predict(x2_test) == y2_test)
    acc2.append([train_acc2, test_acc2])

plt.figure(figsize = (10,7))
plt.plot(np.arange(3,50,2), [i[0] for i in acc2], 'ro-')
plt.plot(np.arange(3,50,2), [i[1] for i in acc2], 'bo-')
plt.legend(["train", "test"])

neigh = KNC(n_neighbors = 1)

neigh.fit(x2_train, y2_train)

train_acc = np.mean(neigh.predict(x2_train) == y2_train)
test_acc = np.mean(neigh.predict(x2_test) == y2_test)

test_acc

train_acc

print(classification_report(y_train, neigh.predict(x_train)))
