import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo = pd.read_csv('/home/sushil/Documents/Assingment/KNN-13/Zoo.csv')

zoo.head()

zoo.shape

zoo.describe()

zoo['animal name'].unique

zoo['animal name'].nunique()

zoo['animal name'].value_counts()

sns.countplot(zoo['animal name'])

from sklearn.model_selection import train_test_split

train,test = train_test_split(zoo,test_size=0.2)

train.head()

train.shape

test.shape

from sklearn.neighbors import KNeighborsClassifier as knc

neigh = knc(n_neighbors = 3)

neigh

neigh.fit(train.iloc[:,2:17],train.iloc[:,17])

train_acc = np.mean(neigh.predict(train.iloc[:,2:17])==train.iloc[:,17])

train_acc

test_acc = np.mean(neigh.predict(test.iloc[:,2:17])==test.iloc[:,17])

test_acc

neigh1 = knc(n_neighbors = 5)

neigh1.fit(train.iloc[:,2:17],train.iloc[:,17])

train_acc1 = np.mean(neigh1.predict(train.iloc[:,2:17])==train.iloc[:,17])

train_acc1

test_acc1 = np.mean(neigh1.predict(test.iloc[:,2:17])==test.iloc[:,17])

test_acc1

neigh2 = knc(n_neighbors = 7)

neigh2.fit(train.iloc[:,2:17],train.iloc[:,17])

train_acc2 = np.mean(neigh2.predict(train.iloc[:,2:17])==train.iloc[:,17])

train_acc2

test_acc2 = np.mean(neigh2.predict(test.iloc[:,2:17])==test.iloc[:,17])

test_acc2

train1,test1 = train_test_split(zoo,test_size=0.4)

train1.shape

test1.shape

neigh3 = knc(n_neighbors = 3)

neigh3.fit(train1.iloc[:,2:17],train1.iloc[:,17])

train_acc3 = np.mean(neigh3.predict(train1.iloc[:,2:17])==train1.iloc[:,17])

train_acc3

test_acc3 = np.mean(neigh3.predict(test1.iloc[:,2:17])==test1.iloc[:,17])

test_acc3

neigh4 = knc(n_neighbors = 5)

neigh4.fit(train1.iloc[:,2:17],train1.iloc[:,17])

train_acc4 = np.mean(neigh4.predict(train1.iloc[:,2:17])==train1.iloc[:,17])

train_acc4

test_acc4 = np.mean(neigh4.predict(test1.iloc[:,2:17])==test1.iloc[:,17])

test_acc4

acc = []
for i in range(3,50,2):
    neigh = knc(n_neighbors = i)
    neigh.fit(train.iloc[:,2:17],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,2:17])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,2:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])

acc

plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
plt.legend(['train','test'])

acc1 = []
for i in range(3,50,2):
    neigh1 = knc(n_neighbors = i)
    neigh1.fit(train1.iloc[:,2:17],train1.iloc[:,17])
    train_acc1 = np.mean(neigh1.predict(train1.iloc[:,2:17])==train1.iloc[:,17])
    test_acc1 = np.mean(neigh1.predict(test1.iloc[:,2:17])==test1.iloc[:,17])
    acc1.append([train_acc1,test_acc1])

acc1

plt.plot(np.arange(3,50,2),[i[0] for i in acc1],"ro-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc1],"bo-")
plt.legend(['train','test'])

from sklearn.metrics import classification_report

classification_report(train.iloc[:,17],neigh.predict(train.iloc[:,2:17]))

classification_report(train1.iloc[:,17],neigh1.predict(train1.iloc[:,2:17]))

