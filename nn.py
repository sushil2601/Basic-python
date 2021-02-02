import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ff = pd.read_csv('/home/sushil/forestfires.csv')

ff.head()

ff.describe()

ff.shape

ff.columns

ff.corr().head()

ff.isnull().sum()

ff['size_category'].value_counts()

sns.heatmap(ff.isnull(), cmap = 'viridis', yticklabels = False)

plt.figure(figsize = (12,8))
sns.countplot('month', hue = 'size_category', data = ff)

sns.stripplot(x = 'size_category', y = 'rain', data = ff)

sns.stripplot(x = 'size_category', y = 'temp', data = ff)

sns.stripplot(x = 'size_category' , y = 'wind',data = ff)

sns.boxplot(ff['temp'])

sns.boxplot(ff['wind'])

sns.boxplot(ff['area'])

sns.boxplot(ff['DMC'])

sns.boxplot(ff['DC'])

sns.boxplot(ff['ISI'])

fire = ff.drop(['month', 'day'], axis = 1)

fire.head()

x = fire.drop('size_category', axis = 1)

x.head()

y = fire['size_category']

y.head()

x = x[x.apply(lambda i: np.abs(i - i.mean()) / i.std() < 3).all(axis=1)]

x.head()

fire = pd.concat([x,y], axis = 1)

fire.head()

fire = fire.dropna(axis = 0)

fire.head()

X =  fire.drop('size_category', axis = 1)

X.head()

X.shape

Y = fire['size_category']

Y.head()

Y.shape

from sklearn.preprocessing import StandardScaler

scaled_features = StandardScaler().fit_transform(X)

scaled_features

X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)

X.head()

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()

model.fit(X,Y)

feat_importances = pd.Series(model.feature_importances_, index = X.columns)

feat_importances.head()

feat_importances.nlargest(18)

feat_importances.nlargest(20).plot(kind = 'barh')

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

x = X[['area', 'temp', 'RH', 'wind', 'FFMC', 'ISI', 'DMC', 'DC', 'daysat', 'dayfri', 'daytue', 'daysun', 'daywed', 'monthaug', 'monthsep','daymon', 'daythu']]      

x.head()

Y.head()

fire2 = pd.concat([x,Y], axis = 1)

fire2.head()

fire2.loc[fire.size_category == 'small', 'size_category'] = 0
fire2.loc[fire.size_category == 'large', 'size_category'] = 1

fire2.head()

train,test = train_test_split(fire2,test_size = 0.3,random_state=42)

trainX = train.drop(["size_category"],axis=1)
trainY = train["size_category"]
testX = test.drop(["size_category"],axis=1)
testY = test["size_category"]

from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda

def prep_model(hidden_dim):
    model = Sequential()
    for i in range(1,len(hidden_dim)-1):
        if (i==1):
            model.add(Dense(hidden_dim[i],input_dim=hidden_dim[0],activation="relu"))
        else:
            model.add(Dense(hidden_dim[i],activation="relu"))
    model.add(Dense(hidden_dim[-1],kernel_initializer="normal",activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer = "rmsprop",metrics = ["accuracy"])
    return model    

first_model = prep_model([17,50,40,20,1])
first_model.fit(np.array(trainX),np.array(trainY),epochs=500)

pred_train = first_model.predict(np.array(trainX))

pred_train

pred_train = pd.Series([i[0] for i in pred_train])

pred_train

category = ['small', 'large']

pred_train_category = pd.Series(["small"]*277)

pred_train_category

pred_train_category[[i>0.5 for i in pred_train]] = "large"

train["original_category"] = "small"

train.drop('original_class', axis = 1, inplace = True)

train.loc[train.size_category == 1,"original_category"] = "large"

train.original_category.value_counts()

from sklearn.metrics import classification_report, confusion_matrix

confusion_matrix(pred_train_category,train.original_category)

print(classification_report(pred_train_category, train.original_category))

np.mean(pred_train_category==pd.Series(train.original_category).reset_index(drop=True))

pd.crosstab(pred_train_category,pd.Series(train.original_category).reset_index(drop=True))

pred_test = first_model.predict(np.array(testX))
pred_test = pd.Series([i[0] for i in pred_test])

pred_test

pred_test_class = pd.Series(["small"]*119)
pred_test_class[[i>0.5 for i in pred_test]] = "large"
test["original_category"] = "small"
test.loc[test.size_category==1,"original_category"] = "large"

test

test.original_category.value_counts()

np.mean(pred_test_class==pd.Series(test.original_category).reset_index(drop=True))

print(classification_report(pred_test_class, test.original_category))

confusion_matrix(pred_test_class, test.original_category)

