import pandas as pd
df = pd.DataFrame({'city':['Belgium','Hyderabad','Newyork','Vizag'],'pollution':[108,235,98,39]})
df
df.columns
df['city'].unique()
df['city'].nunique()
df['pollution'].unique()
df['pollution'].nunique()
total_rows = len(df.axes[0])
total_rows
total_cols = len(df.axes[1])
total_cols
df.shape
df.describe()
df.info()
df.dtypes
df.columns = ['CITY','POLLUTIUON']
df.columns
df
df.columns = df.columns.str.lower()
df
df.columns = df.columns.str.upper()
df


import numpy as np
ufo = pd.read_csv('https://bit.ly/uforeports')
ufo
ufo.head()
ufo.tail()
ufo.shape
ufo.describe()
ufo.info()
ufo.dtypes
ufo.isnull()
ufo.isnull().sum()
ufo.dropna(subset = ['City','Shape Reported'],inplace = True)
#ufo
ufo.dropna(subset = ['City','Shape Reported'],inplace = False)
ufo.isnull().sum()

ufo.fillna(value = 'UNKNOWN',axis =1,inplace = False)
ufo.fillna(value = 'UNKNOWN',axis =1,inplace = True)

ufo['Colors Reported'].fillna(value = 'New')
ufo

movies = pd.read_csv('https://bit.ly/imdbratings')
movies
drinks = pd.read_csv('https://bit.ly/drinksbycountry')
drinks
#movies
movies.head()
drinks.head()
drinks.columns

drinks.set_index('country',inplace = True)
drinks
drinks.head()
drinks.reset_index(inplace = False)
drinks.reset_index(inplace = True)
drinks
drinks.head()
drinks.shape
drinks['continent']
drinks['continent'].unique()
drinks['continent'].nunique()

drinks.loc[5:29,:]
drinks.loc[:,'beer_servings':'wine_servings'].head()
drinks.iloc[[0,1,2],[2,3]]
#drinks.ix[5:29,'beer_servings'].head()

drinks.head()
drinks.columns
drinks.groupby(['continent','country']).beer_servings.max()

drinks.groupby('country').beer_servings.agg(['max','min','count','mean'])
drinks.groupby('country').beer_servings.agg(['max'])
drinks['beer_servings'] = drinks.beer_servings.astype(float)
drinks.dtypes

train = pd.read_csv('https://bit.ly/titanic-train')
train
train.Embarked.value_counts()
train.Embarked.unique()
train.Embarked.nunique()
train.shape
train.Sex.value_counts()
train['Sex'] = train.Sex.map({'male':0,'female':1})
train.Sex.value_counts()
train.dtypes
train.head()
#pd.get_dumies(trian.Embarked,prefix = 'Embark').head()

import numpy as np
drinks.beer_servings.head()
drinks.beer_servings.apply(np.sqrt).head()
drinks.wine_servings.head()
drinks.wine_servings.apply(lambda x : x+(x*x)-(4*x)).head()
ufo.head()
ufo.applymap(lambda x : len(str(x)))