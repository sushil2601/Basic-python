import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

movies = pd.read_csv('/home/sushil/Documents/Assingment/AR-9/my_movies.csv')

movies.head()

from mlxtend.frequent_patterns import association_rules, apriori

items1 = apriori(movies.iloc[:, 5:], min_support = 0.005, max_len = 3, use_colnames = True)

items1.head()

sns.distplot(items1.support)

sns.countplot(items1.support)

items1.sort_values('support', ascending = False, inplace = True)

items1.head()

items1.tail()

rules1 = association_rules(items1, metric = 'lift', min_threshold = 1)

rules1.head()

rules1.sort_values('lift', inplace = True)

rules1.head()

sns.distplot(rules1.lift)

plt.figure(figsize = (30,10))
sns.countplot(rules1.lift)

items2 = apriori(movies.iloc[:, 5:], max_len = 3, min_support = 0.1, use_colnames = True)

items2.head()

items2.sort_values('support', ascending = False, inplace = True)

items2.head()

sns.countplot(items2.support)

rules2 = association_rules(items2, metric = 'lift', min_threshold = 1.42)

rules2.head()

rules2.sort_values('lift', ascending = False, inplace = True)

rules2.tail()

plt.figure(figsize = (30, 10))
sns.countplot(rules2.lift)

rules3 = association_rules(items2, metric = 'lift', min_threshold  = 2.5)

rules3.head()

rules3.sort_values('lift', ascending = False, inplace = True)

rules3.head()

rules3.count()

sns.countplot(rules3.lift)

