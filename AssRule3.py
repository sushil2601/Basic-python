import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

books = pd.read_csv('/home/sushil/Documents/Assingment/AR-9/book.csv')

books

from mlxtend.frequent_patterns import association_rules,apriori

items = apriori(books,min_support=0.005,max_len=3,use_colnames=True)

items

items.sort_values('support',ascending = False,inplace=True)

items

rules = association_rules(items,metric='lift',min_threshold=1)

rules

rules.head()

z=rules.sort_values('lift',ascending=False)

z

import seaborn as sns

plt.boxplot(rules.support)

plt.hist(rules.support)

plt.hist(rules.lift)

sns.distplot(rules.lift)

sns.boxplot(rules.iloc[:,2])

item2 = apriori(books,min_support=0.001,max_len=3,use_colnames=True)

item2

item2.sort_values('support',ascending = False,inplace=True)

item2

rule2 = association_rules(items,metric='lift',min_threshold=2)

rule2

w=rule2.sort_values('lift',ascending=False)

w

item3 = apriori(books,min_support=0.004,max_len=3,use_colnames=True)

item3

item3.sort_values('support',ascending = False,inplace=True)

item3.head()

rule3 = association_rules(items,metric='lift',min_threshold=3)

rule3.head()

u=rule3.sort_values('lift',ascending=False)

u

