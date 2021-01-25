import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules

groceries = []
with open('/home/sushil/Documents/Assingment/AR-9/groceries1.csv') as f:
    groceries = f.read()

groceries

groceries = groceries.split('\n')

groceries

groceries_list=[]

for i in groceries:
    groceries_list.append(i.split(","))
all_groceries_list=[i for item in groceries_list for i in item]

from collections import Counter,OrderedDict

item_frequencies=Counter(all_groceries_list)

item_frequencies

type(item_frequencies)

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])

item_frequencies

frequencies = list(reversed([i[1] for i in item_frequencies]))

frequencies

items = list(reversed([i[0] for i in item_frequencies]))

items

import seaborn as sns
plt.figure(figsize = (24,8))
sns.stripplot(x = items[1:10], y = frequencies[1:10])

groceries_series = pd.DataFrame(pd.Series(groceries_list))

groceries_series

groceries_series = groceries_series.iloc[:9835,:]

groceries_series

groceries_series.columns = ["transaction"]

X = groceries_series['transaction'].str.join(sep='*').str.get_dummies(sep='*')

X

frequent_itemsets = apriori(x,min_support = 0.005,max_len = 3,use_colnames = True)

frequent_itemsets.head()

frequent_itemsets.sort_values('support',ascending = False,inplace = True)

plt.figure(figsize = (24,8))
sns.stripplot(x = items[1:10], y = frequent_itemsets.support[1:10])

rules = association_rules(frequent_itemsets,metric="lift",min_threshold=3)

rules.head(10)

z=rules.sort_values('lift',ascending=False)

z.head()

