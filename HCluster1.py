import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

airline = pd.read_csv('/home/sushil/Documents/Assingment/Clust-Ass7/EastWestAirlines.csv')

airline.head()

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

def_norm = norm_fun(airline.iloc[:,1:])

def_norm.head()

type(def_norm)

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch

z = linkage(def_norm,method = "complete",metric = "euclidean")

z

plt.figure(figsize = (25,10));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('index')

sch.dendrogram(
                z,
                leaf_rotation=0,
                leaf_font_size=10
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete= AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(def_norm)

cluster_labels=pd.Series(h_complete.labels_)
airline['clust']=cluster_labels

airline=airline.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]

airline

airline.iloc[:,2:].groupby(airline.clust).mean()

