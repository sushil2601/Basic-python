import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

crime = pd.read_csv('/home/sushil/Documents/Assingment/Clust-Ass7/crime_data.csv')

crime.head()

def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

def_norm = norm_fun(crime.iloc[:,1:])

def_norm.head()

type(def_norm)

k=list(range(2,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(def_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(def_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,def_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("total_within_ss");plt.xticks(k)

model=KMeans(n_clusters=5)

model

model.fit(def_norm)

model.labels_

df=pd.Series(model.labels_)

df.head()

crime['clust']=df

crime.head()

crime.iloc[:,2:].groupby(cri.clust).mean()