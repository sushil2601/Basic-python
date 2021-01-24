import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wine = pd.read_csv('/home/sushil/Documents/Assingment/Pca-Ass8/wine.csv')

wine.head()

wine.corr()

wine.describe()

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

wine.data=wine.iloc[:,1:]

wine.data.head()

wine.data.corr()

WINE=wine.data.values

WINE

wine_normal=scale(WINE)

wine_normal

pca=PCA(n_components=13)

pca_values=pca.fit_transform(wine_normal)

pca_values

var=pca.explained_variance_ratio_

var

pca.components_[0]

var1=np.cumsum(np.round(var,decimals=4)*100)

var1

plt.plot(var1,color='blue')

x=pca_values[:,0:1]

y=pca_values[:,1:2]

z=pca_values[:,2:3]

#plt.scatter(x,y,z,color=["green","blue","red"])

W = wine.data.iloc[:,1:4]

W.head()

W=W.rename(columns={'Malic':'comp.1','Ash':'comp.2','Alcalinity':'comp.3'})

W.head()

cust=pd.concat([wine.data,W],axis=1)

cust.head()

clus_data=cust.iloc[:,13:17]

clus_data.head()

def norm_fun(i):
    z = (i-i.min())/(i.max()-i.min())
    return(z)

clus_norm = norm_fun(clus_data.iloc[:,0:])

clus_norm.head()

type(clus_norm)

# Hierarchical Clustering

from scipy.cluster.hierarchy import linkage

import scipy.cluster.hierarchy as sch

a = linkage(clus_norm,method = "complete",metric = "euclidean")

a

plt.figure(figsize = (25,10));plt.title('Hierarchical Clustering Dendogram');plt.xlabel('index')

sch.dendrogram(
                a,
                leaf_rotation=0,
                leaf_font_size=10
)
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete= AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(clus_norm)

h_complete

cluster_labels=pd.Series(h_complete.labels_)
cust['clust']=cluster_labels

cluster_labels

cust=cust.iloc[:,[16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]

cust.head()

cust.iloc[:,0:].groupby(cust.clust).mean()

# Kmeans Clustering

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

k=list(range(2,20))
k
TWSS=[]
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(clus_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(clus_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,clus_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-');plt.xlabel("no. of clusters");plt.ylabel("total_within_ss");plt.xticks(k)

model=KMeans(n_clusters=5)

model

model.fit(clus_norm)

model.labels_

df=pd.Series(model.labels_)

df

cust['clust']=df

cust

cust.iloc[:,2:].groupby(cust.clust).mean()

