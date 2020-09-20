import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric

def kdist(i,arr,K):
	k_neigh=dist_mat[i].argsort()[:K+1]
	k_neigh=np.setdiff1d(k_neigh,[i])
	return arr[i][k_neigh[-1]]

def kneigh(i,arr,K):
	k_neigh=dist_mat[i].argsort()[:K+1]
	k_neigh=np.setdiff1d(k_neigh,[i])
	return k_neigh

def reachdist(p,o,arr,K):
	return max(kdist(o,arr,K),arr[p][o])

def lrd(i, arr, K):
	k_neigh=kneigh(i,arr,K)
	temp=0
	for n in k_neigh:
		temp+=reachdist(i,n,arr,K)
	return temp/K
		

#dataset with 50 inliers and 10 outliers
df=pd.read_csv("iris.csv",header=None)
df1=df[df[4]=='Iris-virginica']
df2=df[df[4]=='Iris-setosa'].sample(50).head(n=2)
df3=df[df[4]=='Iris-versicolor'].sample(50).head(n=2)
df1=(df1.append(df2)).append(df3)
print(df1[50:54])
#df1=df1.sample(len(df1))

#LOF algo considering all points are available
X=df1[[0,1,2,3]]
dist=DistanceMetric.get_metric('euclidean')
dist_mat=np.array(dist.pairwise(X))
for K in range(5,20):
	LOF=[]
	for i in range(0,len(dist_mat)):
		lof=0
		k_neigh=kneigh(i, dist_mat, K)
	#	print(k_neigh)
		for n in k_neigh:
			lof+=lrd(n, dist_mat, K)
		lof=lof/K/lrd(i, dist_mat, K)
		LOF.append(lof)
	LOF=np.array(LOF)
	print(K)
	print(LOF.argsort())
	print(LOF)
