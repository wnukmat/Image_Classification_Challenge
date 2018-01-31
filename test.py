from sklearn import cluster
import numpy as np




# a = np.array([1,1])
# b = np.array([0 1])

# dist = np.linalg.norm(a-b)
# print 'dist: ', dist

def find_majority_class(labels):
	print max(set(labels), key=list(labels).count)
	print labels
	


Positive_Class_Set = [[0,0],[2,2],[1,1],[1,2],[2,1],[43,43],[43,43],[43,44],[44,43],[45,43],[69,70],[71,72],[78,80]]

MoreClusters = True
k = 1
while(MoreClusters == True):
	print 'Clustering: K = ' + str(k)
	MoreClusters = False
	k_means = cluster.KMeans(n_clusters=k)
	k_means.fit(Positive_Class_Set) 
	
	for i in range(len(k_means.labels_)):
		dist = np.linalg.norm(Positive_Class_Set[i]-k_means.cluster_centers_[k_means.labels_[i]] )
		if(dist > 10):
			MoreClusters = True

	k += 1

print 'Classes, Cluster_Centers: ', k_means.labels_, k_means.cluster_centers_
majority_class = max(set(k_means.labels_), key=list(k_means.labels_).count)
print 'majority_class, cluster_center of majority_class: ', majority_class, k_means.cluster_centers_[majority_class]
