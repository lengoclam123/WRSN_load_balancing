
import numpy as np
from sklearn.cluster import KMeans
import math

from sklearn.metrics import silhouette_score
from Cluster import Cluster

# INPUT: List of the given targets from the network
# [target1, target2, ...]

# OUTPUT: List of clusters of targets
# [cluster1, cluster2, ...]

def clustering(net):

        def check_valid_cluster(clusterer, data, n_clusters, n_targets):

            # mảng các label, mỗi label cluster là mảng các node
            cluster_labels = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
            cluster = [[] for _ in range(n_clusters)]
            for index_node in range(0, n_targets):
                cluster[cluster_labels[index_node]].append(data[index_node])

            for index_cluster in range(0, n_clusters):
                for target in cluster[index_cluster]: 
                    if math.dist(target, centers[index_cluster]) > 140: 
                        return False    
            return True

        # Input : listTargets
        list_targets_location = []
        for target in net.listTargets:
            list_targets_location.append(target.location)
        list_targets_location = np.array(list_targets_location)
        range_n_clusters = np.arange(2, int(len(list_targets_location)/2))
        
        max_silhouette_score = 0
        n_clusters_optimal = 0

        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, init="k-means++",  algorithm='elkan')

            cluster_labels = clusterer.fit_predict(list_targets_location)
            silhouette_avg = silhouette_score(list_targets_location, cluster_labels)

            if check_valid_cluster(clusterer, list_targets_location, n_clusters, len(list_targets_location)) is True:
                if max_silhouette_score < silhouette_avg:
                    max_silhouette_score = silhouette_avg
                    n_clusters_optimal = n_clusters

        clusterer = KMeans(n_clusters=n_clusters_optimal, init="k-means++",  algorithm='elkan')
        # clusterer = KMeans(n_clusters=n_clusters_optimal, random_state=None,  n_init="auto", init="k-means++", algorithm='lloyd')

        cluster_labels = clusterer.fit_predict(list_targets_location)
        silhouette_avg = silhouette_score(list_targets_location, cluster_labels)
        net.n_clusters_optimal = n_clusters_optimal
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_

        clusters = []
        
        for i in range(0, n_clusters_optimal):
            listTargetsInCluster = []
            for j in range(0, len(list_targets_location)):
                if cluster_labels[j] == i:
                    listTargetsInCluster.append(net.listTargets[j])
            cluster = Cluster(i, listTargetsInCluster, centers[i])
            clusters.append(cluster)
        print("Num cluster:", len(clusters))

        return clusters

