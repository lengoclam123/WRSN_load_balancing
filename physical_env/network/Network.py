import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import math
from scipy.signal import find_peaks

from Cluster import Cluster

class Network:
    def __init__(self, env, baseStation, listTargets, max_time):
        self.env = env

        self.baseStation = baseStation
        self.listTargets = listTargets
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1

        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time

        # it = 0
        # for node in self.listNodes:
        #     node.env = self.env
        #     node.net = self
        #     node.id = it
        #     it += 1
           
        it = 0

        for target in listTargets:
            target.id = it
            it += 1
        

        self.listNodes = []
        self.listClusters = []
        self.listEdges = []

        self.createNodes()
    
    # Function is for setting nodes' level and setting all targets as covered
    def createNodes(self):
        self.listClusters = self.clustering()
        # self.listEdges = self.createEdges(self.listClusters)

        # nodeInsideCluster = self.createNodeInCluster(self.listClusters, self.listEdges)
        # nodeBetweenCluster = self.createNodeBetweenCluster(self.listEdges)

        # self.listNodes = nodeBetweenCluster + nodeInsideCluster
        pass

    def clustering(self):
        # Input : listTargets
        listTargetLocation = []
        for target in self.listTargets:
            listTargetLocation.append(target.location)

        # Elbow's method applying gradient rule to find number of clusters K
        inertias = []
        for k in range(1, len(listTargetLocation) + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(listTargetLocation)
            inertias.append(kmeans.inertia_)

        gradient = np.gradient(inertias)
        peaks, _ = find_peaks(gradient)
        optimal_cluster = peaks[0] + 1
        
        kmeans = KMeans(optimal_cluster)
        kmeans.fit(listTargetLocation)
        centers = kmeans.cluster_centers_

        # temporary drawing process
        x = []
        y = []
        for targetLocation in listTargetLocation:
            x.append(targetLocation[0])
            y.append(targetLocation[1])

        plt.scatter(x, y, c=kmeans.labels_)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=20)  # Vẽ các tâm của mỗi cluster

        plt.show()

        # Output : [Cluster1,Cluster2 , . . . ]
        clusters = []
        labels = kmeans.labels_
        
        for i in range(0, optimal_cluster):
            listTargetsInCluster = []
            for j in range(0, len(listTargetLocation)):
                if labels[j] == i:
                    listTargetsInCluster.append(self.listTargets[j])
            cluster = Cluster(i, listTargetsInCluster, centers[i])
            clusters.append(cluster)
        
        for cluster in clusters:
            print(cluster.id)
            print(cluster.centroid)
            print(cluster.listTargets)
        return clusters
    
    
    def createEdges(self):
        # Input 
            # [Cluster1,Cluster2 , . . . ]

        # Output 
            # [(1,2),(3,2),(4,5) ,  . . .]

        pass

    def createNodeInCluster(self):
        # Input 
            # self.listClusters, self.listEdges

        # Todo
            # for cluster in self.listClusters:
            #     cluster.listNodes = [Node1,Node2 , . . . ]
        pass

    def createNodeBetweenCluster(self):
        
        # Input 
            # self.listEdges

        # Todo
            # add start Id and End Id for relayNode

        # Output
            # [relayNode1,relayNode2 , . . . ]
        pass


        


    def setLevels(self):
        for node in self.listNodes:
            node.level = -1
        tmp1 = []
        tmp2 = []
        for node in self.baseStation.direct_nodes:
            if node.status == 1:
                node.level = 1
                tmp1.append(node)

        for i in range(len(self.targets_active)):
            self.targets_active[i] = 0

        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                for target in node.listTargets:
                    self.targets_active[target.id] = 1
                for neighbor in node.neighbors:
                    if neighbor.status == 1 and neighbor.level == -1:
                        tmp2.append(neighbor)
                        neighbor.level = node.level + 1

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        return

    def operate(self, t=1):
        
        for node in self.listNodes:
            self.env.process(node.operate(t=t))
        self.env.process(self.baseStation.operate(t=t))
        
        while True:
            yield self.env.timeout(t / 10.0)
            self.setLevels() #
            self.alive = self.check_targets()
            yield self.env.timeout(9.0 * t / 10.0)
            if self.alive == 0 or self.env.now >= self.max_time:
                break         
        return

    def check_targets(self):
        return min(self.targets_active)
    
    def check_nodes(self):
        tmp = 0
        for node in self.listNodes:
            if node.status == 0:
                tmp += 1
        return tmp
