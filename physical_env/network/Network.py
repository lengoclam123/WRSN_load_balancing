import copy
import numpy as np
import Cluster
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from itertools import cycle

class Network:
    def __init__(self, env, listNodes, baseStation, listTargets, max_time):
        self.env = env
        # self.listNodes = listNodes

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
        

        self.listNodes = None
        self.listClusters = None
        self.listEdges = None


        self.createNodes()

    
    # Function is for setting nodes' level and setting all targets as covered
    def createNodes(self):
        self.listClusters = self.clustering()
        self.listEdges = self.createEdges(self.listClusters)

        nodeInsideCluster = self.createNodeInCluster(self.listClusters, self.listEdges)
        nodeBetweenCluster = self.createNodeBetweenCluster(self.listEdges)

        self.listNodes = nodeBetweenCluster + nodeInsideCluster

    def clustering(self):
        # Input :
            # listTargets

        # Todo :
            # cluster
            # centroid
            # define id 

        # Output :
            # [Cluster1,Cluster2 , . . . ]
        return None
    


    
    def createEdges(self):
        # Input 
            # [Cluster1,Cluster2 , . . . ]

        # Output 
            # [(1,2),(3,2),(4,5) ,  . . .]
            # funs for create edges
        def calDistanceBS(cluster):
            distance =  np.sqrt((cluster['centroid'][0] - 500)**2 + (cluster['centroid'][1] - 500)**2)
            return distance

        def calDistanceCluster(cluster1, cluster2):
            distance =  np.sqrt(np.sum((np.array(cluster1['centroid']) - np.array(cluster2['centroid']))**2))
            return distance

        def nearest_cluster_neighbor(cluster, listClusters, kdtree):
            nearest_neighbor_idx = kdtree.query_ball_point(cluster['centroid'], calDistanceBS(cluster))
            if len(nearest_neighbor_idx) > 1:
                nearest_neighbor_idx.remove(listClusters.index(cluster))
            if nearest_neighbor_idx:
                nearest_neighbor = listClusters[nearest_neighbor_idx[0]]
                return nearest_neighbor
            else:
                return None

        cluster_centroids = [cluster['centroid'] for cluster in self.listClusters]
        kdtree = cKDTree(cluster_centroids)

        edges = []
        for cluster in self.listClusters:
            nearest_neighbor = nearest_cluster_neighbor(cluster, self.listClusters, kdtree)
            if nearest_neighbor:
                edges.append((cluster, nearest_neighbor))
        
        edge_colors = cycle(['g', 'b', 'y', 'c', 'm', 'k'])

        # Vẽ các điểm trong các cluster và điểm centroid
        cluster_colors = ['g', 'b', 'y', 'c', 'm', 'k']

        for i, cluster_data in enumerate(self.listClusters):
            color = cluster_colors[i % len(cluster_colors)]  
            points = np.array([target["location"] for target in cluster_data["listTargets"]])
            centroid = np.array(cluster_data["centroid"])
            plt.scatter(points[:, 0], points[:, 1], color=color)
            plt.scatter(centroid[0], centroid[1], color='red')

        for edge, color in zip(edges, edge_colors):
            x_values = [edge[0]['centroid'][0], edge[1]['centroid'][0]]
            y_values = [edge[0]['centroid'][1], edge[1]['centroid'][1]]
            plt.plot(x_values, y_values, color=color)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        with open(edges.json, "w") as output_file:
            json.dump(edges, output_file)
        self.listEdges = edges
        return edges


       

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
