
import copy
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import math
from scipy.signal import find_peaks
import json
import copy

from scipy.spatial import cKDTree
from itertools import cycle

from Cluster import Cluster
from Nodes.Node import Node
from Nodes.InNode import InNode
from Nodes.OutNode import OutNode
from Nodes.RelayNode import RelayNode
from Nodes.ConnectorNode import ConnectorNode
from Nodes.SensorNode import SensorNode

# ,OutNode,RelayNode,SensorNode
from utils.PointBetween import point_between
import warnings
warnings.filterwarnings("ignore")

class Network:
    def __init__(self, env, baseStation, listTargets ,max_time,phy):
        self.env = env

        self.baseStation = baseStation
        self.listTargets = listTargets
        self.phy = phy
        self.targets_active = [1 for _ in range(len(self.listTargets))]
        self.alive = 1

        self.Alpha = 0.8

        baseStation.env = self.env
        baseStation.net = self
        self.max_time = max_time

           
        it = 0

        for target in listTargets:
            target.target_id = it
            it += 1
        

        self.listNodes = []
        self.listClusters = []
        self.listEdges = []
        self.createNodes()
        for cluster in self.listClusters:
            for node in cluster.listNodes:
                    node.cluster_id = cluster.id
    
    def createNodes(self):
        self.listClusters = self.clustering()
        self.listEdges = self.createEdges()

        nodeInsideCluster = self.createNodeInCluster()
        nodeBetweenCluster = self.createNodeBetweenCluster()

        self.listNodes = nodeBetweenCluster + nodeInsideCluster
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
        optimal_cluster = peaks[0] + 5
        #optimal_cluster = 19

        
        kmeans = KMeans(optimal_cluster)
        kmeans.fit(listTargetLocation)
        centers = kmeans.cluster_centers_

        # temporary drawing process
        x = []
        y = []
        for targetLocation in listTargetLocation:
            x.append(targetLocation[0])
            y.append(targetLocation[1])

        # plt.scatter(x, y, c=kmeans.labels_)
        # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='o', s=20)  # Vẽ các tâm của mỗi cluster
        # plt.show()

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


        return clusters
    
    def createEdges(self):
        # Input 
            # [Cluster1,Cluster2 , . . . ]

        # Output 
            # [(1,2),(3,2),(4,5) ,  . . .]
            # [(cluster1, cluster2), (cluster, base station),  . . . ]
            # funs for create edges
        def calDistanceBS(cluster):
            distance =  np.sqrt((cluster.centroid[0] - 500)**2 + (cluster.centroid[1] - 500)**2)
            return distance

        def calDistanceCluster(cluster1, cluster2):
            distance =  np.sqrt(np.sum((np.array(cluster1.centroid) - np.array(cluster2.centroid))**2))
            return distance

        # def nearest_cluster_neighbor(cluster, listClusters, kdtree):
        #     nearest_neighbor_idx = kdtree.query_ball_point(cluster.centroid, calDistanceBS(cluster))
        #     if len(nearest_neighbor_idx) > 1:
        #         nearest_neighbor_idx.remove(listClusters.index(cluster))
        #     if nearest_neighbor_idx:
        #         nearest_neighbor = listClusters[nearest_neighbor_idx[0]]
        #         return nearest_neighbor
        #     else:
        #         return None
            

        def nearest_cluster_neighbor(cluster):
            min_distance_cluster = float('inf')
            nearest_neighbor = None
            for other_cluster in self.listClusters:
                if cluster != other_cluster and calDistanceBS(other_cluster) < calDistanceBS(cluster):
                    distance = calDistanceCluster(cluster, other_cluster)
                    if distance < min_distance_cluster:
                        min_distance_cluster = distance
                        nearest_neighbor = other_cluster
            return nearest_neighbor

        cluster_centroids = [cluster.centroid for cluster in self.listClusters]
        kdtree = cKDTree(cluster_centroids)

        edges = []
        for cluster in self.listClusters:
            nearest_neighbor = nearest_cluster_neighbor(cluster)
            if nearest_neighbor:
                if calDistanceCluster(cluster,nearest_neighbor) <  calDistanceBS(cluster):
                    edges.append((cluster, nearest_neighbor))
                else:
                    edges.append((cluster, self.baseStation))
            else:
                edges.append((cluster,self.baseStation))
        edge_colors = cycle([ 'g', 'b', 'y', 'c', 'm', 'k'])

        cluster_colors = ['g', 'b', 'y', 'c', 'm', 'pink', 'orange', 'purple', 
                        'brown', 'olive', 'teal', 'navy', 'maroon', 'lime', 'aqua', 'fuchsia',
                        'indigo', 'gold']
        # plt.figure(figsize=(10, 8))
        for i in range(len(self.listClusters)):
            cluster = self.listClusters[i]
            color = cluster_colors[i % len(cluster_colors)]  
            # Trích xuất các điểm và điểm centroid từ dữ liệu cluster
            points = [target.location for target in cluster.listTargets]
            centroid = cluster.centroid
            x_points = [point[0] for point in points]
            y_points = [point[1] for point in points]
            # Tạo mảng tọa độ x và y của điểm centroid
            centroid_x = centroid[0]
            centroid_y = centroid[1]
            # Vẽ các điểm trong cluster (trừ điểm centroid)
            # plt.scatter(x_points, y_points, color=color)
            # # Vẽ điểm centroid
            # plt.scatter(centroid_x, centroid_y, color='red')
        # Vẽ các cạnh giữa các cluster
        for edge, color in zip(edges, edge_colors):

            if edge[1] is self.baseStation:
                x_values = [edge[0].centroid[0], 500]
                y_values = [edge[0].centroid[1], 500]
                # plt.plot(x_values, y_values, color=color)
            else:
                # Trích xuất tọa độ của các điểm trong cạnh
                x_values = [edge[0].centroid[0], edge[1].centroid[0]]
                y_values = [edge[0].centroid[1], edge[1].centroid[1]]
                # plt.plot(x_values, y_values, color=color)
        # plt.scatter(500, 500, color='red', marker='*', s=300, label='Base Station')
        
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()
        
        # with open(edges.json, "w") as output_file:
        #     json.dump(edges, output_file)
        return edges
    
    def createNodeInCluster(self):
        # Input 
            # self.listClusters, self.listEdges

        # Todo
            # for cluster in self.listClusters:
            #     cluster.listNodes = [Node1,Node2 , . . . ]


        epsilon = 1e-6
        com_range =  self.phy['com_range']
        sen_range =  self.phy['sen_range']
        Cnt_in = [0] * (len(self.listClusters) + 1)
        Cnt_out= [0] * (len(self.listClusters) + 1)

        for edge in self.listEdges:
            if(edge[1].__class__.__name__ != "BaseStation"): 
             Cnt_in[edge[1].id] +=1
            Cnt_out[edge[0].id]+=1
            
        ID = 0
        for node in self.listNodes:
            if node.id > ID: 
                ID = node.id

        nodeInsideCluster = []
        for cluster in self.listClusters:
            id = cluster.id
            # Tạo InNode, OutNode
            phi = 2 * math.pi / (int) (Cnt_in[id] + Cnt_out[id] + 1)
            alpha = 0
            cnt = 0
            for i in range(0,Cnt_in[id] + Cnt_out[id]):
                X = cluster.centroid[0] + (com_range/2) * math.cos(alpha)
                Y = cluster.centroid[1] + (com_range/2) * math.sin(alpha)
                cnt +=1
                ID  +=1
                if(cnt<=Cnt_in[id]): 
                      cluster.listNodes.append(InNode([X,Y],ID,self.phy) )
                else: cluster.listNodes.append(OutNode([X,Y],ID,self.phy ))
                
                alpha += phi
            
            # sắp xếp InNode, OutNode theo khoảng cách đến BaseStation
            for i in range(0,len(cluster.listNodes)):
               for j in range(i+1,len(cluster.listNodes)):
                   if(cluster.listNodes[i].__class__.__name__ == "InNode" and cluster.listNodes[j].__class__.__name__ == "OutNode"):
                       distance_1 = euclidean(self.baseStation.location, cluster.listNodes[i].location)
                       distance_2 = euclidean(self.baseStation.location, cluster.listNodes[j].location)
                       if(distance_1 < distance_2): 
                          tmp = cluster.listNodes[j].location
                          cluster.listNodes[j].location = cluster.listNodes[i].location
                          cluster.listNodes[i].location = tmp
            
            # sắp xếp các Target giảm dần theo khảng cách đến Centroid
            for i in range (0,len(cluster.listTargets)):
                for j in range (i+1,len(cluster.listTargets)):
                       distance_1 = euclidean(cluster.centroid, cluster.listTargets[i].location)
                       distance_2 = euclidean(cluster.centroid, cluster.listTargets[j].location)
                       if(distance_1 < distance_2): 
                          tmp = cluster.listTargets[j].location
                          cluster.listTargets[j].location = cluster.listTargets[i].location
                          cluster.listTargets[i].location = tmp
            # Tạo SensorNode
            for target in cluster.listTargets:
                u , v = target.location
                check = 0
                for i in range(0,len(cluster.listNodes)):
                    if(cluster.listNodes[i].__class__.__name__ == "SensorNode"):
                      distance  = euclidean(cluster.listNodes[i].location,target.location)
                      if(distance <= sen_range): check = 1
                if(check): continue 
                u , v = point_between(target.location,cluster.centroid,sen_range-epsilon)
                ID += 1
                if(u != cluster.centroid[0] and v != cluster.centroid[1]):
                    cluster.listNodes.append(SensorNode([u,v],ID,self.phy))
                U = cluster.centroid[0]
                V = cluster.centroid[1]
                min_distance = 100000007
                for i in range(0,len(cluster.listNodes)):
                    Ok = 0
                    if(Cnt_in[id] == 0 and cluster.listNodes[i].__class__.__name__ =="OutNode"): Ok = 1
                    if(cluster.listNodes[i].__class__.__name__ =="ConnectorNode" or cluster.listNodes[i].__class__.__name__ =="InNode"): Ok = 1
                    if(Ok):
                      x = cluster.listNodes[i].location[0]
                      y = cluster.listNodes[i].location[1]
                      distance = math.sqrt((x-u)**2 + (y-v)**2) 
                      if(distance < min_distance):
                         min_distance = distance
                         U = x 
                         V = y 
                # Tạo ConnectorNode
                while True:
                    distance = math.sqrt((U-u)**2 + (V-v)**2) 
                    if(distance < com_range ): break
                    if(distance < 2*com_range):
                        U, V = point_between ( [U,V], [u,v] , distance/2 - epsilon)
                        ID += 1
                        cluster.listNodes.append(ConnectorNode([U,V],ID,self.phy))
                        break
                    U, V = point_between ( [U,V], [u,v] , com_range - epsilon)
                    ID += 1
                    cluster.listNodes.append(ConnectorNode([U,V],ID,self.phy))

            nodeInsideCluster = nodeInsideCluster + cluster.listNodes  
       
        return nodeInsideCluster   

    def createNodeBetweenCluster(self):
        # input 
            
        # Input 
            # self.listEdges

        # Todo
            # add start Id and End Id for relayNode

        # Output
            # [relayNode1,relayNode2 , . . . ]
        epsilon = 1e-6
        ListRelayNode = []
        range = self.phy['com_range'] * self.Alpha
        ID = 0
        for node in self.listNodes:
            if node.id > ID: 
                ID = node.id
                
        Cnt_in = [0] * (len(self.listClusters) + 1)
        Cnt_out =[0] * (len(self.listClusters) + 1)
        list_edge = []
        for edge in self.listEdges:
            if edge[1].__class__.__name__ == "Cluster":
                  list_edge.append((self.listClusters[edge[0].id],self.listClusters[edge[1].id]))
            else: list_edge.append((self.listClusters[edge[0].id],self.baseStation))
        
        for edge in list_edge:
            u = edge[0]
            v = edge[1]
            U = 0
            V = self.baseStation.location
            cnt = 0
            for node in u.listNodes:
                if(node.__class__.__name__ == "OutNode"):
                      if(cnt == Cnt_out[u.id]):
                          U = node.location.copy()
                          Cnt_out[u.id] += 1
                          break
                      cnt += 1
            cnt = 0
            if v.__class__.__name__ == "Cluster":
             for node in v.listNodes:
                if(node.__class__.__name__ == "InNode"):
                      if(cnt == Cnt_in[v.id]):
                          V = node.location.copy()
                          Cnt_in[v.id] += 1 
                          break
                      cnt += 1   
            while True:
                    distance = euclidean(U,V)
                    if(distance < range ): 
                        break
                    if(distance < 2*range):
                        U[0], U[1] = point_between ( U, V , distance/2 - epsilon)
                        ID += 1
                        ListRelayNode.append(RelayNode([U[0],U[1]],ID,self.phy,u,v))
                        break
                    U[0], U[1] = point_between ( U, V , range - epsilon)
                    ID += 1
                    ListRelayNode.append(RelayNode([U[0],U[1]],ID,self.phy,u,v))

        return ListRelayNode

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
                #print(node.location[0],node.location[1])
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
                #for i in range(len(self.targets_active)):
                    #if(self.targets_active[i] == 0):
                        #print(i,self.targets_active[i])
                print("die")
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


