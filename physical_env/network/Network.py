
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import math
from scipy.signal import find_peaks
import json
import copy
from sklearn.metrics import silhouette_score,silhouette_samples


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

        def check_valid_cluster(clusterer, data, n_clusters, n_targets):
            # mảng các label, mỗi label cluster là mảng các node
            # print(n_clusters)
            cluster_labels = clusterer.fit_predict(data)
            centers = clusterer.cluster_centers_
            cluster = [[] for _ in range(n_clusters)]
            # print(cluster)
            for index_node in range(0, n_targets):
                # print(cluster_labels[index_node])
                # print(data[index_node])
                cluster[cluster_labels[index_node]].append(data[index_node])
                # print(cluster)
            for index_cluster in range(0, n_clusters):
                for target in cluster[index_cluster]: 
                    #
                    if math.dist(target, centers[index_cluster]) > 140: 
                        return False    
            return True

        # Input : listTargets
        list_targets_location = []
        for target in self.listTargets:
            list_targets_location.append(target.location)
        list_targets_location = np.array(list_targets_location)
        # print(list_targets_location)
        range_n_clusters = np.arange(2, int(len(list_targets_location)/2))
        
        max_silhouette_score = 0
        n_clusters_optimal = 0

        for n_clusters in range_n_clusters:
            # clusterer = KMeans(n_clusters=n_clusters, random_state=10, n_init="auto", init="k-means++",  algorithm='elkan')
            clusterer = KMeans(n_clusters=n_clusters, init="k-means++",  algorithm='elkan')

            cluster_labels = clusterer.fit_predict(list_targets_location)
            silhouette_avg = silhouette_score(list_targets_location, cluster_labels)

            # print(
            #     "For n_clusters_optimal =",
            #     n_clusters,
            #     "The max silhouette",
            #     max_silhouette_score,
            #     "The optimal K",
            #     n_clusters_optimal,
            #     "The average silhouette_score is :",
            #     silhouette_avg,
            # )

            if check_valid_cluster(clusterer, list_targets_location, n_clusters, len(list_targets_location)) is True:
                if max_silhouette_score < silhouette_avg:
                    max_silhouette_score = silhouette_avg
                    n_clusters_optimal = n_clusters

        clusterer = KMeans(n_clusters=n_clusters, init="k-means++",  algorithm='elkan')
        # clusterer = KMeans(n_clusters=n_clusters_optimal, random_state=None,  n_init="auto", init="k-means++", algorithm='lloyd')

        cluster_labels = clusterer.fit_predict(list_targets_location)
        silhouette_avg = silhouette_score(list_targets_location, cluster_labels)
        self.n_clusters_optimal = n_clusters_optimal
    
        # Labeling the clusters
        centers = clusterer.cluster_centers_

        clusters = []
        
        for i in range(0, n_clusters_optimal):
            listTargetsInCluster = []
            for j in range(0, len(list_targets_location)):
                if cluster_labels[j] == i:
                    listTargetsInCluster.append(self.listTargets[j])
            cluster = Cluster(i, listTargetsInCluster, centers[i])
            clusters.append(cluster)

        # Chuyển đổi danh sách thành danh sách các từ điển
        json_data = [convert_cluster_to_dict(cluster) for cluster in clusters]
        # Chuyển đổi danh sách thành chuỗi JSON
        json_string = json.dumps(json_data, indent=4)
        with open("clusters.json", "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        return clusters

    def createEdges(self):
        # xét từng centroid
            # tạo ra 3 danh sách s1 s2 s3
            # giao của 3 danh sách là 1 mảng chứa các id của cluster(danh sách các output )
            # diểm 

        # Input 
            # [Cluster1,Cluster2 , . . . ]

        # Output 
            # [(1,2),(3,2),(4,5) ,  . . .]
            # [(cluster1, cluster2), (cluster, base station),  . . . ]
            # funs for create edges
        # def calDistanceBS(cluster):
        #     distance =  np.sqrt((cluster.centroid[0] - 500)**2 + (cluster.centroid[1] - 500)**2)
        #     return distance

        # def calDistanceCluster(cluster1, cluster2):
        #     distance =  np.sqrt(np.sum((np.array(cluster1.centroid) - np.array(cluster2.centroid))**2))
        #     return distance

        # def nearest_cluster_neighbor(cluster):
        #     min_distance_cluster = float('inf')
        #     nearest_neighbor = None
        #     for other_cluster in self.listClusters:
        #         if cluster != other_cluster and calDistanceBS(other_cluster) < calDistanceBS(cluster):
        #             distance = calDistanceCluster(cluster, other_cluster)
        #             if distance < min_distance_cluster:
        #                 min_distance_cluster = distance
        #                 nearest_neighbor = other_cluster
        #     return nearest_neighbor

        # cluster_centroids = [cluster.centroid for cluster in self.listClusters]
        # kdtree = cKDTree(cluster_centroids)

        # edges = []
        # for cluster in self.listClusters:
        #     nearest_neighbor = nearest_cluster_neighbor(cluster)
        #     if nearest_neighbor:
        #         if calDistanceCluster(cluster,nearest_neighbor) <  calDistanceBS(cluster):
        #             edges.append((cluster, nearest_neighbor))
        #         else:
        #             edges.append((cluster, self.baseStation))
        #     else:
        #         edges.append((cluster,self.baseStation))

        def cal_distance(point1, point2):
            return np.sqrt((point1[1]-point2[1])**2+(point1[2]-point2[2])**2)

        def list_nearest_centroid(centroid_x,centroid_y, list_centroid):
            sorted_points = sorted(list_centroid, key=lambda x: ((x[1] - centroid_x) ** 2 + (x[2] - centroid_y) ** 2) ** 0.5)
            # Chọn ra các id của các điểm gần nhất
            closest_point_ids = [point[0] for point in sorted_points]

            # print(closest_point_ids)
            return closest_point_ids
        
        def angle_between_points(point1, point2):
            A= np.array([point1[1],point1[2]])
            B= np.array([point2[1],point2[2]])
            C= np.array(self.baseStation.location)
            vector_AB = B - A
            vector_AC = C - A
            angle_BAC = np.arctan2(np.linalg.det([vector_AB, vector_AC]), np.dot(vector_AB, vector_AC))
            angle_BAC_deg = np.degrees(angle_BAC)
            return angle_BAC_deg

        list_id_centroid_cluster = []
        bs = [-1,500,500]
        edges = []
        edges_id = []
        list_common_elements = []
        for cluster in self.listClusters:
            list_id_centroid_cluster.append((cluster.id,cluster.centroid[0],cluster.centroid[1]))
        # print(list_id_centroid_cluster)
        # sorted_points = sorted(list_id_centroid_cluster, key=lambda point: cal_distance(point,bs))
        sorted_points = sorted(list_id_centroid_cluster, key=lambda point: cal_distance(point, bs), reverse=True)
        sorted_ids_distance = [point[0] for point in sorted_points]
        
        for point in sorted_points:
            list_angle_satisfy= []
            list_point_distance_satisfy = []
            sorted_nearest_point = list_nearest_centroid(point[1],point[2],list_id_centroid_cluster)

            for other_point in list_id_centroid_cluster:
                if point != other_point and angle_between_points(point,other_point)<= 45 and angle_between_points(point,other_point)>= -45:
                    list_angle_satisfy.append(other_point[0])

            for other_point in list_id_centroid_cluster:
                if point != other_point:
                    if cal_distance(point,other_point)< cal_distance(point,bs) and cal_distance(point,bs) > cal_distance(other_point,bs):
                        list_point_distance_satisfy.append(other_point[0])
            
             
            common_elements = []
            for id in sorted_nearest_point:
                if id in list_angle_satisfy and id in list_point_distance_satisfy:
                    common_elements.append(id)
            list_common_elements.append([point[0],common_elements])

            #  phần trên tạo cạnh với một đầu ra 
            # if common_elements:
            #     for cluster in self.listClusters:
            #         if cluster.id == point[0]:
            #             cluster1 = cluster
            #     for cluster in self.listClusters:
            #         if cluster.id == common_elements[0]:
            #             # edges.append((cluster1,cluster))
            #             edges_id.append([cluster1.id,cluster.id])
            #     common_elements.pop(0)
            # else: 
            #     for cluster in self.listClusters:
            #         if cluster.id == point[0]:
            #             # edges.append((cluster,self.baseStation))
            #             edges_id.append([cluster.id,-1])
            # common_elements_2.append((id,common_elements))
        
        print(list_common_elements)

        # Số target mà cụm phải tải ban đầu
        num_targets_clusters=[]
        for id in sorted_ids_distance:
            for cluster in self.listClusters:
                if id == cluster.id:
                    num_targets_clusters.append([cluster.id,len(cluster.listTargets)])
        print(num_targets_clusters)

        # Chọn cạnh
        for point in list_common_elements:
            id =point[0]
            common_points = point[1]
            for targets_cluster1 in num_targets_clusters:
                if id == targets_cluster1[0]:
                    num_targets_point = targets_cluster1[1]
             
            if common_points:
                num = 0
                found = False
                for common_point in common_points:
                    for targets_cluster in num_targets_clusters:
                        if targets_cluster[0]==common_point:
                            if num_targets_point + targets_cluster[1] < 40:
                                edges_id.append([id,common_point])
                                targets_cluster[1]=targets_cluster[1]+num_targets_point
                                found=True
                                break
                    if found:
                        break
                if not found:
                    edges_id.append([id,-1]) 
            else:
                for cluster in self.listClusters:
                    if cluster.id == id:
                        edges_id.append((id,-1))




        for targets_cluster in num_targets_clusters:
            for edge in edges_id:
                if targets_cluster[0] == edge[1]:
                    for add_targets in num_targets_clusters:
                        if add_targets[0] == edge[0]:
                            targets_cluster[1] = targets_cluster[1]+add_targets[1]
                            break
                    
        edges = []
        for edge in edges_id:
            for cluster in self.listClusters:
                if cluster.id == edge[0]:
                    cluster1 = cluster  
                if cluster.id == edge[1]:
                    cluster2 = cluster  
                if edge[1] == -1:
                    cluster2 = self.baseStation 
            edges.append((cluster1,cluster2))
        print(num_targets_clusters)
    
        # Vẽ 
        edge_colors = cycle([ 'g', 'b', 'y', 'c', 'm', 'k'])

        cluster_colors = ['g', 'b', 'y', 'c', 'm', 'pink', 'orange', 'purple', 
                        'brown', 'olive', 'teal', 'navy', 'maroon', 'lime', 'aqua', 'fuchsia',
                        'indigo', 'gold']
        plt.figure(figsize=(10, 10))
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
            plt.scatter(x_points, y_points, color=color)
            # Vẽ điểm centroid
            plt.scatter(centroid_x, centroid_y, color='red')
        # Vẽ các cạnh giữa các cluster
        for edge, color in zip(edges, edge_colors):

            if edge[1] is self.baseStation:
                x_values = [edge[0].centroid[0], 500]
                y_values = [edge[0].centroid[1], 500]
                plt.plot(x_values, y_values, color=color)
            else:
                # Trích xuất tọa độ của các điểm trong cạnh
                x_values = [edge[0].centroid[0], edge[1].centroid[0]]
                y_values = [edge[0].centroid[1], edge[1].centroid[1]]
                plt.plot(x_values, y_values, color=color)
        plt.scatter(500, 500, color='red', marker='*', s=300, label='Base Station')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        # with open(edges.json, "w") as output_file:
        #     json.dump(edges, output_file)


        return edges
        pass

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

def convert_cluster_to_dict(cluster):
        return {
            'cluster_id': cluster.id,
            'listTargets': [target.__dict__ for target in cluster.listTargets],
            'centroid': cluster.centroid.tolist()  # Chuyển mảng numpy thành danh sách Python
        }
