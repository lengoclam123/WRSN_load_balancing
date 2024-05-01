import copy
import numpy as np
import Cluster
import math
from itertools import permutations
from InNode import InNode
from OutNode import OutNode
from RelayNode import RelayNode
from SensorNode import SensorNode 

def point_between(a, b, r):

    d = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    if(d<r):
        return b[0], b[1]
    cx = a[0] + r * (b[0] - a[0]) / d
    cy = a[1] + r * (b[1] - a[1]) / d

    return cx, cy

class Network:
    def __init__(self, env , listNodes , baseStation , listTargets , max_time ):
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


        #self.createNodes()

    
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

        pass

    def createNodeInCluster(self):
        # Input 
            # self.listClusters, self.listEdges

        # Todo
            # for cluster in self.listClusters:
            #     cluster.listNodes = [Node1,Node2 , . . . ]

        # for i in range (0,len(listEdges)): print(i,listEdges[i])  
        com_range =  80
        sen_range =  40
        Cnt_in = [0] * (len(self.listClusters) + 1)
        Cnt_out= [0] * (len(self.listClusters) + 1)
        for edge in self.listEdges:
            Cnt_in[edge[1]] +=1
            Cnt_out[edge[0]]+=1
        ID = 0
        nodeInsideCluster = []
        for cluster in self.listClusters:
            id = cluster.id
            phi = 2 * math.pi / (int) (Cnt_in[id] + Cnt_out[id] + 1)
            alpha = 0
            cnt = 0
            for i in range(0,Cnt_in[id] + Cnt_out[id]):
                X = cluster.centroid[0] + (com_range/2) * math.cos(alpha)
                Y = cluster.centroid[1] + (com_range/2) * math.sin(alpha)
                cnt +=1
                ID  +=1
                if(cnt<=Cnt_in[id]): 
                      cluster.listNodes.append(InNode([X,Y],ID))
                else: cluster.listNodes.append(OutNode([X,Y],ID))
                
                alpha += phi

            for i in range(0,len(cluster.listNodes)):
               for j in range(i+1,len(cluster.listNodes)):
                   if(cluster.listNodes[i].__class__.__name__ == "InNode" and cluster.listNodes[j].__class__.__name__ == "OutNode"):
                       x1 = cluster.listNodes[i].location[0]
                       y1 = cluster.listNodes[i].location[1]
                       x2 = cluster.listNodes[j].location[0]
                       y2 = cluster.listNodes[j].location[1]
                       u  = self.baseStation.location[0]
                       v  = self.baseStation.location[1]
                       distance_1 = math.sqrt((x1-u)**2 + (y1-v)**2)
                       distance_2 = math.sqrt((x2-u)**2 + (y2-v)**2)
                       if(distance_1 > distance_2): 
                          tmp = cluster.listNodes[j].location
                          cluster.listNodes[j].location = cluster.listNodes[i].location
                          cluster.listNodes[i].location = tmp

            for i in range (0,len(cluster.listTargets)):
                for j in range (i+1,len(cluster.listTargets)):
                       x1 = cluster.listTargets[i].location[0]
                       y1 = cluster.listTargets[i].location[1]
                       x2 = cluster.listTargets[j].location[0]
                       y2 = cluster.listTargets[j].location[1]
                       u  = cluster.centroid[0]
                       v  = cluster.centroid[1]
                       distance_1 = math.sqrt((x1-u)**2 + (y1-v)**2)
                       distance_2 = math.sqrt((x2-u)**2 + (y2-v)**2)
                       if(distance_1 > distance_2): 
                          tmp = cluster.listTargets[j]
                          cluster.listTargets[j] = cluster.listTargets[i]
                          cluster.listTargets[i] = tmp

            for target in cluster.listTargets:
                u = target.location[0]
                v = target.location[1]
                check = 0
                for i in range(0,len(cluster.listNodes)):
                    if(cluster.listNodes[i].__class__.__name__ == "SensorNode"):
                      x = cluster.listNodes[i].location[0]
                      y = cluster.listNodes[i].location[1]
                      distance  = math.sqrt((x-u)**2 + (y-v)**2)
                      if(distance <= sen_range): check = 1
                if(check): continue 
                u , v = point_between(target.location,cluster.centroid,sen_range)
                ID += 1
                if(u != cluster.centroid[0] and v != cluster.centroid[1]):
                    cluster.listNodes.append(SensorNode([u,v],ID))
                U = cluster.centroid[0]
                V = cluster.centroid[1]
                min_distance = 100000007
                for i in range(0,len(cluster.listNodes)):
                    if(cluster.listNodes[i].__class__.__name__ =="RalayNode" or cluster.listNodes[i].__class__.__name__ =="InNode"):
                      x = cluster.listNodes[i].location[0]
                      y = cluster.listNodes[i].location[1]
                      distance = math.sqrt((x-u)**2 + (y-v)**2) 
                      if(distance < min_distance):
                         min_distance = distance
                         U = x 
                         V = y 
                while True:
                    distance = math.sqrt((U-u)**2 + (V-v)**2) 
                    if(distance < com_range ): break
                    if(distance < 2*com_range):
                        U, V = point_between ( [U,V], (u,v) , distance/2)
                        ID += 1
                        cluster.listNodes.append(RelayNode([U,V],ID))
                        break
                    U, V = point_between ( [U,V], (u,v) , com_range)
                    ID += 1
                    cluster.listNodes.append(RelayNode([U,V],ID))

            nodeInsideCluster = nodeInsideCluster + cluster.listNodes  
        cntt = 0
        for node in nodeInsideCluster:
            if(node.__class__.__name__ == "RelayNode"): cntt+=1
        print(cntt)  
        return nodeInsideCluster    

    def createNodeBetweenCluster(self):
        
        # Input 
            # self.listEdges

        # Todo
            # add start Id and End Id for relayNode

        # Output
            # [relayNode1,relayNode2 , . . . ]
        ListRelayNode = []
        com_range = 80
        ID = 0
        Cnt_in = [0] * (len(self.listClusters) + 1)
        Cnt_out =[0] * (len(self.listClusters) + 1)
        for edge in self.listEdges:
            u = edge[0]
            v = edge[1]
            U = 0
            V = 0
            for cluster in self.listClusters: 
                if(cluster.id == u):
                  cnt = 0
                  for node in cluster.listNodes:
                    if(node.__class__.__name__ == "OutNode"):
                      if(cnt == Cnt_out[u]):
                          U = node.location
                          Cnt_out[u] += 1
                          break
                      cnt += 1
                      
                if(cluster.id == v):
                  cnt = 0
                  for node in cluster.listNodes:
                    if(node.__class__.__name__ == "InNode"):
                      if(cnt == Cnt_in[v]):
                          V = node.location
                          Cnt_in[v] += 1 
                          break
                      cnt += 1   
            while True:
                    distance = math.sqrt((U[0]-V[0])**2 + (U[1]-V[1])**2) 
                    if(distance < com_range ): break
                    if(distance < 2*com_range):
                        U[0], U[1] = point_between ( U, V , distance/2)
                        ID += 1
                        ListRelayNode.append(RelayNode(U,ID))
                        break
                    U[0], U[1] = point_between ( U, V , com_range)
                    ID += 1
                    ListRelayNode.append(RelayNode([U[0],U[1]],ID))     
        return ListRelayNode    
        
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