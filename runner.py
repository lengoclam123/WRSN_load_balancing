import json 
import sys
import os
import matplotlib.pyplot as plt
from Cluster import Cluster 
from Target import Target
sys.path.append(os.path.dirname(__file__))
from Network import Network
from BaseStation import BaseStation
with open('edges.json', 'r') as f:
    data = json.load(f)
listClusters = [0]*2000
listEdges = [0]*len(data)
for i in range (0,len(data)): 
    u = data[i][0]["cluster_id"] 
    v = data[i][1]["cluster_id"]
    tmp = []
    for j in range (0,len(data[i][0]["listTargets"])):
        target = Target( data[i][0]["listTargets"][j]["location"], data[i][0]["listTargets"][j]["target_id"])
        tmp.append(target)
    listClusters [u] = Cluster (tmp)
    listClusters [u].centroid = data[i][0]["centroid"]
    listClusters [u].id = data[i][0]["cluster_id"]


    tmp = []
    for j in range (0,len(data[i][1]["listTargets"])):
        target = Target( data[i][1]["listTargets"][j]["location"], data[i][1]["listTargets"][j]["target_id"])
        tmp.append(target)
    listClusters [v] = Cluster (tmp)
    listClusters [v].centroid = data[i][1]["centroid"]
    listClusters [v].id = data[i][1]["cluster_id"]

    listEdges[i] = [u,v]
for i in range(0,len(listClusters)):
    if (listClusters[i] == 0 ):
       listClusters = listClusters [:i]
       break
  
net = Network(env = [], listNodes = [], baseStation = BaseStation([500,500]) , listTargets = [],max_time = 60000)
net.listClusters = listClusters 
net.listEdges = listEdges 
net.createNodeInCluster()
plt.figure() 
plt.scatter(net.baseStation.location[0], net.baseStation.location[1], marker= "*", color = "purple",s = 200)
for j in range(0,len(net.listClusters)):
 plt.scatter(net.listClusters[j].centroid[0],net.listClusters[j].centroid[1],color = "pink")
 for i in range(0,len(net.listClusters[j].listNodes)):
    x = net.listClusters[j].listNodes[i].location[0]
    y = net.listClusters[j].listNodes[i].location[1]
    z = net.listClusters[j].listNodes[i].__class__.__name__
    if(z == "InNode"): z = "red"
    if(z == "OutNode"): z = "blue"
    if(z == "SensorNode"): z = "green"
    if(z == "RelayNode"):  z = "yellow"
    plt.scatter(x, y, color = z)
 for i in range(0,len(net.listClusters[j].listTargets)):
    x = net.listClusters[j].listTargets[i].location[0]
    y = net.listClusters[j].listTargets[i].location[1]
    plt.scatter(x, y, color = "black")

test_list = net.createNodeBetweenCluster()
for node in test_list:
   x = node.location[0]
   y = node.location[1]
   plt.scatter(x, y, color = "orange")
plt.show()
