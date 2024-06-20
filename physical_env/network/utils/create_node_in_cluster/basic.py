from scipy.spatial.distance import euclidean
import math

from Nodes.InNode import InNode
from Nodes.OutNode import OutNode
from Nodes.ConnectorNode import ConnectorNode
from Nodes.SensorNode import SensorNode

        # Input 
            # net.listClusters, net.listEdges

        # Todo
            # for cluster in net.listClusters:
            #     cluster.listNodes = [Node1,Node2 , . . . ]

def createNodeInCluster(net):

        com_range =  net.phy['com_range']*0.9
        print(com_range)
        sen_range =  net.phy['sen_range']*0.9
        print(sen_range)
        Cnt_in = [0] * (len(net.listClusters) + 1)
        Cnt_out= [0] * (len(net.listClusters) + 1)

        for edge in net.listEdges:
            if(edge[1].__class__.__name__ != "BaseStation"): 
             Cnt_in[edge[1].id] +=1
            Cnt_out[edge[0].id]+=1
            
        ID = 0
        for node in net.listNodes:
            if node.id > ID: 
                ID = node.id

        nodeInsideCluster = []
        for cluster in net.listClusters:
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
                      cluster.listNodes.append(InNode([X,Y],ID,net.phy) )
                else: cluster.listNodes.append(OutNode([X,Y],ID,net.phy ))
                
                alpha += phi
            
            # sắp xếp InNode, OutNode theo khoảng cách đến BaseStation
            for i in range(0,len(cluster.listNodes)):
               for j in range(i+1,len(cluster.listNodes)):
                   if(cluster.listNodes[i].__class__.__name__ == "InNode" and cluster.listNodes[j].__class__.__name__ == "OutNode"):
                       distance_1 = euclidean(net.baseStation.location, cluster.listNodes[i].location)
                       distance_2 = euclidean(net.baseStation.location, cluster.listNodes[j].location)
                       if(distance_1 < distance_2): 
                          tmp = cluster.listNodes[j].location
                          cluster.listNodes[j].location = cluster.listNodes[i].location
                          cluster.listNodes[i].location = tmp

            nearest_node_targets = []
            for target in cluster.listTargets:
                min_dis = math.inf
                nearest_node = None
                for node in cluster.listNodes:
                    if euclidean(node.location, target.location) <= min_dis:
                        min_dis=euclidean(node.location, target.location)
                        nearest_node = node
                        active = 0
                nearest_node_targets.append([target,nearest_node,min_dis, active])

                #  sắp xếp mảng  nearest_node_targets:
            nearest_node_targets =  sorted(nearest_node_targets, key=lambda element: element[2], reverse=True)

            for element in nearest_node_targets:
                target_x , target_y = element[0].location
                nearest_node_x, nearest_node_y = element[1].location
                min_dis_node = element[2]
                beta = min_dis_node/sen_range
                if beta < 1:
                    element[3] = 1
                    continue
                else:
                    # Câu lệnh if check ở đây để kiểm tra xem cái target này đã được theo dõi chưa
                    delta_x = nearest_node_x -target_x 
                    delta_y = nearest_node_y -target_y
                    sensor_x = target_x + delta_x/beta
                    sensor_y = target_y + delta_y/beta
                    ID+=1
                    cluster.listNodes.append(SensorNode([sensor_x,sensor_y],ID,net.phy))
                    beta = element[2]/com_range
                    delta_x = nearest_node_x -sensor_x 
                    delta_y = nearest_node_y -sensor_y
                    for i in range(1,int(min_dis_node/com_range)):
                        x_new = sensor_x + i*delta_x/beta
                        y_new = sensor_y + i*delta_y/beta
                        ID+=1
                        cluster.listNodes.append(ConnectorNode([x_new,y_new],ID,net.phy))
                    
            nodeInsideCluster = nodeInsideCluster + cluster.listNodes  
             
        return nodeInsideCluster   

