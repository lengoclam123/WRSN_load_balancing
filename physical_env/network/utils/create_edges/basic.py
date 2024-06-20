import numpy as np

# Input 
    # [Cluster1,Cluster2 , . . . ]

# Output 
    # [(1,2),(3,2),(4,5) ,  . . .]
    # [(cluster1, cluster2), (cluster, base station),  . . . ]
def createEdges(net):

        def cal_distance(point1, point2):
            return np.sqrt((point1[1]-point2[1])**2+(point1[2]-point2[2])**2)

        def list_nearest_centroid(centroid_x,centroid_y, list_centroid):
            sorted_points = sorted(list_centroid, key=lambda x: ((x[1] - centroid_x) ** 2 + (x[2] - centroid_y) ** 2) ** 0.5)
            closest_point_ids = [point[0] for point in sorted_points]
            return closest_point_ids
        
        def angle_between_points(point1, point2):
            A= np.array([point1[1],point1[2]])
            B= np.array([point2[1],point2[2]])
            C= np.array(net.baseStation.location)
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
        para = len(net.listTargets)/5
        print(para)
        print(len(net.listTargets))
        for cluster in net.listClusters:
            list_id_centroid_cluster.append((cluster.id,cluster.centroid[0],cluster.centroid[1]))

        sorted_points = sorted(list_id_centroid_cluster, key=lambda point: cal_distance(point, bs), reverse=True)
        sorted_ids_distance = [point[0] for point in sorted_points]
        print("target",len(net.listTargets))
        print("target",net.listTargets)
        print("target")

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

        # Số target mà cụm phải tải ban đầu
        num_targets_clusters=[]
        for id in sorted_ids_distance:
            for cluster in net.listClusters:
                if id == cluster.id:
                    num_targets_clusters.append([cluster.id,len(cluster.listTargets)])

        # Chọn cạnh
        for point in list_common_elements:
            id =point[0]
            common_points = point[1]
            for targets_cluster1 in num_targets_clusters:
                if id == targets_cluster1[0]:
                    num_targets_point = targets_cluster1[1]
             
            if common_points:
                found = False
                for common_point in common_points:
                    for targets_cluster in num_targets_clusters:
                        if targets_cluster[0]==common_point:
                            if num_targets_point + targets_cluster[1] < para:
                                edges_id.append([id,common_point])
                                targets_cluster[1]=targets_cluster[1]+num_targets_point
                                found=True
                                break
                    if found:
                        break
                if not found:
                    edges_id.append([id,-1]) 
            else:
                for cluster in net.listClusters:
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
            for cluster in net.listClusters:
                if cluster.id == edge[0]:
                    cluster1 = cluster  
                if cluster.id == edge[1]:
                    cluster2 = cluster  
                if edge[1] == -1:
                    cluster2 = net.baseStation 
            edges.append((cluster1,cluster2))
        return edges
