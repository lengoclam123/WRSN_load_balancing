from Node import Node
from scipy.spatial.distance import euclidean
class ConnectorNode(Node):
    def __init__(self, location ,id , phy):
       super().__init__(location , phy)
       self.location = location 
       self.id = id
       self.cluster_id = 0
    def find_receiver(self): 
        distance_min = 10000007
        node_min = None
        for node in self.neighbors:
            name = node.__class__.__name__
            if(name == "ConnectorNode" or name == "InNode" or name == "OutNode"):
                distance =  euclidean(node.location, self.net.listClusters[self.cluster_id].centroid)
                if distance < distance_min:
                    node_min = node
                    distance_min = distance
        return node_min