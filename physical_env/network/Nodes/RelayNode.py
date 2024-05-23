from Node import Node
from scipy.spatial.distance import euclidean
class RelayNode(Node):
    def __init__(self, location ,id , phy,start,end):
       super().__init__(location , phy)
       self.location = location 
       self.id = id
       self.start = start
       self.end = end
    def find_receiver(self): # define outnode
        for node in self.neighbors:
            if(node.__class__.__name__ == "RelayNode"):
              if(self.start.id == node.start.id and self.end.id == node.end.id):
                Location_end = self.net.baseStation.location
                if self.end.__class__.__name__ == "BaseStation":
                       Location_end = self.end.location
                else: Location_end = self.end.centroid
                
                distance_1 =  euclidean(node.location, Location_end)
                distance_2 =  euclidean(self.location, Location_end)
                if distance_1 < distance_2:
                    return node
        for node in self.neighbors:
            if(node.__class__.__name__ == "InNode"):
                if(node.cluster_id == self.end.id):
                 return node
        pass