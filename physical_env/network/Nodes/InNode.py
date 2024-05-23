# import Node
from .Node import Node

class InNode(Node):
    def __init__(self, location ,id , phy):
       super().__init__(location , phy)
       self.location = location 
       self.id = id
       self.cluster_id = 0
    def find_receiver(self): # define outnode
        for node in self.neighbors: 
            if(node.__class__.__name__ == "OutNode"):
                return node
        pass