from Node import Node

class OutNode(Node):
    def __init__(self, location ,id , phy):
       super().__init__(location , phy)
       self.location = location 
       self.id = id
       self.cluster_id = 0
    def find_receiver(self): # define outnode
        for node in self.neighbors:
            if(node.__class__.__name__ == "RelayNode") and self.level > node.level:
               if(node.start.id == self.cluster_id): 
                return node
        pass