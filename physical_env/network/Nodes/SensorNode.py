from Node import Node

class SensorNode(Node):
    def __init__(self, location ,id):
       self.location = location 
       self.id = id
    def find_receiver(self): # define outnode
        pass